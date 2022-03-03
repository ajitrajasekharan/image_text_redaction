from paddleocr import PaddleOCR,draw_ocr
import pdb
import PIL
from PIL import ImageDraw
from PIL import Image
import traceback
from paddleocr import PaddleOCR,draw_ocr
import OCRWrapper as ocr
import sys,os
import argparse
import json
import SnapToLineGrid as sn
import RegionsCropper as cropper
import Redactor as rd
from collections import OrderedDict

class DeIdentify:
    def __init__(self,device):
        self.easy_ocr =  ocr.OCRWrapper(device,ocr_model = ocr.EASY_OCR)
        self.tr_ocr_hw =  ocr.OCRWrapper(device,ocr_model = ocr.TR_OCR_HW)
        self.tr_ocr_print =  ocr.OCRWrapper(device,ocr_model = ocr.TR_OCR_PRINT)
        self.paddle_ocr =  ocr.OCRWrapper(device,ocr_model = ocr.PADDLE_OCR)
        self.snap = sn.SnapToLineGrid()
        self.cropper = cropper.RegionsCropper()
        self.redactor = rd.Redactor()

    def convert_if_needed(self,input_file):
        if (input_file.endswith(".png")):
            im1 = Image.open(input_file).convert('RGB')
            conv_file = ''.join(input_file.split(".")[:-1]) + ".jpg"
            im1.save(conv_file)
        else:
            conv_file = input_file
        return conv_file

    def convert_confidences_to_str(self,predictions):
        for key in predictions:
            node = predictions[key]
            for i in range(len(node)):
                node[i]["conf"] = str(node[i]["conf"])

    def extract_text(self,params):
        mode = params.mode
        input_file_or_dir = params.input
        results_file = params.results
        detailed_results_file = params.full_results
        wfp = open(results_file,"w")
        detailed_wfp = open(detailed_results_file,"w")
        if (mode == "single"):
            self.extract_text_individual(input_file_or_dir,wfp,detailed_wfp)
        else:
            for filename in os.listdir(input_file_or_dir):
                f = os.path.join(input_file_or_dir, filename)
                if os.path.isfile(f) and (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")):
                    print(f)
                    self.extract_text_individual(f,wfp,detailed_wfp)
        wfp.close()
        detailed_wfp.close()


    def extract_text_individual(self,input_file,wfp,detailed_wfp):
        predictions = OrderedDict()
        input_file = self.convert_if_needed(input_file) #convers to jpg if in png format
        img = PIL.Image.open(input_file)
        easy_ocr_predictions = {}

        try:
            easy_ocr_predictions = self.easy_ocr.detect([img]) #this returns bbox and predictions - works for print - not hand written
        except:
            print("Exception in Easy OCR :", sys.exc_info()[0])
            traceback.print_exc(file=sys.stdout)

        easy_ocr_predictions_coalesced = self.snap.to_lines(easy_ocr_predictions)   #coalesce fragmented bboxes to help improve recognition for 
                                                                                    #TR OCR model that only recognizes - not detects. So coalesced bbox improve prediction accuracy
        self.debug_print("easy_ocr",input_file,easy_ocr_predictions_coalesced,wfp) 
        original_regions = self.cropper.crop(img,easy_ocr_predictions,"original") 
        coalesced_regions = self.cropper.crop(img,easy_ocr_predictions_coalesced,"coalesced") #Create cropped regions for TR OCR model to predict
        predictions["easy_ocr"] = easy_ocr_predictions_coalesced

         
        #The coalesced text regions from easy ocr is used for TR models below since they dont have region detection capability
        tr_ocr_hw_predictions = self.tr_ocr_hw.detect(coalesced_regions,easy_ocr_predictions_coalesced) #TR OCE takes coalesced regions
        self.debug_print("tr_ocr_hw",input_file,tr_ocr_hw_predictions,wfp)
        redacted_img = self.redactor.redact(tr_ocr_hw_predictions,img,prefix_name="redacted_" + ''.join(input_file.split('/')[-1]))
        predictions["tr_ocr_hw"] =  tr_ocr_hw_predictions

        tr_ocr_print_predictions = self.tr_ocr_print.detect(coalesced_regions,easy_ocr_predictions_coalesced) #TR OCE takes coalesced regions
        self.debug_print("tr_ocr_print",input_file,tr_ocr_print_predictions,wfp)
        predictions["tr_ocr_print"] =  tr_ocr_print_predictions

        paddle_predictions = self.paddle_ocr.detect([input_file]) #paddle take image path
        paddle_predictions_coalesced = self.snap.to_lines(paddle_predictions)   #coalesce fragmented bboxes to help improve recognition for 
        predictions["paddle_ocr"] = paddle_predictions_coalesced
        self.debug_print("paddle_ocr",input_file,paddle_predictions_coalesced,wfp)
        #redacted_img = self.redactor.redact(paddle_predictions_coalesced,img,prefix_name="redacted_" + ''.join(input_file.split('/')[-1]))

        self.convert_confidences_to_str(predictions)
        json_object = json.dumps(predictions, indent = 4)
        #print(json_object)
        detailed_wfp.write(json_object + "\n")


    def debug_print(self,model_name,input_file,predictions,wfp):
        write_str = "\nInput: {input_file}\nPredictions from model:{model_name}".format(model_name=model_name,input_file=input_file)
        print(write_str)
        wfp.write(write_str + "\n")
        for node in predictions:
            #print(node["prediction"],node["left"],node["right"],node["top"],node["bottom"])
            write_str = "Pred: {prediction}   Conf: {conf} Time(in secs): {time_val}".format(prediction = node["prediction"],conf = round(node["conf"],2),time_val = node["inf_time"])
            print(write_str)
            wfp.write(write_str + "\n")
        #pdb.set_trace()


         


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeIdentify images - prototype ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', dest="input", action='store',help='Input image in jpg/png format if mode is single; else considered as directory')
    parser.add_argument('-device', dest="device", action='store',default="cpu",help='Input image in jpg/png format')
    parser.add_argument('-mode', dest="mode", action='store',default="single",help='Mode could be single or batch')
    parser.add_argument('-results', dest="results", action='store',default="results.txt",help='file containing predictions - brief format')
    parser.add_argument('-full_results', dest="full_results", action='store',default="full_results.txt",help='file containing predictions - detailed format')

    results = parser.parse_args()
    try:
        #pdb.set_trace()
        #mocr = PaddleOCR(det_model_dir='det_model', rec_model_dir='rec_model',  cls_model_dir='angle_model', use_angle_cls=True,use_gpu=True,show_log=False,enable_mkldnn=True)
        #result = mocr.ocr("../test_images/slide1.jpg",rec=True,det=True,cls=True)
        main_obj = DeIdentify(results.device)
        main_obj.extract_text(results)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
