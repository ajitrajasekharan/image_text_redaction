import pdb
import PIL
from PIL import ImageDraw
from PIL import Image
import OCRWrapper as ocr
import sys
import argparse
import json
import SnapToLineGrid as sn
import RegionsCropper as cropper
import Redactor as rd
from collections import OrderedDict

class DeIdentify:
    def __init__(self,device):
        self.easy_ocr =  ocr.OCRWrapper(device,ocr_model = ocr.EASY_OCR)
        self.tr_ocr =  ocr.OCRWrapper(device,ocr_model = ocr.TR_OCR)
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

    def extract_text(self,input_file):
        predictions = OrderedDict()
        input_file = self.convert_if_needed(input_file) #convers to jpg if in png format
        img = PIL.Image.open(input_file)
        easy_ocr_predictions = self.easy_ocr.detect([img]) #this returns bbox and predictions - works for print - not hand written
        easy_ocr_predictions_coalesced = self.snap.to_lines(easy_ocr_predictions)   #coalesce fragmented bboxes to help improve recognition for 
                                                                                    #TR OCR model that only recognizes - not detects. So coalesced bbox improve prediction accuracy
        self.debug_print("easy_ocr",easy_ocr_predictions_coalesced) 
        original_regions = self.cropper.crop(img,easy_ocr_predictions,"original") 
        coalesced_regions = self.cropper.crop(img,easy_ocr_predictions_coalesced,"coalesced") #Create cropped regions for TR OCR model to predict
        predictions["easy_ocr"] = easy_ocr_predictions_coalesced

         
        tr_ocr_predictions = self.tr_ocr.detect(coalesced_regions,easy_ocr_predictions_coalesced) #TR OCE takes coalesced regions
        self.debug_print("tr_ocr",tr_ocr_predictions)
        redacted_img = self.redactor.redact(tr_ocr_predictions,img,prefix_name="redacted_" + ''.join(input_file.split('/')[-1]))
        predictions["tr_ocr"] =  tr_ocr_predictions
        paddle_predictions = self.paddle_ocr.detect([input_file]) #paddle take image path
        paddle_predictions_coalesced = self.snap.to_lines(paddle_predictions)   #coalesce fragmented bboxes to help improve recognition for 
        predictions["paddle_ocr"] = paddle_predictions_coalesced
        self.debug_print("paddle_ocr",paddle_predictions_coalesced)

        self.convert_confidences_to_str(predictions)
        #json_object = json.dumps(predictions, indent = 4)
        #print(json_object)


    def debug_print(self,model_name,predictions):
        print()
        print("Predictions from model:",model_name)
        for node in predictions:
            #print(node["prediction"],node["left"],node["right"],node["top"],node["bottom"])
            print("Prediction:",node["prediction"],"    Confidence:",round(node["conf"],2))
        #pdb.set_trace()


         


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeIdentify images - prototype ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', dest="input", action='store',help='Input image in jpg/png format')
    parser.add_argument('-device', dest="device", action='store',default="cpu",help='Input image in jpg/png format')

    results = parser.parse_args()
    try:
        main_obj = DeIdentify(results.device)
        main_obj.extract_text(results.input)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
