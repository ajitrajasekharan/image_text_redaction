import pdb
import easyocr
from collections import OrderedDict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from paddleocr import PaddleOCR,draw_ocr
import time


EASY_OCR = 1
TR_OCR_HW = 2
TR_OCR_PRINT = 3
PADDLE_OCR = 4



class OCRWrapper:
    def __init__(self,device,ocr_model=EASY_OCR):
        self.ocr_model = ocr_model
        if (ocr_model == EASY_OCR):
            is_gpu = True if device == "gpu" else False
            self.reader = easyocr.Reader(['en'],gpu=is_gpu)
        elif (ocr_model == TR_OCR_HW):
            #self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
            #self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
        elif (ocr_model == TR_OCR_PRINT):
            #self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
            #self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed") 
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed") 
        else:
            print("Loading OCR model Paddle OCR.Note detection,recognition, and direction models need to be present in directories det_model, rec_model and angle_model")
            is_gpu = True if device == "gpu" else False
            self.model = PaddleOCR(det_model_dir='det_model', rec_model_dir='rec_model',  cls_model_dir='angle_model', use_angle_cls=True,use_gpu=is_gpu,show_log=False,enable_mkldnn=True) #the enable mkl is use to avoid a dll load error

    def easy_ocr_detect(self,img_arr,bound):
            assert(len(img_arr) == 1) #this is primarily used only for region detection. So only one image accepted
            img  = img_arr[0]
            start = time.time()
            bound = self.reader.readtext(img)
            end = time.time()
            predictions = []
            for i in range(len(bound)):
                bb = bound[i]
                left = bb[0][0][0]
                top = bb[0][0][1]
                right = bb[0][2][0]
                bottom = bb[0][2][1]
                #This check below is for inconsistent generation which happens for this detector for text that is at angle (not horizontally placed  in image)
                if (left > right):
                    temp = left
                    left = right
                    right = temp
                if (top > bottom):
                    temp = bottom
                    top = bottom
                    bottom = temp
                if (left == right):
                    right += 1
                if (top == bottom):
                    bottom += 1
                predictions.append({"left":int(left),"top":int(top),"right":int(right),"bottom":int(bottom),"prediction":bb[1],"conf":round(bb[2],2),"inf_time":round(float(end-start)/len(bound),2)})
            return predictions

    def detect(self,img_arr,bound=None):
        if (self.ocr_model == EASY_OCR):
            return self.easy_ocr_detect(img_arr,bound)
        elif (self.ocr_model == TR_OCR_PRINT or self.ocr_model == TR_OCR_HW):
            return self.tr_ocr_detect(img_arr,bound)
        else:
            return self.paddle_detect(img_arr,bound)

    def tr_ocr_detect(self,img_arr,bound):
        assert(len(bound) == len(img_arr))
        predictions = []
        for i in range(len(img_arr)):
            image = img_arr[i].convert("RGB")
            start = time.time()
            pixel_values = self.processor(image, return_tensors="pt").pixel_values 
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0] 
            end = time.time()
            #print(generated_text)
            predictions.append({"left":bound[i]["left"],"top":bound[i]["top"],"right":bound[i]["right"],"bottom":bound[i]["bottom"],"prediction":generated_text,"conf":1,"inf_time":round(end-start,2)})
        return predictions

    def paddle_detect(self,img_arr,bound):
            assert(len(img_arr) == 1) #Paddle can do bounds and text recognition. So onle one image expected in this call
            img  = img_arr[0]
            start = time.time()
            result = self.model.ocr(img,rec=True,det=True,cls=True)
            end = time.time()
            predictions = [] 
            for line in result:
                #print(line)
                predictions.append({"left":line[0][0][0],"top":line[0][0][1],"right":line[0][1][0],"bottom":line[0][2][1],"prediction":line[1][0],"conf":round(line[1][1],2),"inf_time":round(float(end-start)/len(result),2)})
            return predictions

