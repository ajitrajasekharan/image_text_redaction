import pdb
import PIL
from PIL import ImageDraw
import sys
import argparse
import json
import os


DEFAULT_SAVE_DIRECTORY="crops"

class RegionsCropper:
    def __init__(self,save_crops = True,save_dir=DEFAULT_SAVE_DIRECTORY):
        self.save_crops = save_crops
        self.save_dir = save_dir
        if save_crops  and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def crop(self,img,regions,prefix_name = "c_"):
        ret_arr = []
        for i in range(len(regions)):
            bbox = regions[i]
            im1 = img.crop((bbox["left"],bbox["top"],bbox["right"],bbox["bottom"]))
            if (self.save_crops):
                im1.save(self.save_dir + "/" + prefix_name + "_" +  str(i) + ".jpg")  
            ret_arr.append(im1)
        return ret_arr
         


