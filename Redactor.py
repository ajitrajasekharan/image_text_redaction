import pdb
import PIL
from PIL import ImageDraw
from PIL import Image
import os


DEFAULT_SAVE_DIRECTORY="crops"

class Redactor:
    def __init__(self,save_redacts = True,save_dir=DEFAULT_SAVE_DIRECTORY):
        self.save_redacts = save_redacts
        self.save_dir = save_dir
        if save_redacts  and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def redact(self,regions,img,prefix_name = "redacted"):
        
        for i in range(len(regions)):
            bbox = regions[i]
            smudge_img = Image.new('RGB',(int(bbox["right"] - bbox["left"]),int(bbox["bottom"] - bbox["top"])),(0,0,0))
            draw = ImageDraw.Draw(smudge_img)
            draw.text((0,0), "Redacted", fill=(255, 255, 255))
            img.paste(smudge_img,(int(bbox["left"]),int(bbox["top"])))
        if (len(regions) > 0 and self.save_redacts):
            if (prefix_name.endswith(".jpg")):
                file_name = prefix_name
            else:
                file_name = prefix_name + ".jpg"
            img.save(self.save_dir + "/" + file_name)  
        return img
         


