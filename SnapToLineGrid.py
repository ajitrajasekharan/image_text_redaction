import pdb
import PIL
from PIL import ImageDraw
import sys
import argparse
import json
from collections import OrderedDict

class SnapToLineGrid:
    '''
    Coalescing bbox outputs of text region detectors is required particularly for downstream transformer based OCR model (works well for hnadwritten text but poorly for print text) that benefit from larger sentence context. 
    This class attempts to coalesce bbboxes of words into sentences if the words are spaced by space character size. 
    Space character size is currently computed using an estimate from the OCR output of the region detector that is currently used(easyOCR) which also performs OCR - works well for print text but with errors  for handwritten text.
    The computed space character size is then  used to break lines (detected as the first step) into separate segments.
    This coalescing is also useful to aggregate bboxes in a line even in print case -  particularly if a downstream deidentifcation module benefits from larger context
    '''
    def __init__(self,spacing_scale = 5):
        self.spacing_scale = spacing_scale

    #The coordindate system's origin is top left
    def get_max_bbox(self,predictions):
        bbox = {"left":0,"right":0,"top":0,"bottom":0}
        for i in range(len(predictions)):
            if (i == 0):
                bbox["left"] = predictions[i]["left"]
                bbox["right"] = predictions[i]["right"]
                bbox["top"] = predictions[i]["top"]
                bbox["bottom"] = predictions[i]["bottom"]
                continue
            if (predictions[i]["left"] < bbox["left"]):
                bbox["left"] = predictions[i]["left"]
            if (predictions[i]["right"] > bbox["right"]):
               bbox["right"] = predictions[i]["right"]
            if (predictions[i]["top"] < bbox["top"]):
               bbox["top"] = predictions[i]["top"]
            if (predictions[i]["bottom"] > bbox["bottom"]):
               bbox["bottom"] = predictions[i]["bottom"]
        bbox["width"] = bbox["right"] - bbox["left"]
        bbox["height"] = bbox["bottom"] - bbox["top"]
        return bbox

    def horizontal_overlap(self,node,candidates_dict):
        for key in candidates_dict:
            picked_node = candidates_dict[key]
            #check if one node is within the span of another
            if (((node["left"] <= picked_node["left"])  and (node["right"] >= picked_node["right"])) or (picked_node["left"] <= node["left"] and (picked_node["right"] >= node["right"]))):
                return True
            else:
                return False

    def pick_row_candidates(self,complete_candidates_dict,in_data_dict,running_top,picked_count,max_bbox):
        lowest_point = max_bbox["bottom"]
        topmost_point = max_bbox["top"]
        candidates_dict = OrderedDict()

        #Step 1: find the top mode node[s] first and picks it bottom val
        bottom_most = lowest_point
        width_span = 0
        for i in range(len(in_data_dict)):
            node = in_data_dict[i]
            if (i in complete_candidates_dict):
                continue
            if (node["top"] <= running_top):
                #candidates_dict[i] = node
                #picked_count += 1 
                if (node["bottom"] < bottom_most):
                    bottom_most  = node["bottom"]

        #Step 2: pick all nodes within the bottom most point range
        for i in range(len(in_data_dict)):
            if (i in candidates_dict or i in complete_candidates_dict):
                continue
            node = in_data_dict[i]
            if (node["top"] <= bottom_most and not self.horizontal_overlap(node,candidates_dict)):
                candidates_dict[i] = node
                picked_count += 1 
                        

        new_top = lowest_point
        #Step 3. Find the new top for the next iteration in the nodes left
        for i in range(len(in_data_dict)):
            if (i in candidates_dict or i in complete_candidates_dict):
                continue
            node = in_data_dict[i]
            if (node["top"] < new_top):
                new_top = node["top"]
        return candidates_dict,picked_count,new_top

    def extract_lines(self,in_data,max_bbox):
        ret_rows = []
        in_data_dict = OrderedDict()
        complete_candidates_dict = OrderedDict()
        for i in range(len(in_data)):
            in_data_dict[i] = in_data[i]
        picked_count = 0
        total_count = len(in_data_dict)
        running_top = max_bbox["top"]
        while (picked_count < total_count):
            pick_row_candidates,picked_count,running_top = self.pick_row_candidates(complete_candidates_dict,in_data_dict,running_top,picked_count,max_bbox)
            complete_candidates_dict.update(pick_row_candidates)
            ret_rows.append(pick_row_candidates)
        return ret_rows

        
    def compute_spacing_size(self,in_data):
        total_length = 0
        total_words = 1
        for i in range(len(in_data)):
            node = in_data[i]
            total_length += (node["right"] - node["left"])
            total_words += len(node["prediction"])
        spacing_estimate = (int(float(total_length)/total_words))*self.spacing_scale
        return spacing_estimate

    def break_into_phrases(self,lines_arr,spacing_size):
        ret_phrases = []
        phrase_bbox = {}
        for i in range(len(lines_arr)):
            values_list = list(lines_arr[i].values())
            sorted_list = sorted(values_list, key=lambda d: d['left']) 
            #print(sorted_list)
            phrase_bbox = {}
            for j in range(len(sorted_list)):
                node = sorted_list[j]
                if (len(phrase_bbox) == 0):
                    phrase_bbox["left"] = node["left"]
                    phrase_bbox["right"] = node["right"]
                    phrase_bbox["top"] = node["top"]
                    phrase_bbox["bottom"] = node["bottom"]
                    phrase_bbox["prediction"] = node["prediction"]
                    phrase_bbox["conf"] = node["conf"]
                    phrase_bbox["inf_time"] = node["inf_time"]
                    phrase_bbox["count"] = 1
                else:
                    if (node["left"] - phrase_bbox["right"] > spacing_size):
                        phrase_bbox["conf"] = float(phrase_bbox["conf"])/phrase_bbox["count"] #Average confidence
                        phrase_bbox["inf_time"] = float(phrase_bbox["inf_time"])/phrase_bbox["count"] #Average prediction time
                        ret_phrases.append(phrase_bbox)
                        phrase_bbox = {}
                        phrase_bbox["left"] = node["left"]
                        phrase_bbox["right"] = node["right"]
                        phrase_bbox["top"] = node["top"]
                        phrase_bbox["bottom"] = node["bottom"]
                        phrase_bbox["prediction"] = node["prediction"]
                        phrase_bbox["conf"] = node["conf"]
                        phrase_bbox["inf_time"] = node["inf_time"]
                        phrase_bbox["count"] = 1
                    else:
                        #Note left is not need - this is a concat of x axis sorted boxes
                        phrase_bbox["right"] = node["right"]
                        phrase_bbox["top"] = node["top"] if node["top"] < phrase_bbox["top"] else phrase_bbox["top"]
                        phrase_bbox["bottom"] = node["bottom"] if node["bottom"] > phrase_bbox["bottom"] else phrase_bbox["bottom"]
                        phrase_bbox["prediction"] = phrase_bbox["prediction"] + " " + node["prediction"]
                        phrase_bbox["conf"] = phrase_bbox["conf"]  + node["conf"]
                        phrase_bbox["inf_time"] = phrase_bbox["inf_time"]  + node["inf_time"]
                        phrase_bbox["count"] += 1
            if (len(phrase_bbox) != 0):
                phrase_bbox["conf"] = float(phrase_bbox["conf"])/phrase_bbox["count"] #Average confidence
                phrase_bbox["inf_time"] = float(phrase_bbox["inf_time"])/phrase_bbox["count"] #Average prediction time
                ret_phrases.append(phrase_bbox)
                phrase_bbox = {}
        assert(len(phrase_bbox) == 0)
        return ret_phrases
                    



    def to_lines(self,in_data):
        #Step 1: Get maximal bounding box of all boxes
        max_bbox = self.get_max_bbox(in_data)
        lines_arr = self.extract_lines(in_data,max_bbox)
        spacing_size = self.compute_spacing_size(in_data)
        phrases_bbox_arr = self.break_into_phrases(lines_arr,spacing_size)
        return phrases_bbox_arr
