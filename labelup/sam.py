from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import numpy as np


class SAMInference:
    def __init__(self, model_ckpt: str, device: str='cuda'):
        self.model_ckpt = model_ckpt
       
        self.device = torch.device(device)

        self.model = self.prepare_predictor()
        


    def prepare_predictor(self):
        sam = sam_model_registry["vit_h"](self.model_ckpt)        # Note: YOU MUST HAVE THE MODEL SAVED IN THE SAME DIRECTORY AS YOUR BACKEND
        sam.to(device=self.device)

        predictor = SamPredictor(sam)
        return predictor
    

    def set_image(self,pth):
        image = cv2.imread(pth)               
        self.model.set_image(image)
    
    '''
    
    point = (x,y)
    '''
    def predict(self, point, img_path = None):
        if img_path is not None:
            self.set_image(img_path) 

        input_point = np.array([[int(point[0]), int(point[1])]])
        input_label = np.array([1])
        masks, scores, logits = self.model.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        return masks
    


    



