import os

import torch
import numpy as np

from .models_seggpt import seggpt_vit_large_patch16_input896x448

from typing import List, Dict, Tuple, Optional, Any
import torch.nn.functional as F
import numpy as np
from PIL import Image


from typing import List, Tuple, Optional, Any, Union
from .utils import download_seggpt_model, is_dir

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
res, hres = 448, 448


class SegGPTInference:
    def __init__(self, ckpt_path: str, model: str='seggpt_vit_large_patch16_input896x448', 
                 seg_type: str='instance', device: str='cuda'):
        """
        Initializes the SegGPTInference class with the provided checkpoint path, model, segmentation type, device, and output directory.

        Args:
        ckpt_path (str): Path to the model checkpoint.
        model (str): Name of the model.
        seg_type (str): Type of segmentation ('instance' or 'semantic').
        device (str): Device to run the model on ('cpu' or 'cuda').
        output_dir (str): Directory to save the output images or videos.
        """
        self.ckpt_path = ckpt_path
       
        self.model_name = model # not used, only uses seggpt_vit_large_patch16_input896x448
        self.seg_type = seg_type
        self.device = torch.device(device)
        #self.output_dir = output_dir
        self.model = self.prepare_model()

    def prepare_model(self) -> torch.nn.Module:
        """
        Prepares the model for inference. Loads the model weights from the checkpoint file and sets the model to evaluation mode.

        Returns:
        torch.nn.Module: The prepared model.
        """
        if self.ckpt_path is None or is_dir(self.ckpt_path):
            self.ckpt_path = download_seggpt_model(self.ckpt_path)
        model = seggpt_vit_large_patch16_input896x448() #getattr(models_seggpt, self.model_name)()
        model.seg_type = self.seg_type
        checkpoint = torch.load(self.ckpt_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()
        return model.to(self.device)

    def run_inference_image(
            self, input_image: str, prompt_image: List[str], prompt_mask: List[str], threshold: float= 0.25
    ) -> Tuple[dict, Optional[Dict], Any]:
        """
        Runs inference on an image using the SegGPT model. The output is saved as an image file in the output directory.

        Args:
        input_image (str): Path to the input image file.
        prompt_image (List[str]): List of paths to the prompt image files.
        prompt_mask (List[str]): List of paths to the prompt target files.
        threshold (float): Threshold for the prediction mask.

        Returns:
        Tuple[Dict, Optional[Dict], Any]: The output in COCO format, and the RLE mask.
        """
        assert os.path.exists(input_image), f"Input image not found at {input_image}"
        assert all(os.path.exists(img) for img in prompt_image), f"One or more prompt images not found"
        assert all(os.path.exists(img) for img in prompt_mask), f"One or more prompt targets not found"

        img_name = os.path.basename(input_image)
        
        masked_image, orig_image, mask = self.inference_image(
            input_image, prompt_image, prompt_mask, threshold
        )
        '''
        coco_output = coco_format(
            image_id=str(uuid.uuid4()),
            rle_mask=binary_mask_to_rle(mask),
            image_path=input_image,
            bbox=None,# this never gets set due to mask presence

        )
        '''
        

        return  to_boolean_mask(mask)

    @torch.no_grad()
    def run_one_image(self, img: np.ndarray,
                    tgt: np.ndarray,
                    input_size: Tuple[int, int],
                    threshold: float) -> Tuple[np.ndarray, Dict, float, List[float], Dict]:
        """
        Runs inference on one image using the provided model.

        Args:
        img (np.ndarray): The input image.
        tgt (np.ndarray): The target image.
        model (torch.nn.Module): The model to use for inference.
        device (torch.device): The device to run the model on.
        input_size (Tuple[int, int]): The size of the input image.
        threshold (float): The threshold to use for the mask.

        Returns:
        Tuple[np.ndarray, Dict, float, List[float]]: The output image, the mask in RLE format, the area of the mask,
        the bounding box of the mask, and metrics from the mask predictions.
        """
        x = torch.tensor(img)
        # make it a batch-like
        x = torch.einsum('nhwc->nchw', x)

        tgt = torch.tensor(tgt)
        # make it a batch-like
        tgt = torch.einsum('nhwc->nchw', tgt)

        bool_masked_pos = torch.zeros(self.model.patch_embed.num_patches)
        bool_masked_pos[self.model.patch_embed.num_patches // 2:] = 1
        bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
        valid = torch.ones_like(tgt)

        if self.model.seg_type == 'instance':
            seg_type = torch.ones([valid.shape[0], 1])
        else:
            seg_type = torch.zeros([valid.shape[0], 1])

        feat_ensemble = 0 if len(x) > 1 else -1
        _, y, mask_pos = self.model(x.float().to(self.device), tgt.float().to(self.device), bool_masked_pos.to(self.device),
                        valid.float().to(self.device), seg_type.to(self.device), feat_ensemble)
        y = self.model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        output = y[0, y.shape[1] // 2:, :, :]
        output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)

        

        return output

    def prep_image_prompt(self,input_image_path,  prompt_image_paths, prompt_mask_paths):
        image = Image.open(input_image_path).convert("RGB")

        image = np.array(image.resize((res, hres))) / 255.

        image_batch, target_batch = [], []
        for img2_path, tgt2_path in zip(prompt_image_paths, prompt_mask_paths):
            img2 = Image.open(img2_path).convert("RGB")
            img2 = img2.resize((res, hres))
            img2 = np.array(img2) / 255.

            tgt2 = Image.open(tgt2_path).convert("RGB")
            tgt2 = tgt2.resize((res, hres), Image.NEAREST)
            tgt2 = np.array(tgt2) / 255.

            tgt = tgt2  # tgt is not available
            tgt = np.concatenate((tgt2, tgt), axis=0)
            img = np.concatenate((img2, image), axis=0)

            assert img.shape == (2 * res, res, 3), f'{img.shape}'
            # normalize by ImageNet mean and std
            img = img - imagenet_mean
            img = img / imagenet_std

            assert tgt.shape == (2 * res, res, 3), f'{img.shape}'
            # normalize by ImageNet mean and std
            tgt = tgt - imagenet_mean
            tgt = tgt / imagenet_std

            image_batch.append(img)
            target_batch.append(tgt)

        img = np.stack(image_batch, axis=0)
        tgt = np.stack(target_batch, axis=0)
        return img, tgt
    def inference_image(self, 
                        img_path: str,
                        img2_paths: List[str],
                        tgt2_paths: List[str],
                        threshold: float) -> Tuple[np.ndarray, np.ndarray, Dict, float, List[float], Dict]:
        """
        Runs inference on an image using the provided model.

        Args:
        model (torch.nn.Module): The model to use for inference.
        device (torch.device): The device to run the model on.
        img_path (str): Path to the input image file.
        img2_paths (List[str]): List of paths to the prompt image files.
        tgt2_paths (List[str]): List of paths to the prompt target files.
        out_path (str): Path to the output image file.
        prediction_metrics (dict):

        Returns:
        Tuple[np.ndarray, np.ndarray, Dict, float, List[float], float]: The output image, the original image,
        the mask, the area of the mask, k, and the mask metrics.
        """
        

        img, tgt = self.prep_image_prompt(img_path, img2_paths, tgt2_paths)
        image = Image.open(img_path).convert("RGB")
        orig_image = np.array(image)
        size = image.size
        """### Run SegGPT on the image"""
        # make random mask reproducible (comment out to make it change)
        torch.manual_seed(2)
        #output, mask, area, prediction_metrics = self.run_one_image(img, tgt, size, threshold)
        output = self.run_one_image(img, tgt, size, threshold)
        output_img = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2),
            size=[size[1], size[0]],
            mode='nearest',
        ).permute(0, 2, 3, 1)[0].numpy()

        # TODO: Refactor for more efficiency.  Combine with code in run_one_image so mask is calculated only once.
        output_img[:, :, 1:2] = 0  # Keep only red
        mask =  np.where(output_img > threshold * 255, 255, 0).astype(np.uint8)
        output_img = Image.fromarray(np.where(output_img > threshold * 255, 255, orig_image).astype(np.uint8))

        #make mask single channel
        mask = mask[:,:,0]

        #output.save(out_path)
        #print("inference image", mask)
        return np.array(output_img), orig_image, np.array(mask)#, area,  prediction_metrics



def coco_format(
        image_id: str, rle_mask: Optional[Dict], image_path: str, bbox: List[float], area: float, metrics: dict
) -> Dict:
    """
    Formats the segmentation result into COCO format.

    Args:
    image_id (str): Unique identifier for the image.
    rle_mask (Optional[Dict]): The mask for the image in RLE format. If no mask, pass None.
    image_path (str): Path to the image file.
    bbox (List[float]): Bounding box for the object in the format [x,y,width,height].
    area (float): The area of the object.
    metrics (dict): The prediction metrics.

    Returns:
    Dict: The segmentation result in COCO format.
    """
    if rle_mask is not None:
        rle_mask['counts'] = str(rle_mask['counts'])

    annotations = [] if rle_mask is None else [
        {
            "bbox": bbox,
            "area": str(area),
            "is_crowd": 0,
            "segmentation": rle_mask,
            "category_id": 0,
        }
    ]

    return {
        "image_id": image_id,
        "filename": image_path,
        "width": rle_mask['size'][1] if rle_mask else 0,
        "height": rle_mask['size'][0] if rle_mask else 0,
        "annotations": annotations,
        "metrics": metrics
    }
def to_boolean_mask(mask):
    return np.where(mask>0, True, False)


