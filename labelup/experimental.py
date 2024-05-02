
from .seggpt import SegGPTInference
from .utils import get_files, write_boolean_mask
import os
import numpy as np

#from onyximages.embeddings import dinov2embed, QDrantStorage
#from onyximages.qdrant_utils import create_collection
from torch.nn import BCELoss
from PIL import Image
import torch
import uuid
import json
MODEL_PATH= "/data2/models/"

from os import listdir
from os.path import isfile, join


'''
it is expected that the image and mask paths files are sorted such that they 
pair together

'''
def bi_directional_rank(target_img_dir, prompt_img_paths, prompt_mask_paths,working_dir = "tmp"):
    seg_gpt = SegGPTInference(MODEL_PATH)
    _make_dir(working_dir)
    
    target_mask_dir = os.path.join(working_dir, "tmhat")

    _make_dir(target_mask_dir)
    
    '''
    
    initial prediction
    
    '''
    print(f"Running initial prediction from {len(prompt_img_paths)} on directory {target_img_dir}")
    predict_dir(prompt_img_paths, prompt_mask_paths, target_img_dir, target_mask_dir)


    # skip evaluate true error


    '''
    
    run inverse predictions
    
    '''
    prompt_pred_mask_dir = os.path.join(working_dir, "pmhat")
    _make_dir(prompt_pred_mask_dir)


    pairs = json.load(open(os.path.join(target_mask_dir,"pairs.json")))
    target_image_paths =[p['image'] for p in pairs]
    target_mask_paths =[p['mask'] for p in pairs]
    print(f"Running inverse predictions on {len(target_image_paths)} target images and {prompt_mask_paths} prompts")

    loss = BCELoss()
    y_true = [Image.open(p) for p in prompt_mask_paths]
    results = []
    


    for img, mask in zip(target_image_paths, target_mask_paths):
        
        for prompt_image, prompt_mask in zip(prompt_img_paths, y_true):
            pmhat = seg_gpt.run_inference_image(prompt_image,  [img],  [mask] )
            #create new name
            name = str(uuid.uuid4())+"_mask.png"
            write_boolean_mask(os.path.join(prompt_pred_mask_dir, name), pmhat)

            # norm from boolean
            pmhat = pmhat * 1.0

            prompt_mask = np.where(np.array(prompt_mask) > 128., 1., 0.)
            # calc loss
            bce = loss(torch.from_numpy(prompt_mask), torch.from_numpy(pmhat))
            bce = bce.item()


            result_entry={
                "Ti": img,
                "Pi": prompt_image,
                "TmHat": mask,
                "PmHat": name,
                "bce": bce
            }
            results.append(result_entry)
    with open(os.path.join(prompt_pred_mask_dir, "results.json"), 'w') as f:
        json.dump(results, f)

    return os.path.join(prompt_pred_mask_dir, "results.json")
        

def sort_results(results_path):

    results = json.load(open(results_path,"r"))
    results.sort(key = lambda x: x["bce"])
    return results

def _make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

    


def write_mask(mask_data, path):

    write_boolean_mask(path, mask_data)

def predict_dir(prompt_imgs, prompt_paths, target_dir, output_dir):

    seg_gpt = SegGPTInference(MODEL_PATH)

    unlabeled_images = get_files(target_dir, ".jpg")# fix this to be flexible

    for pi in prompt_imgs:
        if pi in unlabeled_images:
            unlabeled_images.remove(pi)
    #keep track of pairs

    pairs = []
    #run inference
    for ui in unlabeled_images:
        mask = seg_gpt.run_inference_image(ui, 
                                        prompt_imgs, 
                                        prompt_paths
        )
        base_name = os.path.splitext(os.path.basename(ui))[0]
        mask_file = os.path.join(output_dir, base_name+"_mask.png")
        write_boolean_mask(mask_file, mask)
        pairs.append({"image":ui, "mask":mask_file})
    
    with open(os.path.join(output_dir, "pairs.json"),'w') as f:
        json.dump(pairs, f)

'''
def embed_dir(target_dir, collection_name):
    embed = dinov2embed(cache_dir="/data2/dinov2/",device= 'cuda')
    embedding_dimensions = embed.feat_dim

    #check 
    create_collection(collection_name, embedding_dimensions)

     # embed
    store = QDrantStorage(collection_name)

    count = store.get_total_count()
    if count == 0 :
        for features, paths in embed.batch_embed(target_dir, batch_size=32,data_workers=4 ):
            store.batch_store(features, paths)
#reduce mask to single class

'''
   


