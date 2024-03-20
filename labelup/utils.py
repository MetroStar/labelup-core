import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Optional, Any, Union
import os
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)     

def show_arrow(coords,ax):
    bbox_args = dict(boxstyle="round", fc="0.8")
    an = ax.annotate(
        'annotation',
        xy=coords, xycoords='data',
        xytext=(-50, 30), textcoords='offset points',
   
        bbox=bbox_args,
        arrowprops=dict(arrowstyle="->"))
    an.arrow_patch.set_color('red')
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def plot_mask(path, mask, point=None, use_arrow=False):
    if path is None:
        image_array = np.full(np.shape(mask), 0)
    else:
        image_array = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
    plt.imshow(image_array)
    show_mask(mask, plt.gca())  
    if point is not None:
        if use_arrow:
            show_arrow(point, plt.gca())
        else:
            input_label = np.array([1])
            input_point = np.array([[point[0], point[1]]])
            show_points(input_point, input_label, plt.gca())
    plt.show()

def plot_img(path, point=None,use_arrow=False):
    image_array = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    plt.imshow(image_array)
    
    if point is not None:
        if use_arrow:
            show_arrow(point, plt.gca())
        else:
            input_label = np.array([1])
            input_point = np.array([[point[0], point[1]]])
            show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()   


def write_boolean_mask(pth, mask):
    h, w = mask.shape[-2:]
    img =mask.reshape(h, w, 1) * 255
    cv2.imwrite(pth, img) 

def get_files(project_location: str, ext: str=None) -> List[str]:
    """
    Returns a list of file paths in the specified project location that have the specified extension.

    Args:
    project_location (str | pathlib.Path): Directory containing the files.
    ext (str): File extension to filter by.

    Returns:
    List[str]: A list of file paths.
    """
    if str == None:
        return [os.path.join(project_location, file)
            for file in os.listdir(project_location)
            if os.path.isfile(os.path.join(project_location, file))]
    return [os.path.join(project_location, file)
            for file in os.listdir(project_location)
            if os.path.isfile(os.path.join(project_location, file)) and file.endswith(ext)]


# Visualization stuff
def plot_image_and_bboxes(image, bboxes, figsize=(15,15)):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    import matplotlib.image as mpimg
    ig, ax = plt.subplots()

    # Display the image
    if type(image)==str:
        ax.imshow(mpimg.imread(image))
    elif type(image) == np.ndarray:
        ax.imshow(image)

    # Create a Rectangle patch
    for bbox in bboxes:

        w = bbox[2]- bbox[0]
        h = bbox[3]-bbox[1]

        rect = patches.Rectangle((bbox[0], bbox[1]), w, h, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def plot_images_and_masks(images, masks,count=5, figsize=(15,15)):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    c = len(images)
    if c > count: c = count
    plt.figure(figsize=figsize)
    f, axarr = plt.subplots(2,c)
    if c >1:
        if type(images[0])== str:
            for i in range(c):
                axarr[0,i].imshow(mpimg.imread(images[i]))
                axarr[1,i].imshow(mpimg.imread(masks[i]))
        elif type(images[0]) == np.ndarray:
            for i in range(c):
                axarr[0,i].imshow(images[i])
                axarr[1,i].imshow(masks[i])
    elif c==1:
        if type(images[0])== str:
            axarr[0].imshow(mpimg.imread(images[0]))
            axarr[1].imshow(mpimg.imread(masks[0]))
        elif type(images[0]) == np.ndarray:
            
            axarr[0].imshow(images[i])
            axarr[1].imshow(masks[i])
    plt.show()

def to_boolean_mask(mask):
    return np.where(mask>0, True, False)

def mask_to_bbox(mask: np.ndarray) -> List[float]:
    """
    Calculates the bounding box of the object(s) in a mask.

    Args:
    mask (np.ndarray): The mask of the object(s).

    Returns:
    List[float]: The bounding box of the object(s) in the format [xmin, ymin, width, height]. If no object is found, returns [0, 0, 0, 0].
    """
    # Convert the mask to a binary mask
    binary_mask = mask.astype(np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #print(contours)
    # Initialize list to hold x, y, w, h for each contour
    boxes = []

    # Loop over each contour
    for contour in contours:
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(contour)
        # Add it to the list
        boxes.append([x, y, x + w, y + h])
    
    #return all the bboxes
    return boxes

sam_file_name = "sam_vit_h_4b8939.pth"
seggpt_file_name = "seggpt_vit_large.pth"

def download(url, pth):
    import urllib.request
    
    if not os.path.exists(pth):
        print(f"Downloading from {url} to {pth}")
        urllib.request.urlretrieve(url,pth)
        return True
    return False
    
def download_sam_model(dir=None):
    dir = create_cache(dir)
    pth =  os.path.join(dir, sam_file_name)
    download("https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth",pth)
    return pth

def download_seggpt_model(dir=None):
    dir = create_cache(dir)
    pth =  os.path.join(dir, seggpt_file_name)
    download("https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth",pth)

    return pth

def is_dir(pth):
    return os.path.exists(pth) and os.path.isdir(pth)
def create_cache(dir=None):
    if dir == None:
        dir = os.path.join(os.path.expanduser('~'), ".cache/")

    if not os.path.exists(dir):
        os.mkdir(dir)

    #make labelup specific path
    dir = os.path.join(dir, "labelup/")
    if not os.path.exists(dir):
        os.mkdir(dir)

    return dir
