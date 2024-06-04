from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import os
from segment_anything import sam_model_registry, SamPredictor
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import  read_video_from_path
import torch
from labelup.utils import  mask_to_bbox,write_boolean_mask
import numpy as np
from labelup.seggpt import SegGPTInference
import uuid
import cv2

# model init - outside class definition due to multithreading from labelstudio ml-backend
# cotracker init 




#sam init
# SAM

#print("loading cotracker")
#cotracker = CoTrackerPredictor(checkpoint=os.path.join('/app/co-tracker/checkpoints/cotracker2.pth'))


print("loading sam")
reg = sam_model_registry["vit_h"]('/app/sam_weights/sam_vit_h_4b8939.pth') 
#reg.to(device='cuda') #  RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method

# need two beefy gpus to run!
if torch.cuda.is_available():
    reg.to(device='cuda:0')
    
    sam = SamPredictor(reg)
    
    seg_gpt = SegGPTInference(
        '/app/seggpt_weights/seggpt_vit_large.pth',
        device='cuda:1')
    print("loaded sam and seggpt")
else:
    sam = SamPredictor(reg)
    
    seg_gpt = SegGPTInference(
        '/app/seggpt_weights/seggpt_vit_large.pth',
        device='cpu')
    print("loaded sam and seggpt")
def set_image(image):
    sam.set_image(image)

def run_sam(input_point):
    return sam.predict(
            point_coords=input_point,
            point_labels=np.array([1]),
            multimask_output=False)

def run_seggpt(target_file, prompt_images, prompt_masks):
    return seg_gpt.run_inference_image(target_file, 
                                        prompt_images, 
                                        prompt_masks)
#TODO move all predicts to outside of class defs to use gpus
class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    
    def setup(self):
        """Configure any parameters of your model here
        """

        #TODO MOVE MODEL LOADING CODE TO OUTSIDE CLASS DESCRIPTION SO IT CAN LOAD 
        # similar to here https://github.com/HumanSignal/label-studio-ml-backend/blob/17e09323acceccd50e2d2cdc053afa430f232354/label_studio_ml/examples/grounding_dino/dino.py#L98
        self.root_dir = "/label-studio/data"
        self.upload_dir= "/label-studio/data/media/upload"
        
        #TODO mount gpus - how many do we need

        # make working directory if not available
        self.working_dir = os.path.join(self.root_dir,"working")
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)



        self.set("model_version", "0.0.1")
    
    def sam_image(self, image, bbox):
        # TODO add additional points - maybe in  cross
        center_x = bbox[0]+ (bbox[2]/2.)
        center_y = bbox[1]+ (bbox[3]/2.)
        print(f"saming image at point {center_x}, {center_y} : from bbox {bbox}")
        input_point=np.array([[center_x, center_y]])
        set_image(image)
        masks, scores, logits = run_sam(input_point)
        maxscore=-1
        maxi = -1
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score > maxscore: 
                maxscore = score
                maxi = i
        #return masks[maxi] # return best
        return masks[0] # return first

    def get_file_path(self, url):
        print(f"processing url {url}")
        folders = url.split("/")
        fn = folders[-1]
        f = folders[-2]
        return os.path.join(self.upload_dir, f, fn)
    '''
    def mask_to_bbox(self, mask):
        minx = 99999999
        miny = 99999999
        maxx = -1
        maxy=-1
        sh = np.shape(mask)
        for i in range(sh[1]):
            for j in range(sh[0]):
                if mask[j][i]== 1:
                    if i < minx: minx = i
                    if i > maxx: maxx = i
                    if j < miny: miny = j
                    if j > maxy: maxy = j
        return (minx, miny, maxx, maxy)
    '''

    def sam_and_seggpt(self,video_frames, sequence): # not really co track but sam + seggpt
        sequences = []
        width = np.shape(video_frames[0])[1]
        height = np.shape(video_frames[0])[0]
        print(f"video width {width}, height {height}, and {len(video_frames)} frames")
        frame = sequence['frame']
        x = width * sequence['x'] /100.
        y = height * sequence['y'] /100.
        box_width = width * sequence['width']/100.
        box_height = height * sequence['height'] /100.


        #  decrease frame by some number - cause label studio does index at 1
        frame -= 1
        mask = self.sam_image(video_frames[frame], (x,y,box_width, box_height))
   
        vid_len= len(video_frames)
        print(f"mask dims {np.shape(mask)}")
        #test put result on next frame
        #bbox = self.mask_to_bbox(mask) # this should be from labelup-core but import cv2 doesnt work for some reason
        
        files_to_delete=[]
        # change to iterate through all frames
        prompt_images =[]
        prompt_masks=[]
        prompt_len =3
        while frame < (100): # artificially constrained due to front end limitations, true use would be video_len -1 
            
        #while mask is not None:
            
            #write mask and frame
            if mask is not None:
                mask_file = os.path.join(self.working_dir, str(uuid.uuid4())+f"_mask_{frame}.png")
                write_boolean_mask(mask_file, mask)
                files_to_delete.append(mask_file)
                
                prompt_file = os.path.join(self.working_dir, str(uuid.uuid4())+f"_prompt_{frame}.png")
                cv2.imwrite(prompt_file, video_frames[frame])
                files_to_delete.append(prompt_file)

                prompt_images.append(prompt_file)
                prompt_masks.append(mask_file)

                if len(prompt_images) >prompt_len:
                    prompt_images.pop(0)
                    prompt_masks.pop(0)
            print(f"Running predictions with {len(prompt_masks)} prompts")
            # move frame forward
            frame+=1
           
            
            print(f"target image is frame {frame}")
            #write new frame
            target_file =os.path.join(self.working_dir, str(uuid.uuid4())+f"_target_{frame}.png")
            cv2.imwrite(target_file, video_frames[frame])
            files_to_delete.append(target_file)
            # predict seggpt

            mask = run_seggpt(target_file, 
                                        prompt_images, 
                                        prompt_masks
            )

            bboxes = mask_to_bbox(mask)
            print(f"predicted bboxes {bboxes}")
        
            #if bbox[2]<0: return sequences
            for bbox in bboxes:
                copy_box = sequence.copy()
                copy_box['frame']= frame +1 # cause labelstudio starts at 1
                w = bbox[2]- bbox[0]
                h = bbox[3]-bbox[1]
                mask_x = float(bbox[0])/ width *100.
                mask_y = float(bbox[1])/height *100.
                mask_width = float(w)/width *100.
                mask_height = float(h)/height *100.

                copy_box['x'] = mask_x
                copy_box['y'] = mask_y
                copy_box['width'] = mask_width
                copy_box['height'] = mask_height
                copy_box['enabled']=False # this is interpolation

                sequences.append(copy_box)

            # check we returned a good mask
            # use sum above threshold , mask is boolean array
            threshold = 30
            if mask.sum() < threshold:
                mask = None
                print(f"threshold missed on frame {frame}")
         
        # delete files
        print(f"deleting f{len(files_to_delete)} files")
        for f in files_to_delete:
            os.remove(f)

        return sequences


        

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        '''
         Parsed JSON Label config: {'videoLabels': {'type': 'Labels', 'to_name': ['video'], 
         'inputs': [{'type': 'Video', 'value': 'video'}], 'labels': ['vehicle'], 
         'labels_attrs': {'vehicle': {'value': 'vehicle', 'background': '#FFA39E'}}},
           'box': {'type': 'VideoRectangle', 'to_name': ['video'], 'inputs': [{'type': 'Video', 'value': 'video'}],
             'labels': [], 'labels_attrs': {}}}
        
        '''
        #TOD check for 'action'
        # first id 
        task = tasks[0]
        annotations = task['annotations']
        #if not context or not context.get('result'):
            # if there is no context, no interaction has happened yet
        #    return []
        #image_width = context['result'][0]['original_width']
        #image_height = context['result'][0]['original_height']
        d = task['data']
        if 'video' not in d:
            print("this is not a video task")
            return [] # this is not a video task
        
        

        results =[]

        
        # load video
        url = d['video']
        fp = self.get_file_path(url)
        if not os.path.exists(fp):
            print(f"file not found: {fp}")
            return []
        print(f"resolved filepath {fp}")
        # run through cotracker
        video = read_video_from_path(fp)
        if video is None:
            print(f"read video returned none")
            return []
        print(f"found video of length {len(video)} and size {np.shape(video[0])}")


        # get annotations
        starting_boxes = []
        for annotation in annotations:
            for result in annotation['result']:
                
                result_objects=[]
                for sequence in result['value']['sequence']:
                    

                    ps = self.sam_and_seggpt(video, sequence) # fix this to make distinct results - need to gen new ids
                    print(f"returned {len(ps)} predicted sequences")
                    for prediction in ps:
                        value_obj = result['value'].copy()
                       
                        value_obj['sequence']= [prediction.copy()]
                        value_obj['duration']=1
                        value_obj['framesCount']=1
                        #value_obj['frames']=1
                        result_obj = result.copy()
                        result_obj['id'] = str(uuid.uuid4())[:8]
                        result_obj['value'] = value_obj
                        result_objects.append(result_obj)


                
                results.extend(result_objects)

       

            

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]

        #HOw to download video...

        
        predictions = [{
            'result':results,
            'model_version':"0.1",
            'score':1.0
        }]
        print(predictions)
        
        return ModelResponse(predictions=predictions)
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

