from msilib.schema import Class
from cv2 import CLAHE
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
from mrcnn import model as modellib, utils
import segmentation_models as sm

import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

def load_semantic_model(checkpoint_path, by_name=True):
    '''
    Trains a semantic model
    '''
    seg_model = sm.Unet('efficientnetb7', classes=1, activation='sigmoid')
    seg_model.load_weights(checkpoint_path, by_name=True)
        
    
    return seg_model

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

#CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
CLASS_NAMES = ['BG','cat1']
#CLASS_NAMES = ['BG', 'mito_red', 'mito_green', 'mito_overlap']
class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "mito_normal_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
semantic_model = load_semantic_model(r"antenna_sem_seg_model.h5")
model = modellib.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd(),
                             #semantic_model = semantic_model,
                             )

# Load the weights into the model.
file_dir = r'D:\Nirwan\MRCNN_TF2\Mask-RCNN-TF2\logs\antenna_normal20220809T1748'
i=0
for files in os.listdir(file_dir): 

    file = os.path.join(file_dir, files)
    
    model.load_weights(filepath=file, 
                    by_name=True, #exclude=["mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask"]
                    )

    # load the input image, convert it from BGR to RGB channel
    image = cv2.imread("antenna.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform a forward pass of the network to obtain the results
    r = model.detect([image])

    # Get the results for the first image.
    r = r[0]
    
    

    # Visualize the detected objects.
    mrcnn.visualize.display_instances(image=image, 
                                    boxes=r['rois'], 
                                    masks=r['masks'], 
                                    class_ids=r['class_ids'], 
                                    class_names=CLASS_NAMES, 
                                    scores=r['scores'],
                                    save_fig_path=os.path.join(r'inferred_images\antenna_normal', str(i)),
                                    i=i)
    i += 1