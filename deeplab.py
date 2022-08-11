from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import segmentation_models as sm
from tensorflow import keras

## For visualizing results
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def filterDataset(ann_path, classes):    
    # initialize COCO api for instance annotations
    annFile = ann_path
    coco = COCO(annFile)
    
    images = []
    if classes!=None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
    
    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])
            
    random.shuffle(unique_images)
    dataset_size = len(unique_images)
    
    return unique_images, dataset_size, coco

train_ann_path = r"D:\Nirwan\PANet\foo-upgraded\coco\annotations\instances_train2014.json"
val_ann_path = r"D:\Nirwan\PANet\foo-upgraded\coco\annotations\instances_val2014.json"

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return None

def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name'])/255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img
    
def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className)+1
        new_mask = cv2.resize(coco.annToMask(anns[a])*pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask  
    
def getBinaryMask(imageObj, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)
        
        #Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def dataGeneratorCoco(images, classes, coco, folder, 
                      input_image_size=(224,224), batch_size=4, mask_type='binary'):
    
    img_folder = '{}'.format(folder)
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)
    
    c = 0
    while(True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

        for i in range(c, c+batch_size): #initially from 0 to batch_size, when c = 0
            imageObj = images[i]
            
            ### Retrieve Image ###
            #print(imageObj)
            train_img = getImage(imageObj, img_folder, input_image_size)
            
            ### Create Mask ###
            if mask_type=="binary":
                train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)
            
            elif mask_type=="normal":
                train_mask = getNormalMask(imageObj, classes, coco, catIds, input_image_size)                
            
            # Add to respective batch sized arrays
            img[i-c] = train_img
            mask[i-c] = train_mask
            
        c+=batch_size
        if(c + batch_size >= dataset_size):
            c=0
            random.shuffle(images)
        
        assert not np.any(np.isnan(img))
        assert not np.any(np.isnan(mask))
        
        yield img, mask/255
        


classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

print(len(classes))

train_images, train_dataset_size, train_coco = filterDataset(train_ann_path, classes,)
val_images, val_dataset_size, val_coco = filterDataset(val_ann_path, classes,)

batch_size = 4
input_image_size = (256,256)
mask_type = 'normal'

train_img_folder = r'D:\Nirwan\PANet\foo-upgraded\coco\train2014'
val_img_folder = r'D:\Nirwan\PANet\foo-upgraded\coco\val2014'

train_gen = dataGeneratorCoco(train_images, classes, train_coco, train_img_folder,
                            input_image_size, batch_size, mask_type)
val_gen = dataGeneratorCoco(val_images, classes, val_coco, val_img_folder,
                            input_image_size, batch_size, mask_type)

n_epochs = 20
steps_per_epoch = train_dataset_size // batch_size
validation_steps = val_dataset_size // batch_size

model = sm.Unet('efficientnetb7', classes=80, activation='sigmoid')
model.load_weights(r'./training_checkpoints/coco/ckpt_23')

optim = keras.optimizers.Adam(0.0001)
dice_loss = sm.losses.DiceLoss(class_weights=np.ones(80)) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model.compile(optim, total_loss, metrics)

checkpoint_dir = './training_checkpoints/coco'
        # Define the name of the checkpoint files.
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

callbacks = [
        keras.callbacks.TensorBoard(log_dir='./semantic_logs'),
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, verbose=0,
                                            save_weights_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=0.001,
                                            patience=1,
                                            verbose=1,
                                            mode="auto",
                                            baseline=None,
                                            restore_best_weights=True),
    ]

history = model.fit(x=train_gen, validation_data=val_gen, epochs=1, steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps, verbose=True, callbacks=callbacks)

model.save("coco_sem_seg_model.h5")