import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from skimage import io, img_as_int, img_as_ubyte
from skimage.filters import threshold_otsu

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from PIL import Image


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
  
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            class_values=None,
            start=None,
            end=None,
            row=None,
            column=None
        
        
    ):
        self.ids = os.listdir(images_dir)#[start: end]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = class_values#[self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.row=row
        self.column=column
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image =image[:self.row,:self.column] 

       # image=image[580:-579,580:-579,:]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = mask[:self.row,:self.column] 
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
       
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
      
      
      
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
#paths

noise_image_path='./GoogleDrive/My Drive/RMIT Master/porous_dataset_2019/deep_learning/noise/input'
noise_ground_truth_path='./GoogleDrive/My Drive/RMIT Master/porous_dataset_2019/deep_learning/noise/output'

real_image_path='./GoogleDrive/My Drive/RMIT Master/porous_dataset_2019/deep_learning/real/input'
real_ground_truth_path='./GoogleDrive/My Drive/RMIT Master/porous_dataset_2019/deep_learning/real/output'

model_path='./GoogleDrive/My Drive/CSIRO_image_august_2019/2D/selected_images'
spath='./GoogleDrive/My Drive/colab packages/HPC/xlearn'


battery_in_path='./GoogleDrive/My Drive/RMIT Master/porous_dataset_2019/battery/NMC_90wt_0bar/grayscale'
battery_out_path='./GoogleDrive/My Drive/RMIT Master/porous_dataset_2019/battery/NMC_90wt_0bar/binarized'

bread_syntheic_in='./GoogleDrive/My Drive/CSIRO_image_august_2019/2D/selected_images/250_250_synthetic/input'
bread_syntheic_out='./GoogleDrive/My Drive/CSIRO_image_august_2019/2D/selected_images/250_250_synthetic/output'

bread_real_in='./GoogleDrive/My Drive/CSIRO_image_august_2019/2D/selected_images/250_250_real/input'
bread_real_out='./GoogleDrive/My Drive/CSIRO_image_august_2019/2D/selected_images/250_250_real/output'


x_train_dir = noise_image_path
y_train_dir = noise_ground_truth_path

x_valid_dir = noise_image_path
y_valid_dir = noise_ground_truth_path

x_test_dir =real_image_path
y_test_dir = real_ground_truth_path

CLASSES = ['pore']
class_values=[255]
Dataset(
    x_train_dir, 
    y_train_dir, 
    classes=CLASSES, 
)
# Lets look at data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=['pore'],class_values=[255], row=192,column=192)

image, mask = dataset[5] # get some sample
visualize(
    image=image, 
    pore=mask[..., 0].squeeze(),
    )
    
# Dataset for train images
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    classes=CLASSES, class_values=[255], row=192,column=192,
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    classes=CLASSES, class_values=[255], row=192,column=192,
)

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    classes=CLASSES,  class_values=[255],  row=192,column=192,
)
