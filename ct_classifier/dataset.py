'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''


import os
import json
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import math


class CTDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split

        class FixedHeightResize:
            def __init__(self, size):
                self.size = size
                
            def __call__(self, img):
                w,h = img.size
                aspect_ratio = float(h) / float(w)
                if h>w:
                    new_w = math.ceil(self.size / aspect_ratio)
                    img = T.functional.resize(img, (self.size, new_w))
                else:
                    new_h = math.ceil( aspect_ratio * self.size) # it eas / before diving
                    img = T.functional.resize(img, (new_h,self.size))

                w,h = img.size # PIL image formats are in w and h, transformed to rgb, h, w later, and needs to see size, Tensors # are seen in shape
                pad_diff_h = self.size - h 

                pad_diff_w =self.size - w
                
                padding = [0, pad_diff_h, pad_diff_w, 0]
                padder = T.Pad(padding)
                img = padder(img)

                return img



        self.transform = T.Compose([
            FixedHeightResize(cfg['image_size'][0]),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            T.ColorJitter(brightness=.5, 
                         contrast=.3, 
                         saturation=.5),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.RandomRotation(degrees=(0, 180)), 
            T.RandomCrop(size=cfg['image_size'][0]),   
            T.ToTensor()
        ])
        
        # index data into list
        self.data = []

        # load annotation file
        annoPath = os.path.join(
            self.data_root,
            'train/train.json' if self.split=='train' else 'val/val.json'
        )
        meta = json.load(open(annoPath, 'r'))

        images = dict([[i['id'], i['file_name']] for i in meta['images']])          # image id to filename lookup
        labels = dict([[c['id'], idx] for idx, c in enumerate(meta['categories'])]) # custom labelclass indices that start at zero
        
        # since we're doing classification, we're just taking the first annotation per image and drop the rest
        images_covered = set()      # all those images for which we have already assigned a label
        for anno in meta['annotations']:
            imgID = anno['image_id']
            if imgID in images_covered:
                continue
            
            # append image-label tuple to data
            imgFileName = images[imgID]
            label = anno['category_id']
            try:
                labelIndex = labels[label]
            except:
                print('bad', anno['image_id'])
            self.data.append([imgFileName, labelIndex])
            images_covered.add(imgID)       # make sure image is only added once to dataset
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = os.path.join(self.data_root,
                                   'train/' if self.split=='train' else 'val/', 
                                   image_name)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)

        return img_tensor, label