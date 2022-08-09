from pathlib import Path

import sys
import cv2
import numpy as np

from .base import BaseDataset
np.set_printoptions(threshold=sys.maxsize)

class ToyDataset(BaseDataset):
    def __init__(self, dataset_path, split, **kwargs):
        super(ToyDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split

        self.dataset_samples = []
        images_path = sorted((self.dataset_path / split).rglob('*rgb.png'))
        for image_path in images_path:
            image_path = str(image_path)
            mask_path = image_path.replace('rgb.png', 'im.png')
            self.dataset_samples.append((image_path, mask_path))

    def get_sample(self, index):
        image_path, mask_path = self.dataset_samples[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        instances_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Extract the class and instance info from the instances mask
        instances_mask, class_matrix, instance_matrix = self.get_class_instance(instances_mask)   

        sample = {'image': image}
        if self.with_segmentation:
            # The class matrix is the semantic segmentation label
            sample['semantic_segmentation'] = class_matrix
        else:
            instances_mask += 1

        instances_ids = self.get_unique_labels(instances_mask, exclude_zero=True)
        instances_info = {
            # Extract the class ID of an instance and set it in the value 
            # dictionary of the instance ID
            x: {'class_id': int(str(x)[0]), 'ignore': False}
            for x in instances_ids
        }

        sample.update({
            'instances_mask': instances_mask,
            'instances_info': instances_info,
        })

        return sample

    def get_class_instance(self, instances_mask):
        # The left number gives the class ID and the right digit gives the 
        # instance ID after dividing the intensity of the pixel by 4
        
        instances_mask = np.floor(instances_mask/4)
        instances_mask = instances_mask.astype(np.int32)

        class_matrix = np.zeros_like(instances_mask) 
        instance_matrix = np.zeros_like(instances_mask)

        for r_idx, r_pixel in enumerate(instances_mask):
            for c_idx, _ in enumerate(r_pixel):
                intensity = str(instances_mask[r_idx][c_idx])
                
                # If the intensity is a single-digit, prefix it with a 0
                if len(intensity) == 1:
                    intensity = '0' + intensity

                class_matrix[r_idx][c_idx] = int(intensity[0])
                instance_matrix[r_idx][c_idx] = int(intensity[1])   
        
        return instances_mask, class_matrix, instance_matrix
    
    @property
    def stuff_labels(self):
        return [0]

    @property
    def things_labels(self):
        return [1]
    
