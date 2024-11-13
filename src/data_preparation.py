#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def dataset_split(data_dir, output_dir, test_size=0.2, val_size=0.2):
    #create directories for the train, validation, and test sets
    splits = ['train', 'val', 'test']
    class_names = os.listdir(data_dir)

    for split in splits:
        for class_name in class_names:
            class_dir_out = os.path.join(output_dir, split, class_name)
            os.makedirs(class_dir_out, exist_ok=True)

    #process each class directory
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        #conditions for images check
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
        train_val, test_images = train_test_split(images, test_size=test_size, random_state=42)
        train_images, val_images = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=42)

        #function to copy images to the respective directories
        def copy_images(image_list, output_dir):
            for image in image_list:
                dest = os.path.join(output_dir, os.path.basename(image))
                shutil.copy(image, dest)

        #Copy images to their respective directories for each set
        copy_images(train_images, os.path.join(output_dir, 'train', class_name))
        copy_images(val_images, os.path.join(output_dir, 'val', class_name))
        copy_images(test_images, os.path.join(output_dir, 'test', class_name))

# Example usage
source_directory = 'data/Garbage classification'
target_directory = 'data/Split_Garbage_classification'
dataset_split(source_directory, target_directory)

