#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def display_class_images(directory_path):
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    categories = sorted(os.listdir(directory_path))
    for ax, category in zip(axes, categories):
        category_folder = os.path.join(directory_path, category)
        images = os.listdir(category_folder)
        selected_image = random.choice(images)
        img = Image.open(os.path.join(category_folder, selected_image))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"Category: {category}")
    plt.show()

# Example usage
directory_path = 'data/processed/test'
display_class_images(directory_path)

