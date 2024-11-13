#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_dir, batch_size=32):
    model = load_model(model_path)
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir, target_size=(256, 256), 
                                                                       batch_size=batch_size, class_mode='categorical', 
                                                                       shuffle=False)
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    predictions = model.predict(test_gen)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_gen.classes
    misclassified_indices = np.where(predicted_labels != true_labels)[0]

    return misclassified_indices, test_gen, predicted_labels, true_labels

# Example usage
model_path = "model_augmented.h5"
test_dir = 'data/processed/test'
evaluate_model(model_path, test_dir)

