# Garbage Classification with TensorFlow and Keras

## Project Overview
This project classifies images of garbage into various categories using Convolutional Neural Networks (CNNs) in TensorFlow/Keras. It includes data preparation, model training with and without data augmentation, and evaluation on test images. The project aims to improve model performance with augmented data and provides a visual analysis of misclassified images.

## Objectives
- Split the dataset into train, validation, and test sets.
- Train a CNN model to classify garbage images.
- Enhance model performance using data augmentation.
- Evaluate the model on test data and visualize misclassified examples.

## Requirements
Install the following packages using ```requirements.txt```:

```bash
numpy==1.21.0
matplotlib==3.4.2
tensorflow==2.5.0
scikit-learn==0.24.2
pillow==8.2.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation:** Run data_preparation.py to split the images into train, validation, and test sets.
2. **Model Training:** Use model_training.py to train the model with and without augmentation.
3. **Evaluation:** Evaluate model accuracy on the test set using evaluation.py.
4. **Visualization:** Use visualization.py to display sample images and analyze misclassified images.

```bash
python src/data_preparation.py
python src/model_training.py
python src/evaluation.py
python src/visualization.py
```
## Data Preparation
The data preparation script data_preparation.py splits the dataset into training, validation, and test sets. It organizes the data into separate directories for each class and applies the necessary transformations.

## Model Training
The model_training.py script builds and trains a convolutional neural network (CNN) using the training and validation datasets. Data augmentation techniques such as rotation, shifting, and zooming are applied to improve the model's robustness.

## Key Model Architecture
Convolutional layers with ReLU activation MaxPooling layers Dense layers with Dropout for regularization Output layer with softmax activation for multi-class classification

## Data Preparation
The data preparation script data_preparation.py splits the dataset into training, validation, and test sets. It organizes the data into separate directories for each class and applies the necessary transformations.

## Model Training
The model_training.py script builds and trains a convolutional neural network (CNN) using the training and validation datasets. Data augmentation techniques such as rotation, shifting, and zooming are applied to improve the model's robustness.

**Key Model Architecture**
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dense layers with Dropout for regularization
- Output layer with softmax activation for multi-class classification

## Model Evaluation
The model_evaluation.py script evaluates the trained model on the test dataset. It reports the model's accuracy and identifies misclassified images for further analysis.

## Results

**Without Data Augmentation**
- Test Accuracy: 62.40%
- Training Time: Shorter per epoch
**With Data Augmentation**
- Test Accuracy: 69.30%
- Training Time: Longer per epoch
**Unseen Data**
- Test Accuracy on Unseen Data: 88.24%
The model with data augmentation achieved higher accuracy, demonstrating the effectiveness of augmentation in improving model generalization.

## Results
The project assesses model accuracy with and without data augmentation, and visualizes misclassified examples for further analysis.

## Key Points
- **Augmentation Effect:** Data augmentation significantly improved model performance by enabling the model to generalize across varied conditions, such as slight rotations, shifts, and zooms.
- **Misclassification Insights:** Analyzing misclassified images showed potential improvements in model architecture to handle challenging cases, especially for classes with similar visual characteristics.

## Applications
This project demonstrates a practical application in waste management:

- **Automated Waste Sorting:** By accurately classifying waste types, this model can be integrated into automated waste sorting systems to improve recycling efficiency.
- **Environmental Impact:** Improved waste sorting directly supports better recycling practices, contributing to environmental sustainability by reducing landfill waste.

## Future Enhancements
- **Advanced Architectures:** Implementing architectures such as ResNet or Inception could improve accuracy further, especially for challenging classes.
- **Hyperparameter Tuning:** Experimenting with different hyperparameters like learning rate, dropout rates, and optimizer configurations could enhance model performance.
- **Larger Dataset:** Collecting more diverse samples or using transfer learning could increase model generalization.

## Acknowledgement
This project was conducted as part of an assessment and the dataset used is [![TrashNet Dataset]](https://www.kaggle.com/datasets/feyzazkefe/trashnet)


## Contributors
Pradipta Dutta - Data Scientist
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pradiptadutta63)
