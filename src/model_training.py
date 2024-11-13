#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def make_model(class_num=6, input_shape=(256, 256, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(class_num, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=0.0001),
                  metrics=['acc'])
    return model

def train_model(model, train_dir, val_dir, batch_size=32, epochs=20, augment=False):
    if augment:
        data_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, 
                                      shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    else:
        data_gen = ImageDataGenerator(rescale=1./255)

    train_data = data_gen.flow_from_directory(train_dir, target_size=(256, 256), batch_size=batch_size, class_mode='categorical')
    val_data = data_gen.flow_from_directory(val_dir, target_size=(256, 256), batch_size=batch_size, class_mode='categorical')

    history = model.fit(train_data, validation_data=val_data, steps_per_epoch=train_data.samples // batch_size,
                        validation_steps=val_data.samples // batch_size, epochs=epochs)
    return model, history

# Example usage
train_dir = 'data/processed/train'
val_dir = 'data/processed/val'
model = make_model(class_num=6)
model, history = train_model(model, train_dir, val_dir, augment=True)
model.save("model_augmented.h5")

