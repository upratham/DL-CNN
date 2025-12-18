import tensorflow as tf 
from keras import layers,models

def build_cnn(droup_outsize,n_classes):
    cnn = models.Sequential([
        layers.Input(shape=(200, 200, 3)),

        layers.Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(droup_outsize),

        layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(droup_outsize),

        layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(droup_outsize),

        layers.Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"),  
        #last layer no max pooling
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax")
    ])

    return cnn

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model.summary()