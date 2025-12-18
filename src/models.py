import tensorflow as tf 
from keras import layers,models
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def build_cnn(droup_outsize,n_classes):
    cnn = models.Sequential([
        layers.Input(shape=(200, 200, 3)),
        layers.Rescaling(1./255), # input image pixel normalization 

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
        layers.Dropout(0.25),
        layers.Dense(n_classes, activation="softmax")

    ])
    print(cnn.summary())


    return cnn

def main():
    build_cnn(droup_outsize=0.15,n_classes=10)

if __name__== '__main__':
    main()