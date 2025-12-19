import tensorflow as tf 
from keras import layers,models
import tensorflow as tf
from keras.applications import VGG16
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def build_cnn(droup_outsize,n_classes):
    cnn = models.Sequential([
        layers.Input(shape=(200, 200, 3)),
        
        layers.Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        

        layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(droup_outsize),

        layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(droup_outsize),

        layers.Conv2D(512, (3, 3), strides=1, padding="same", activation="relu"),  
        #last layer no max pooling
        layers.GlobalAveragePooling2D(), # instead off flatten to handle GPU memory allocation 
        layers.Dense(512, activation="relu"), 
        layers.Dropout(0.25),
        layers.Dense(n_classes, activation="softmax",dtype="float32")

    ])
    print(cnn.summary())


    return cnn



def build_VGG16(droup_outsize,n_classes,trasnfer_flag=False,input_shape=(200, 200, 3)):
 
    conv_base = VGG16(
        weights="imagenet",
        include_top=trasnfer_flag,              # remove VGG16 classifier
        input_shape=input_shape
    )
    conv_base.trainable = trasnfer_flag        # freeze feature extractor

    model = models.Sequential([
        conv_base,
        layers.GlobalAveragePooling2D(),# instead off flatten,GlobalAveragePooling2D is used  to handle GPU memory allocation 
        layers.Dense(512, activation="relu"),
        layers.Dropout(droup_outsize),
        layers.Dense(n_classes, activation="softmax",dtype="float32")
    ])
    print(model.summary())

    return model





def main():
   # build_cnn(droup_outsize=0.15,n_classes=10)
    build_VGG16(droup_outsize=0.15,n_classes=10)

if __name__== '__main__':
    main()