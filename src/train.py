import data_preprocess
import models
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


def train(Images,label_idx,model,epochs,batch_size):
    X_train, X_test, y_train, y_test = train_test_split(Images, label_idx,test_size=0.2,
                                                        random_state=21,stratify=label_idx)
    
   
    print(model.summary())
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)
    history = model.fit(X_train, y_train,epochs=epochs,batch_size=batch_size,validation_split=0.15,callbacks=[early_stop])

    return X_test,y_test,model,history


def main():
    img_size=(200,200)
    path=r'data/flowers'
    Images,labels,label_idx=data_preprocess.data_preprocess(img_size=img_size,data_dir=path)
    Images=np.array(Images)
    label_idx=np.array(label_idx)
    print('img shape:',Images.shape)
    print('label idx shape:',label_idx.shape)
    print('label length:',len(labels))
    test_size=0.2
    model=models.build_cnn(droup_outsize=0.15,n_classes=10)
    batch_size=64
    n_epochs=50
    X_test,y_test,model,history=train(Images,label_idx,model,n_epochs,batch_size)


if __name__=='__main__':
    main()


    
