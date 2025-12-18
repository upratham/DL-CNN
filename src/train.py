import data_preprocess
import models
import numpy as np
from sklearn.model_selection import train_test_split



img_size=(200,200)
path=r'data/flowers'
Images,labels,label_idx=data_preprocess.data_preprocess(img_size=img_size,data_dir=path)
Images=np.array(Images)
label_idx=np.array(label_idx)
print('img shape:',Images.shape)
print('label idx shape:',label_idx.shape)
print('label length:',len(labels))

X_train,y_train,X_test,y_test=train_test_split(Images,label_idx,test_size=0.2,random_state=21,stratify=label_idx)

cnn=models.cnn
cnn.summary()
