import data_preprocess
import models
import train 
import numpy as np

#### ----- Data Preprocessing -----####
img_size=(200,200)
path=r'D:\MSML\SEM 1\ML\ML Projects\Project 3\DL-CNN\data\flowers'
label_names = ["bougainvillea","daisies","garden_roses","gardenias","hibiscus","hydrangeas","lilies","orchids","peonies","tulip"]
Images,labels,label_idx=data_preprocess.preprocess(img_size=img_size,data_dir=path,label_names=label_names)
Images=np.array(Images)
label_idx=np.array(label_idx)
print('img shape:',Images.shape)
print('label idx shape:',label_idx.shape)
print('label length:',len(labels))


unique_labels = sorted(set(labels))
print(unique_labels)
print("Number of unique labels:", len(unique_labels))


import matplotlib.pyplot as plt

counts = np.bincount(label_idx.astype(int), minlength=len(label_names))
x = np.arange(len(label_names))

plt.figure()
bars = plt.bar(x, counts, color=plt.cm.tab10(x % 10))
plt.xticks(x, label_names, rotation=90, ha="right")
plt.xlabel("Label")
plt.ylabel("Number of images")
plt.title("Images per label")

for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()


#### ----- Model build ----- ####

cnn=models.build_cnn(droup_outsize=0.15,n_classes=10)


### ---- Compile and train ----- ###


test_size=0.2
batch_size=64
n_epochs=50
X_test,y_test,model,history=train.train(Images,label_idx,cnn,n_epochs,batch_size)