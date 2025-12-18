import os;import re;import numpy as np
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tqdm import tqdm 
import cv2

def data_preprocess(img_size,data_dir):
    label_names = ["bougainvillea","daisies","garden_roses","gardenias","hibiscus","hydrangeas","lilies","orchids","peonies","tulip"]
    Images= []
    labels= []
    label_idx=[]

    # --------------label encoding and image resize -------------# 
    for f in tqdm(glob.glob(data_dir+'\*.jpg')):
        #split on first number and remove last character
        name = re.split(r"\d", os.path.splitext(os.path.basename(f))[0], 1)[0][:-1] 
        idx=label_names.index(name)
        image = cv2.imread(f)
        image = cv2.resize(image,img_size)
        labels.append(name)
        Images.append(image)
        label_idx.append(idx)
        
        # print("\nExtracting")
        # print(name)
        # print(idx)
        # #print(image)
    return Images, labels, label_idx








    






def main():
    img_size=(200,200)
    path=r'data/flowers'
    Images,labels,label_idx=data_preprocess(img_size=img_size,data_dir=path)
    Images=np.array(Images)
    label_idx=np.array(label_idx)
    print('img shape:',Images.shape)
    print('label idx shape:',label_idx.shape)
    print('label length:',len(labels))

if __name__== "__main__":
    main()