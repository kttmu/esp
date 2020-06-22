import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from config import width, height, train_path, train_ano_path#, val_path, val_ano_path


def center_normalize(x):
    return (x - np.mean(x)) / np.std(x)


def data_split(image_dir, ann_dir):
    #changed point
    #images = glob.glob(image_dir + "*")
    #annotation = glob.glob(ann_dir + "*")
    #substantially added
    images = glob.glob(train_path + "*")
    annotation = glob.glob(train_ano_path + "*")

    #img_dict = {i.split("/")[-1].split(".")[0]: i for i in images}
    #ann_dict = {i.split("/")[-1].split(".")[0].rstrip("_segmentation"): i for i in annotation}
    #added point
    img_dict = {i.split(".")[-2].split("/")[-1]: i for i in images}
    ann_dict = {i.split("/")[-1].split("_")[-3]: i for i in annotation}

    #print(img_dict)
    #print(ann_dict)
    concat_dict = {}
    for key, value in img_dict.items():
        for k, v in ann_dict.items():
            if k == key:
                concat_dict[value] = v

    img_list = []
    ann_list = []
    for im, an in concat_dict.items():
        #print("im_dir: " + im)
        #print("ano_dir: " + an)
        #cv2.imshow('name', cv2.imread(im))
        #src = input()
        img_list.append(cv2.resize(cv2.imread(im), (width, height), interpolation=cv2.INTER_NEAREST))
        ann_list.append(cv2.resize(cv2.imread(an, 0), (width, height), interpolation=cv2.INTER_NEAREST))
        #cv2.imshow("name", cv2.imread(im))

    img_arr = np.array(img_list)
    ann_arr = np.array(ann_list)
    ann_arr = ann_arr / 255

    X_Train, X_Test, y_Train, y_Test = train_test_split(img_arr, ann_arr, test_size=0.2, random_state=42)

    X_Train = center_normalize(X_Train)
    X_Test = center_normalize(X_Test)
    print("over read")
    #src = input()
    return X_Train, X_Test, y_Train, y_Test
