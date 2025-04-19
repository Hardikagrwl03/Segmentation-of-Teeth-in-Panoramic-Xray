from images_prepare import *
from masks_prepare import *
import cv2
from PIL import Image
import numpy as np
import albumentations as A
import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from model import *
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Train", help = "Training of Model", action = 'store_true')
parser.add_argument("-p", "--Predict_Path", help = "Path of Image for Predicting")
args = parser.parse_args()
train = args.Train

model=UNET(input_shape=(512,512,1),last_activation='sigmoid')
model.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

def augment_dataset(input, output):
    aug = A.Compose([
        A.OneOf([A.RandomCrop(width=512, height=512),
                 A.PadIfNeeded(min_height=512, min_width=512, p=0.5)],p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25,p=0.5),
        A.Compose([A.RandomScale(scale_limit=(-0.15, 0.15), p=1, interpolation=1),
                                A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT), 
                                A.Resize(512, 512, cv2.INTER_NEAREST), ],p=0.5),
        A.ShiftScaleRotate (shift_limit=0.325, scale_limit=0.15, rotate_limit=15,border_mode=cv2.BORDER_CONSTANT, p=1),
        A.Rotate(15,p=0.5),
        A.Blur(blur_limit=1, p=0.5),
        A.Downscale(scale_min=0.15, scale_max=0.25,  always_apply=False, p=0.5),
        A.GaussNoise(var_limit=(0.05, 0.1), mean=0, per_channel=True, always_apply=False, p=0.5),
        A.HorizontalFlip(p=0.25),
    ])
    x_train1=np.copy(input)
    y_train1=np.copy(output)
    count=0
    while(count<4):
        x_aug2=np.copy(x_train1)
        y_aug2=np.copy(y_train1)
        for i in range(len(x_train1)):
            augmented=aug(image=x_train1[i,:,:,:],mask=y_train1[i,:,:,:])
            x_aug2[i,:,:,:]= augmented['image']
            y_aug2[i,:,:,:]= augmented['mask']
        input=np.concatenate((input,x_aug2))
        output=np.concatenate((output,y_aug2))
        if count == 9:
            break
        count+=1
    print("dataset augmented !! ...")
    return input, output

def load_data(path='./data'):
    X, X_sizes = pre_images((512, 512), path, True)
    Y = pre_splitted_masks(path='./Custom_Masks')  # Custom Splitted MASKS size 512x512

    X = np.float32(X / 255)
    Y = np.float32(Y / 255)

    x_train = X[:105, :, :, :]
    y_train = Y[:105, :, :, :]
    x_test = X[105:, :, :, :]
    y_test = Y[105:, :, :, :]

    return x_train, y_train, x_test, y_test

def train_model(x_train, y_train):
    model.fit(x_train,y_train,batch_size=2,epochs=50,verbose=1)
    model.save_weights('./dental_xray_seg.h5')

def predict(path):
    img1=Image.open(path)
    img1 =img1.resize((512, 512),Image.ANTIALIAS)
    img=convert_one_channel(np.asarray(img1))
    img = np.reshape(img, (1, 512, 512, 1))
    img = np.float32(img/255)
    predicted_image = model.predict(img)
    return img1, predicted_image[0, :, :, :] 


if train == True:
    x_train, y_train, x_test, y_test = load_data(path='./data') 
    x_train, y_train = augment_dataset(x_train, y_train) 
    train_model(x_train, y_train)
    model.save_weights('./dental_xray_seg.h5')
else:
    model.load_weights('./dental_xray_seg.h5')

if __name__ == "__main__":

    input_img, predicted_img= predict(args.Predict_Path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(input_img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Predicted Segmented Mask")
    plt.imshow(predicted_img, cmap='gray')
    plt.show()