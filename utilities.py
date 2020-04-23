import matplotlib.pyplot as plt
import random 
from sklearn.model_selection import train_test_split
import pickle 
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np 

AUTOTUNE = tf.data.experimental.AUTOTUNE #to parallelize the data augmentation

import glob #useful working with paths
import cv2 #useful working with video and images



####################################### FROM VIDEO TO IMAGES ###################################################


def vid_to_jpg(file_num):
    vidcap = cv2.VideoCapture('./cnr_issia_soccer/filmrole' + str(file_num) + '.avi')
    success, image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite("./dataset/cnr_issia_soccer_dataset/imgs_cam%d/%d.jpg" % (file_num, count), image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
    return ('done')


####################################### DATA EXTRACTION ###################################################


def get_data(cams=[1, 3, 4], no_ball='all', batch_size=4, val_split=False, reduction='none'):
    """
    cams: each cam 1-6 is a cam recording a part of the game pitch
    no_ball_freq: number of images without the ball to have in our dataset with respect to the ball one
    batch_size: batch_size that we will use. Needed to have already right dimension dataset
    val_split: validation set size with respect to the train
    reduction: factor to reduce the size of our dataset
    """
    random.seed(42)
    #import data
    ball_pos={} #dict with ball positions
    play_pos={} #dict with player positions
    imgs=[] #list with valid paths (pickle dicts contain some non valid paths)
    for cam in cams: #for each cam load the associated data
        cam=str(cam)
        with open('./coordinates/ball_pos_cam'+cam+'.pickle', 'rb') as p:
            b = pickle.load(p)
        with open('./coordinates/player_pos_cam'+cam+'.pickle', 'rb') as p:
            p = pickle.load(p)
        img=[s.replace('\\', '/') for s in glob.glob("./dataset/cnr_issia_soccer_dataset/imgs_cam"+cam+"/*.jpg")]
        ball_pos.update(b)
        play_pos.update(p)
        imgs.extend(img)
    
    imgs=set(imgs)    
    ball_imgs=set(ball_pos.keys())
    play_imgs=set(play_pos.keys())
    
    #take only images with annotation
    im_path=ball_imgs.union(play_imgs)
    if no_ball=='no':
        im_path=ball_imgs
    if no_ball=='only':
        im_path=play_imgs-ball_imgs
    #delete unvalid paths
    im_path=list(imgs.intersection(im_path)) 
    
    
    #apply subsampling to get less data
    if reduction is not 'none':
        im_path=random.sample(im_path, int(reduction*len(im_path)))
    
    #create tensor with correct label
    labels=[]
    for im in im_path:
        label=[0,0]
        if im in ball_imgs:
            coords = ball_pos[im]
            x, y= int(coords[0][0]),int(coords[0][1])
            x, y=int(tf.math.floor(x/4)), int(tf.math.floor(y/4))            
            if x==0: x=1
            if y==0: y=1
            if y==267: y=266
            if y==268: y=266
            if x==479: x=478
            label=[x,y] 
        labels.append(label)
        
    # create one ore two tensorflow dataset from tensor slices
    if val_split==False:
        max_length=len(im_path)-len(im_path)%4
        paths = tf.constant(im_path)[0:max_length]
        labels = tf.constant(labels)[0:max_length]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        
        return dataset
    
    else:
        
        train_path, val_path, train_labels, val_labels = train_test_split(
            im_path, labels, test_size=val_split, random_state=42)
        
        max_length_train=len(train_path)-len(train_path)%batch_size
        train_paths = tf.constant(train_path)[0:max_length_train]
        train_labels = tf.constant(train_labels)[0:max_length_train]
        trainset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        
        max_length_val=len(val_path)-len(val_path)%batch_size
        val_paths = tf.constant(val_path)[0:max_length_val]
        val_labels = tf.constant(val_labels)[0:max_length_val]
        valset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        
        return trainset,valset

    
####################################### PRE TRAIN PLOTS ###################################################
    
def no_ball():
    ball_map=tf.zeros((268,480))
    return ball_map
def ball(x,y):
    ball_map=tf.zeros((268,480))
    #tensors does not support 2d indices, so "slices"==rows are created and inserted at the right position
    indices =([[y-1],[y+1],[y]]) 
    rep= tf.concat([tf.zeros(x-1), tf.ones(3),tf.zeros(478-x)], -1)
    updates = [rep,rep,rep]
    ball_map= tf.tensor_scatter_nd_update(ball_map, indices, updates)
    return ball_map

def preprocess(path, label):
  
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3) 
    image = tf.image.resize_with_crop_or_pad(image, 1080, 1920) 
    image = tf.cast(image, tf.float32) / 255.0
    label=tf.cond(label[0]==tf.constant([0]), lambda: no_ball() , lambda:ball(label[0],label[1]))  
 
    return image, label


def augmenter(image, label):
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, upper=1.5, lower=0.5)
    image = tf.image.random_saturation(image,upper=1.5, lower=0.5)
    image = tf.image.random_hue(image, 0.15)
    if tf.random.uniform([])>0.5:
        image= tf.image.flip_left_right(image)
        label= tf.image.flip_left_right(tf.expand_dims(label, axis=-1))
        label= tf.squeeze(label, axis=-1)
    if tf.random.uniform([])>0.75:
        a= tf.random.uniform([],minval=0, maxval=67, dtype=tf.int32)
        b= tf.random.uniform([],minval=0, maxval=120, dtype=tf.int32)
        c= tf.random.uniform([],minval=a+134, maxval=268, dtype=tf.int32)-a
        d= tf.random.uniform([],minval=b+240, maxval=480, dtype=tf.int32)-b
        image=tf.image.crop_to_bounding_box(image, a*4,b*4, c*4, d*4)
        image=tf.image.pad_to_bounding_box(image,a*4, b*4, 1080, 1920)
        label=tf.expand_dims(label, axis=-1)
        label=tf.image.crop_to_bounding_box(label, a,b,c,d)
        label=tf.image.pad_to_bounding_box(label,a,b, 268, 480)
        label=tf.squeeze(label, axis=-1)
    return image, label


def plot_pretrain(data, batch=4, how_many=1, augment=False):
    if augment==False:
        dataset= data.map(preprocess).shuffle(100).batch(batch).take(how_many)
    else:
        dataset= data.map(preprocess).map(augmenter).shuffle(100).batch(batch).take(how_many)
    for xb, yb in dataset:
        for i in range(0,4):
            f = plt.figure(figsize=(15,10))
            
            f.add_subplot(2,2,1)
            plt.imshow(xb[i].numpy())
            plt.title('Plain image')

            f.add_subplot(2,2,2)
            a = yb[i]
            im=cv2.resize(xb[i].numpy(), (480, 270))
            plt.imshow(im[0:268,0:480])
            plt.imshow(a,alpha=0.6, cmap='Oranges')
            plt.title('Ground truth projected')

####################################### HISTORY PLOT ###################################################
            
def plot_history(history, title):
    f = plt.figure(figsize=(15,4))
    plt.suptitle(title)
    f.add_subplot(1,2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['Train loss', 'Val loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    f.add_subplot(1,2, 2)
    plt.plot(history['ball_detection_accuracy'])
    plt.plot(history['val_ball_detection_accuracy'])
    plt.legend(['Train Accuracy', 'Val accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    f.savefig('plots/'+title+'.png')            

    
####################################### POST TRAIN PLOTS ###################################################


def plot_post(model, data, trash=0.35):  
    dataset=data.map(preprocess).batch(4).take(1)
    for xb, yb in dataset:
        pred=model.predict(xb)
        for i in range(0,4):
            f = plt.figure(figsize=(15,10))
            f.add_subplot(2,2, 1)

            plt.imshow(xb[i].numpy())
            plt.title('Plain image')

            f.add_subplot(2,2, 2)
            im=cv2.resize(xb[i].numpy(), (480, 270))
            plt.imshow(im[0:268,0:480])
            plt.imshow(yb[i],alpha=0.7, cmap='Oranges')
            plt.title('Ground truth')

            f.add_subplot(2,2, 3)
            mappa=pred[i]
            plt.imshow(mappa)
            plt.title('Confidence map')

            f.add_subplot(2,2, 4)
            im=(xb[i].numpy())
            if tf.math.reduce_max(mappa)> trash:
                cent=tf.unravel_index(tf.math.argmax(tf.reshape(mappa,[-1])), [268,480])
                cv2.circle(im, (cent[1]*4, cent[0]*4), 5, (1,0,0), 3)
                
            plt.imshow(im)
            plt.title('Result')
            
            
def show_result(model,data, trash=0.35, take=1):
    dataset=data.map(preprocess).batch(4).take(take)
    for xb, yb in dataset:
        for i in range(0,4):    
            mappa=model.predict(xb)[i]

            plt.figure(figsize=(20,10))
            im=(xb[i].numpy())
            if tf.math.reduce_max(mappa)> trash:
                cent=tf.unravel_index(tf.math.argmax(tf.reshape(mappa,[-1])), [268,480])
                cv2.circle(im, (cent[1]*4, cent[0]*4), 7, (1,0,0), 5)
            plt.imshow(im)
            
            
def show_errors(model,data, trash=0.35, take=15):
    dataset=data.map(preprocess).batch(4).take(take)
    for xb, yb in dataset:
        for i in range(0,4):    
            mappa=model.predict(xb)[i]
            mappa_true=yb[i]
            
            im=(xb[i].numpy())
            if tf.math.reduce_max(mappa)> trash:
                if tf.math.reduce_max(mappa_true)==1:                
                    if tf.math.reduce_max(mappa)!=(tf.math.reduce_max(tf.where(mappa_true==1, mappa, 0))):
                        cent=tf.unravel_index(tf.math.argmax(tf.reshape(mappa,[-1])), [268,480])
                        cv2.circle(im, (cent[1]*4, cent[0]*4), 7, (1,0,0), 5)
                        plt.figure(figsize=(20,10)) 
                        plt.title('wrong place detection')
                        plt.imshow(im)
                else:
                    cent=tf.unravel_index(tf.math.argmax(tf.reshape(mappa,[-1])), [268,480])
                    cv2.circle(im, (cent[1]*4, cent[0]*4), 7, (1,0,0), 5)
                    plt.figure(figsize=(20,10)) 
                    plt.title('wrong detection')
                    plt.imshow(im)
            else:
                if tf.math.reduce_max(mappa_true)==1:              
                    plt.figure(figsize=(20,10))
                    plt.title('missed detection')
                    plt.imshow(im)
                

