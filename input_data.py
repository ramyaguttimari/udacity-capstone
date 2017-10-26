
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import csv
import numpy as np
from ast import literal_eval


# you need to change this to your data directory
train_dir ='F:/Udacity ML/git/udacity-capstone'

def get_files():
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels

    '''
    print(tf.__version__)
    
    image_list=[]
    label_list=[]
    
    with open('F:/Udacity ML/git/udacity-capstone/images10.csv') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
                        
            image_list.append(row[0])
            temp=literal_eval(row[1])            
            label_list.append(temp)
    print (type(image_list[0]))
    print (type(label_list[0]))
    
    return image_list, label_list




def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    print ("Hello")
    print (type(image[0]))
    print (type(label[0]))
    print (image)
    print (label)
    image = tf.cast(image, tf.string)#cast image list values to string tensors
    label = tf.to_int32(label)
    #print (type(image))
    #print (type(label))


    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    #print(input_queue[0])
    #print (input_queue[1])
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    #image_batch = tf.cast(image_batch, tf.float32)
    # 
    #image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 2, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    #image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


 
#TEST
# To test the generated batches of images
# When training the model, DO comment the following codes




import matplotlib.pyplot as plt

BATCH_SIZE = 2
CAPACITY = 10
IMG_W = 512
IMG_H = 512


image_list, label_list = get_files()
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        while not coord.should_stop() and i<1:
            
            img, label = sess.run([image_batch, label_batch])
            
            # just test one batch
            for j in np.arange(BATCH_SIZE):
                print('label: %d' %label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i+=1
            
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    #coord.join(threads)


#