# -*- coding: utf-8 -*-
# USAGE: python code_1.py
# Please change cafee_root and data_path location accordingly

import sys
import os
import glob
import numpy as np
import caffe
import matplotlib.pyplot as plt
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

## CHECK REQUIRED MODEL AND DATA FILES
# Set Caffe Root folder containing model and data files
caffe_root = './'
data_path = caffe_root+'/data/ilsvrc12/images/'
    
dependency = True

# Check model files
if not os.path.isfile(caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    dependency = False
    print('Please download pre-trained CaffeNet model using the below python script:')
    print('caffe_root/scripts/download_model_binary.py caffe_root/models/bvlc_reference_caffenet')
    print()
    
# Check ImageNet labels
labels_file = caffe_root + '/data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    dependency = False
    print('Please download labels_file using: caffe_root/data/ilsvrc12/get_ilsvrc_aux.sh')
    print()
    
# Check ImageNet validation data
fileList = glob.glob(data_path+'/*')
if len(fileList) == 0:
    dependency = False
    print('Please put validation images to folder caffe_root/data/ilsvrc12/images/')
    print('The train/validation dataset can be downloaded at http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar')
    print()

if not dependency:
    sys.exit()
    
## LOAD MODEL AND DATA    
# Load label data
labels = np.loadtxt(labels_file, str, delimiter='\t')

# Load pre-trained Caffe model
caffe.set_mode_cpu()
model_def = caffe_root + '/models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the BGR mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

## PREDICTION
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# prediction for first batch_size images in validation data folder
batch_size = 10
if len(fileList) < batch_size:
    batch_size = len(fileList)
    print("Warning: There are only %i validation image files." %batch_size)

net.blobs['data'].reshape(batch_size,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227
imageList = []
for file in fileList[0:batch_size]:
    image = caffe.io.load_image(file)
    imageList.append(image)

inputs = np.array([transformer.preprocess('data', image) for image in imageList])

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = inputs

# perform classification
output = net.forward()
output_prob = output['prob'] # the output probability vector
prediction = np.argmax(output['prob'], axis=1)

## VISUALIZATION
# display image and predicted label:
for i in range(len(imageList)):
    plt.figure(i+1)
    plt.title('Predicted label: %s' % labels[prediction[i]], loc='left')
    plt.imshow(imageList[i]); plt.axis('off')


