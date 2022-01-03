# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:08:17 2021

@author: DELL
"""



# You'll generate plots of attention in order to see which parts of an image
# your model focuses on during captioning
import matplotlib.pyplot as plt

# import collections
# import random
import numpy as np
import os
# import time
# import json
from PIL import Image
# import pickle
# import os
import streamlit as st
import tensorflow as tf
# import imageio

# from tensorflow.keras.preprocessing import image
def process_path(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    img = tf.image.resize(img, (96, 128), method='nearest')
    
    return img


def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

st.title('Image SEGMENTATION')

uploaded_image = st.file_uploader('Choose an image')

unet=tf.keras.models.load_model('unet_model_v4.h5')


if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        

        # extract the features
        
        image_url = os.path.join('uploads',uploaded_image.name)
        image_extension = image_url[-4:]
        image_path = os.path.join('uploads',uploaded_image.name)#tf.keras.utils.get_file('image'+image_extension, origin=image_url)
        # print(image_path)

#         input_img=process_path(image_path)
#         input_img=input_img[tf.newaxis,...]

#         mask=unet.predict(input_img)
#         pred_mask = tf.argmax(mask, axis=-1)
#         pred_mask = pred_mask[..., tf.newaxis]
#         pred_mask=pred_mask[0]
#         input_img=input_img[0]
#         #print('Prediction Caption:', ' '.join(result))
#         #plot_attention(image_path, result, attention_plot)
#         # opening the image
        
#         fig, arr = plt.subplots(1, 2, figsize=(14, 10))
#         arr[0].imshow(tf.keras.preprocessing.image.array_to_img(input_img))
#         arr[0].set_title('Image')
#         arr[1].imshow(tf.keras.preprocessing.image.array_to_img(pred_mask))
#         arr[1].set_title('Segmentation')
#         plt.savefig('foo.png')
#         # col1 = st.beta_columns(1)

        display_image = Image.open("foo.png")
        # with col1:
            # st.header('Your uploaded image')
        st.image(display_image)
        # with col2:
        #     st.header('Your Segmentated image')
        #     st.image(tf.keras.preprocessing.image.array_to_img(pred_mask))
        os.remove(image_path)
