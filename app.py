from turtle import mode

import tensorflow as tf
# from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False


model = tf.keras.Sequential([
    model,
    GlobalMaxPool2D(),
])
# print(model.summary())

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)  # FIXED
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / np.linalg.norm(result)

    return normalized_result

arr = os.listdir('fashion-dataset/images')
arr_img = []
for item in arr:
    if item.endswith('jpg'):
        arr_img.append(os.path.join('fashion-dataset/images', item))
# print(arr_img[0:6])

feature_list = []

for file in tqdm(arr_img):
    feature_list.append(extract_features(file, model))

# print(np.array(feature_list).shape)
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(arr_img, open('filenames.pkl', 'wb'))