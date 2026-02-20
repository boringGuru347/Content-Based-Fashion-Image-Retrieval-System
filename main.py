import pickle
# from locale import normalize
from annoy import AnnoyIndex
import streamlit as st
import tensorflow as tf
# from keras.src.utils import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import os
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors



feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPool2D(),
])

# u = AnnoyIndex(len(feature_list[0]), 'euclidean')
# for i in range(len(feature_list)):
#     u.add_item(i, feature_list[i])
# u.build(10)
# u.save('fashion.ann')

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)
    return normalized_result

def recommend(img_features, features_list):
    u = AnnoyIndex(2048, 'euclidean')
    u.load('fashion.ann')
    indices1 = u.get_nns_by_vector(img_features, 5)
    return indices1

upload_img = st.file_uploader('Choose an image')

if upload_img is not None:
    if save_uploaded_file(upload_img):
        display_img = Image.open(upload_img)
        st.image(display_img)
        features = feature_extraction(os.path.join('uploads', upload_img.name), model)
        # print(features)
        # st.text(features)
        indices = recommend(features, feature_list)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0]])

        with col2:
            st.image(filenames[indices[1]])

        with col3:
            st.image(filenames[indices[2]])

        with col4:
            st.image(filenames[indices[3]])

        with col5:
            st.image(filenames[indices[4]])

        # with col5:
        #     st.image(filenames[indices[0][0]])
    else:
        st.header('Some error occurred in file upload')