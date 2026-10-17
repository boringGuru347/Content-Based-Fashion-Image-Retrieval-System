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



with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tf.keras.Sequential([
        base_model,
        GlobalMaxPool2D(),
    ])

@st.cache_resource
def load_annoy_index():
    u = AnnoyIndex(2048, 'euclidean')
    u.load('fashion.ann')
    return u

model = load_model()

# u = AnnoyIndex(len(feature_list[0]), 'euclidean')
# for i in range(len(feature_list)):
#     u.add_item(i, feature_list[i])
# u.build(10)
# u.save('fashion.ann')

st.title('Fashion Recommender System')

def feature_extraction(img_file, model):
    img = Image.open(img_file).convert('RGB').resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / np.linalg.norm(result)
    return normalized_result

def recommend(img_features):
    u = load_annoy_index()
    indices1, distances = u.get_nns_by_vector(img_features, 5, include_distances=True)
    scores = [max(0.0, 1.0 - (d ** 2) / 2.0) * 100.0 for d in distances]
    return indices1, scores

upload_img = st.file_uploader('Choose an image')

if upload_img is not None:
    display_img = Image.open(upload_img)
    st.image(display_img)
    features = feature_extraction(upload_img, model)
    indices, scores = recommend(features)
    cols = st.columns(5)
    for col, idx, score in zip(cols, indices, scores):
        with col:
            img_path = filenames[idx]
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning(f"Image missing:\n`{os.path.basename(img_path)}`")
            st.caption(f"🎯 **{score:.1f}% Match**")