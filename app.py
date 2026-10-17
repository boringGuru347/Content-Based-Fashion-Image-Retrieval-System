import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm
import pickle
from annoy import AnnoyIndex

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPool2D(),
])

def extract_features_batch(img_paths, model):
    batch_images = []
    for p in img_paths:
        img = image.load_img(p, target_size=(224, 224))
        batch_images.append(image.img_to_array(img))
    batch_array = np.array(batch_images)
    preprocessed_batch = preprocess_input(batch_array)
    results = model.predict(preprocessed_batch, verbose=0)
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (results / norms).tolist()

dataset_dir = os.path.join('fashion-dataset', 'images')
arr = os.listdir(dataset_dir)
arr_img = [
    os.path.join('fashion-dataset/images', item).replace('\\', '/')
    for item in arr
    if item.lower().endswith(('jpg', 'jpeg', 'png'))
]

print(f"Found {len(arr_img)} images to index.")

batch_size = 64
feature_list = []

for i in tqdm(range(0, len(arr_img), batch_size), desc="Extracting features"):
    batch_paths = arr_img[i:i + batch_size]
    feature_list.extend(extract_features_batch(batch_paths, model))

with open('embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)

with open('filenames.pkl', 'wb') as f:
    pickle.dump(arr_img, f)

print("Building Annoy index...")
dim = len(feature_list[0])
u = AnnoyIndex(dim, 'euclidean')
for i, vec in enumerate(feature_list):
    u.add_item(i, vec)
u.build(10, n_jobs=-1)
u.save('fashion.ann')
print("Offline indexing pipeline complete! Saved embeddings.pkl, filenames.pkl, and fashion.ann.")