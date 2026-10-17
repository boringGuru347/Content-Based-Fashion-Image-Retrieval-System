# 👗 Content-Based Fashion Image Retrieval System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-red?logo=streamlit)
![Annoy](https://img.shields.io/badge/ANN-Annoy-green)
![Precision@1](https://img.shields.io/badge/Precision%401-87.4%25-brightgreen)
![MRR](https://img.shields.io/badge/MRR-0.910-brightgreen)

A high-performance Content-Based Image Retrieval (CBIR) system that recommends visually similar fashion products for any uploaded image. It leverages a pretrained **ResNet50** CNN to extract deep visual embeddings and uses **Approximate Nearest Neighbor (ANN)** search via Spotify's **Annoy** for sub-100ms similarity lookup — all delivered through an interactive **Streamlit** web application with confidence scoring and an empirical Information Retrieval evaluation suite.

Live Demo: [Streamlit Cloud](https://content-based-fashion-image-retrieval-system-2xnpmt5qhyy33gcke.streamlit.app/)

---

## 🔍 How It Works

The system operates in two decoupled stages:

### Stage 1 — Offline Feature Extraction & Indexing (`app.py`)
1. All catalog images are loaded from `fashion-dataset/images/`.
2. Each image is resized to **224×224** and passed through a pretrained **ResNet50** (ImageNet weights, classification head removed) followed by **GlobalMaxPooling2D**, outputting a **2048-dimensional** representation.
3. Batched inference (`batch_size=64`) saturates GPU/CPU throughput, accelerating full-catalog indexing by over 10x.
4. Extracted feature vectors are **L2-normalized** onto the unit hypersphere:
   $$\hat{v} = \frac{v}{\|v\|_2}$$
5. An **Annoy index** (`fashion.ann`) with 10 random projection trees is automatically constructed and saved alongside `embeddings.pkl` and `filenames.pkl`.

### Stage 2 — Online Serving & Retrieval (`main.py`)
1. **Zero-Disk Streaming:** Uploaded query images are processed entirely in-memory using `io.BytesIO` streams, eliminating disk writes and filename collision bugs.
2. **Resource Caching:** The ResNet50 model and 370 MB Annoy index are loaded as singletons using Streamlit's `@st.cache_resource`, reducing search latency to **<80ms**.
3. **Distance-to-Cosine Mapping:** Because all embeddings are unit-normalized ($\|u\|_2 = \|v\|_2 = 1$), squared Euclidean distance $d^2$ maps directly to Cosine Similarity:
   $$\text{Cosine Similarity} = 1 - \frac{d^2}{2} \implies \text{Match \%} = \max\left(0, 1 - \frac{d^2}{2}\right) \times 100\%$$
4. **Interactive Display:** Renders a 5-column responsive grid with visual recommendations, confidence match percentages (`🎯 XX.X% Match`), and graceful missing-image fallbacks.

```
User Upload ──► In-Memory PIL (BytesIO) ──► ResNet50 + GlobalMaxPool2D ──► L2-Normalized Vector
                                                                                  │
                                                                          Annoy ANN Search
                                                                          (fashion.ann cached)
                                                                                  │
                                                                   Top-5 Items + Cosine Scores
```

---

## 📊 Empirical Evaluation Benchmark

The system includes a dedicated Information Retrieval benchmark ([`evaluate.py`](evaluate.py)) that measures categorical retrieval precision against Kaggle's ground-truth `styles.csv` metadata across 500 test queries:

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Precision@1** | **87.40%** | The top recommendation matches the query category ~9 out of 10 times |
| **Precision@5** | **79.12%** | ~4 out of every 5 items in the 5-column recommendation grid are relevant |
| **Mean Reciprocal Rank (MRR)** | **0.9095** | The first relevant product appears almost exclusively at rank 1 or rank 2 |

---

## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backbone Model** | TensorFlow / Keras ResNet50 | 2048-dim deep visual feature extraction |
| **Vector Indexing** | Spotify Annoy | Sub-linear $\mathcal{O}(\log N)$ Approximate Nearest Neighbor search |
| **Web Interface** | Streamlit | Interactive web UI with cached serving |
| **Image Processing** | Pillow, Keras Preprocessing | In-memory image decoding, channel centering, resizing |
| **Numerical Computing** | NumPy | L2-normalization, vectorized Euclidean-to-Cosine conversion |
| **Evaluation Suite** | Pandas | Precision@K and MRR computation against category metadata |
| **Serialization** | Pickle | Lightweight metadata and embedding serialization |

---

## 📁 Project Structure

```
recommender/
│
├── main.py                 # Streamlit web app: cached model serving & UI
├── app.py                  # Offline pipeline: batched feature extraction & Annoy builder
├── evaluate.py             # Quantitative evaluation: Precision@K & MRR reporting
│
├── embeddings.pkl          # Precomputed 44,441 × 2048 ResNet50 feature vectors
├── filenames.pkl           # List of 44,441 indexed image paths
├── fashion.ann             # Annoy multi-tree vector search index
│
├── fashion-dataset/        # Kaggle dataset (git-ignored)
│   ├── images/             # Source fashion images (*.jpg)
│   └── styles.csv          # Product category & style metadata
│
├── uploads/                # Directory for sample test query images
├── requirements.txt        # Pinned Python package dependencies
├── runtime.txt             # Deployment Python runtime specification
└── README.md               # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/boringGuru347/Content-Based-Fashion-Image-Retrieval-System.git
cd Content-Based-Fashion-Image-Retrieval-System
```

### 2. Create and Activate Virtual Environment
> **Recommended:** Use **Python 3.10** for full compatibility with TensorFlow 2.15 and Annoy binary wheels.

**On Windows (PowerShell):**
```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(On Windows, pre-compiled Annoy wheels can be installed directly from [Sprocketer/annoy-wheels](https://github.com/Sprocketer/annoy-wheels) if C++ build tools are not present).*

### 4. Prepare the Dataset
Download the [Kaggle Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) and place:
- The images in: `fashion-dataset/images/`
- The metadata in: `fashion-dataset/styles.csv`

---

## 🚀 Usage

### 1. Launch the Web Application
```bash
streamlit run main.py
```
Open `http://localhost:8501` in your browser, upload any fashion item, and explore instant visually similar recommendations.

### 2. Run the Evaluation Benchmark
```bash
python evaluate.py --queries 500 --k 5
```
Evaluates retrieval performance across 500 sampled queries and outputs a complete Precision@K and MRR summary table.

### 3. (Optional) Rebuild Index on a New Dataset
```bash
python app.py
```
Automatically extracts batched ResNet50 embeddings and builds a fresh `fashion.ann` index end-to-end.
