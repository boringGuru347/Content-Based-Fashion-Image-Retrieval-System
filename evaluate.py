"""
Quantitative Evaluation Script for Fashion Image Retrieval System
Computes Information Retrieval (IR) metrics:
- Precision@1
- Precision@5
- Mean Reciprocal Rank (MRR)
"""

import os
import pickle
import random
import argparse
from annoy import AnnoyIndex
import pandas as pd

def load_styles_metadata():
    """Attempts to find and load styles.csv from known dataset locations."""
    possible_paths = [
        os.path.join('fashion-dataset', 'styles.csv'),
        'styles.csv',
        os.path.join('dataset', 'styles.csv')
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading metadata from: {path}")
            # on_bad_lines='skip' handles occasional malformed rows in Kaggle dataset
            df = pd.read_csv(path, on_bad_lines='skip')
            # Create mapping: str(id) -> articleType (or subCategory if articleType missing)
            cat_col = 'articleType' if 'articleType' in df.columns else 'subCategory'
            id_col = 'id' if 'id' in df.columns else df.columns[0]
            metadata = dict(zip(df[id_col].astype(str), df[cat_col].astype(str)))
            return metadata, cat_col
    return None, None

def evaluate_retrieval(num_samples=500, k=5, seed=42):
    random.seed(seed)
    
    # 1. Load filenames and Annoy index
    if not os.path.exists('filenames.pkl') or not os.path.exists('fashion.ann'):
        print("Error: filenames.pkl or fashion.ann not found.")
        return

    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)

    u = AnnoyIndex(2048, 'euclidean')
    u.load('fashion.ann')
    total_items = u.get_n_items()
    print(f"Annoy index loaded with {total_items} items.")

    # 2. Load metadata
    metadata, cat_col = load_styles_metadata()

    if metadata is None:
        print("\n[!] 'styles.csv' was not found.")
        print("    Place 'styles.csv' (from the Kaggle Fashion dataset) into 'fashion-dataset/' to compute Precision@K and MRR.")
        print("\nRunning quick self-retrieval sanity test instead...")
        sample_indices = random.sample(range(total_items), min(100, total_items))
        self_match_count = 0
        for idx in sample_indices:
            top_match = u.get_nns_by_item(idx, 1)[0]
            if top_match == idx:
                self_match_count += 1
        print(f"Self-retrieval accuracy (Querying item returns itself): {self_match_count}/{len(sample_indices)} ({self_match_count/len(sample_indices)*100:.1f}%)")
        return

    # 3. Map index positions to category labels
    index_to_cat = {}
    valid_indices = []
    for idx, path in enumerate(filenames):
        img_id = os.path.splitext(os.path.basename(path))[0]
        if img_id in metadata:
            category = metadata[img_id].strip().lower()
            if category and category != 'nan':
                index_to_cat[idx] = category
                valid_indices.append(idx)

    print(f"Matched {len(valid_indices)} / {total_items} indexed images to '{cat_col}' categories.")

    if len(valid_indices) < num_samples:
        test_indices = valid_indices
    else:
        test_indices = random.sample(valid_indices, num_samples)

    print(f"Evaluating {len(test_indices)} test queries with Top-{k} recommendations...\n")

    # 4. Compute IR Metrics
    p_at_1_scores = []
    p_at_k_scores = []
    reciprocal_ranks = []

    for query_idx in test_indices:
        query_cat = index_to_cat[query_idx]
        
        # Retrieve top k+1 (first item is query itself, so we skip index 0)
        neighbors = u.get_nns_by_item(query_idx, k + 1)
        # Exclude self if present
        retrieved = [n for n in neighbors if n != query_idx][:k]

        matches = [1 if (n in index_to_cat and index_to_cat[n] == query_cat) else 0 for n in retrieved]

        # Precision@1
        p_at_1_scores.append(matches[0] if matches else 0)

        # Precision@K
        p_at_k_scores.append(sum(matches) / len(matches) if matches else 0)

        # Reciprocal Rank (1 / rank of first relevant match)
        if 1 in matches:
            first_rank = matches.index(1) + 1
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            reciprocal_ranks.append(0.0)

    # 5. Output Summary Report
    avg_p_1 = sum(p_at_1_scores) / len(p_at_1_scores) * 100.0
    avg_p_k = sum(p_at_k_scores) / len(p_at_k_scores) * 100.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("==========================================================")
    print("      CONTENT-BASED RETRIEVAL EVALUATION REPORT           ")
    print("==========================================================")
    print(f"  Test Queries Evaluated : {len(test_indices)}")
    print(f"  Category Target        : '{cat_col}'")
    print(f"  Top-K Cutoff           : {k}")
    print("----------------------------------------------------------")
    print(f"  Precision@1            : {avg_p_1:.2f}%")
    print(f"  Precision@{k}            : {avg_p_k:.2f}%")
    print(f"  Mean Reciprocal Rank   : {mrr:.4f}")
    print("==========================================================")
    print(f"\n[Resume Highlight]")
    print(f"\"Evaluated retrieval accuracy across {total_items:,} items, achieving {avg_p_1:.1f}% Precision@1, {avg_p_k:.1f}% Precision@{k}, and {mrr:.3f} MRR using ResNet50 deep visual embeddings and Annoy ANN indexing.\"")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Image Retrieval Precision@K and MRR")
    parser.add_argument('--queries', type=int, default=500, help="Number of test queries to sample (default: 500)")
    parser.add_argument('--k', type=int, default=5, help="Top-K cutoff for recommendations (default: 5)")
    args = parser.parse_args()

    evaluate_retrieval(num_samples=args.queries, k=args.k)
