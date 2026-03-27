"""
sentence_embeddings.py : Convert TSV files to sentence embeddings using a trained FastText model.

Two embedding modes are available, selected with --sif:
1. Mean (default)
   Each sentence embedding is the unweighted mean of the FastText character vectors.

2. SIF — Smooth Inverse Frequency  (--sif flag)
   
   > The algorithm has two steps:

   Step 1 – Weighted average
       For each character token w in a sentence compute a weight
           weight(w) = a / (a + p(w))
       where a is a smoothing hyperparameter (--sif_a, default 1e-3) and
       p(w) = count(w) / total_characters is the relative frequency of w
       estimated from all input files combined. The sentence embedding is the
       weighted mean of the character vectors.

   Step 2 – Common-component removal
       Stack all sentence embeddings into a matrix and compute its first
       principal component via SVD. Subtract from every sentence vector its
       projection onto this component. This removes the dominant direction
       shared by all sentences (analogous to subtracting a "background"
       frequency effect).

   Latin/ASCII characters are treated exactly like Chinese characters (individual code-points), so SIF weights are computed for them too.

------ Output
For each input .tsv file a corresponding .npz file is written to
--output_dir (or next to the input file if --output_dir is not given).
Each .npz contains three arrays:
    > embeddings : float32 array of shape (N, dim)
    > labels : object array of shape (N,) — category strings
    > ids : int64  array of shape (N,) — index_id values

------Usage examples
Mean embeddings (default):
    python scripts/sentence_embeddings.py \
        --model models/fasttext.model \
        --input_files data/train.tsv data/dev.tsv data/test.tsv \
        --output_dir embeddings/

SIF embeddings:
    python scripts/sentence_embeddings.py \
        --model models/fasttext.model \
        --input_files data/train.tsv data/dev.tsv data/test.tsv \
        --output_dir embeddings_sif/ \
        --sif \
        --sif_a 1e-3
"""

import argparse                   
import os                         
from collections import Counter  
import numpy as np               
import pandas as pd               
from gensim.models import FastText  


# Tokenisation
def tokenize(sentence: str) -> list[str]:
    """Split a sentence into individual characters."""
    return list(sentence) 


# Mean embedding
def mean_vector(model: FastText, tokens: list[str]) -> np.ndarray:
    """
    Unweighted mean of FastText vectors for a list of character tokens.
    Returns a zero vector if no token has a vector.
    """
    #number of dimensions in each vector
    dim = model.vector_size  
    
    vectors = [model.wv[t] for t in tokens if t in model.wv]  # look up each character's vector
    if vectors:
        return np.mean(vectors, axis=0).astype(np.float32)# average all vectors into one
    # fallback: zero vector for empty sentence
    return np.zeros(dim, dtype=np.float32) 


# SIF embedding helpers
def build_freq_table(tsv_paths: list[str]) -> tuple[Counter, int]:
    """
    Count character frequencies across all input files.
    Returns (counter, total_count).
    """
    counter: Counter = Counter() # dictionary that counts occurrences
    for path in tsv_paths: # for each tsv file
        df = pd.read_csv(path, sep="\t") # first, we read it as a dataframe
        for sentence in df["text"]: # for each sentence 
            counter.update(tokenize(str(sentence))) # add this sentence's characters to the count
    total = sum(counter.values())  # total number of characters seen across all files
    return counter, total


def sif_vector(model: FastText, tokens: list[str], freq: Counter, total: int, a: float) -> np.ndarray:
    """
    SIF-weighted mean of FastText vectors.
    Returns a zero vector if no token has a vector.
    """
    dim = model.vector_size
    weighted_vecs = []
    weight_sum = 0.0
    
    for token in tokens:
        if token not in model.wv:
            continue # skip characters the model doesn't know
        p_w = freq.get(token, 1) / total # how common is this character (0 to 1)
        w = a / (a + p_w) #so rare chars get higher weight than common ones
        weighted_vecs.append(w * model.wv[token]) # scale the vector by its weight
        weight_sum += w # we keep track of total weight for normalisation
    
    if weighted_vecs: # if it wasn't empty
        return (np.sum(weighted_vecs, axis=0) / weight_sum).astype(np.float32)  # weighted average
    return np.zeros(dim, dtype=np.float32) # fallback: zero vector

def remove_first_pc(embeddings: np.ndarray) -> np.ndarray:
    """
    Subtract each sentence vector's projection onto the first principal component of the embedding matrix (Step 2 of SIF).
    """
    # SVD decomposes the matrix; vt[0] is the direction of greatest variance
    _, _, vt = np.linalg.svd(embeddings, full_matrices=False)
    first_pc = vt[0] #the dominant direction shared by all sentences

    # For each sentence, compute how much of it points in the first_pc direction,
    # then subtract that component
    proj = (embeddings @ first_pc)[:, np.newaxis] * first_pc[np.newaxis, :]
    return (embeddings - proj).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Produce sentence embeddings (mean or SIF) for TSV files.")
    
    parser.add_argument("--model", required=True, help="Path to a saved Gensim FastText model (from train_fasttext.py).")
    
    parser.add_argument("--input_files", nargs="+", required=True, help="One or more .tsv files to convert.")
    
    parser.add_argument("--output_dir", default=None,
        help="Directory where .npz files are written. "
             "Defaults to the same directory as each input file.")
    
    # flag: activate SIF mode
    parser.add_argument("--sif", action="store_true", help="Use SIF weighted embeddings with first-PC removal instead of plain mean.")
    
    # smoothing constant for SIF weights
    parser.add_argument("--sif_a", type=float, default=1e-3, help="SIF smoothing parameter a (default: 1e-3).")

    args = parser.parse_args()

    print(f"Loading FastText model from {args.model} ...")
    
    model = FastText.load(args.model) # load the previously trained FastText model
    print(f"Model loaded. Vector size: {model.vector_size}")
    
    # to show the mode
    print(f"Embedding mode: {'SIF (a={})'.format(args.sif_a) if args.sif else 'mean'}")


    # --- SIF pre-pass: count character frequencies over all files 
    if args.sif:
        print("Building character frequency table over all input files...")
        freq, total = build_freq_table(args.input_files) 
        print(f"  Vocabulary size: {len(freq)}  Total characters: {total}")

    # --- Compute sentence embeddings for every file ---
    
    all_embeddings: list[np.ndarray] = [] # sentence vectors, one list per file
    all_labels: list[np.ndarray] = [] # topic labels, one list per file
    all_ids: list[np.ndarray] = [] #row IDs, one list per file
    file_sizes: list[int] = [] # how many sentences each file has

    for tsv_path in args.input_files:
        df = pd.read_csv(tsv_path, sep="\t") 
        embs = []
        
        for sentence in df["text"]:
            tokens = tokenize(str(sentence))
            if args.sif:
                vec = sif_vector(model, tokens, freq, total, args.sif_a) # SIF weighted average
            else:
                vec = mean_vector(model, tokens) #Simple unweighted average
            embs.append(vec)

        embs_arr = np.array(embs, dtype=np.float32) # convert list of vectors to a 2D array
        all_embeddings.append(embs_arr)
        all_labels.append(df["category"].to_numpy()) 
        all_ids.append(df["index_id"].to_numpy(dtype= np.int64))  
        file_sizes.append(len(embs_arr))               

    # SIF Step 2: remove first principal component globally
    if args.sif:
        print("Removing first principal component (global)...")
        combined = np.vstack(all_embeddings) # stack all files into one big matrix
        combined = remove_first_pc(combined) # subtract the dominant shared direction
        
        # Split the big matrix back into separate arrays, one per file
        splits = np.cumsum([0] + file_sizes) #These are the boundary indices 
        all_embeddings = [combined[splits[i]:splits[i + 1]] for i in range(len(file_sizes))]

    # --- Save .npz files 
    for i, tsv_path in enumerate(args.input_files):
        basename = os.path.splitext(os.path.basename(tsv_path))[0] # e.g. "train" in "data/train.tsv"
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True) # create output folder if needed
            output_path = os.path.join(args.output_dir, basename)
        else:
            output_path = os.path.join(os.path.dirname(tsv_path), basename)

        # Save embeddings + labels + IDs together in one compressed numpy file
        np.savez(output_path, embeddings=all_embeddings[i], labels=all_labels[i], ids=all_ids[i])
        
        print(f"  {tsv_path} -> {output_path}.npz  "
              f"({file_sizes[i]} sentences, dim={all_embeddings[i].shape[1]})")

    print("Done.")


if __name__ == "__main__":
    main()
