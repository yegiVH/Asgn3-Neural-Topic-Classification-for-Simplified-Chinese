"""
train_fasttext.py: Train FastText word embeddings over one or more .tsv files.

---- Tokenisation strategy
Chinese text is segmented at the character level (every Unicode code-point becomes one token). 
Latin/ASCII characters that appear in the corpus are treated the same way (each character is its own token). 

----- Usage example
    python scripts/train_fasttext.py \
        --input_files data/train.tsv data/dev.tsv data/test.tsv \
        --dim 100 \
        --output_model models/fasttext.model

----- Optional flags
    --epochs : Number of training epochs (default 5)
    --window : Context window size (default 5)
    --sg : Use skip-gram instead of CBOW (flag, default CBOW)
    --workers : Number of worker threads (default 4)
"""

import argparse          
import pandas as pd      
from gensim.models import FastText 


def tokenize(sentence: str) -> list[str]:
    """Split a sentence into individual characters (works for Chinese & Latin)."""
    return list(sentence)


def main():
    # --- Setting up command-line argument parsing 
    parser = argparse.ArgumentParser(description="Train FastText character-level embeddings over TSV corpora.")
    
    # one or more file paths
    parser.add_argument("--input_files", nargs="+", required=True, help="One or more .tsv files whose 'text' column is used for training.")
    
    #size of each embedding vector
    parser.add_argument("--dim", type=int, default=100, help="Embedding dimensionality (default: 100).")
    
    # where to save the trained model
    parser.add_argument("--output_model", required=True, help="Path where the trained Gensim FastText model will be saved.")
    
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    
    # how many characters left/right to consider as context
    parser.add_argument("--window", type=int, default=5, help="Context window size (default: 5).")
    
    # flag: if set, use skip-gram instead of CBOW
    parser.add_argument("--sg", action="store_true", help="Use skip-gram training (default: CBOW).")
    
    # parallel threads to speed up training
    parser.add_argument("--workers", type=int, default=4, 
        help="Number of worker threads (default: 4).")

    args = parser.parse_args()  

    sentences = []  # will hold all tokenised sentences from all input files

    for file in args.input_files:
        df = pd.read_csv(file, sep="\t") # read the TSV file into a dataframe
        for s in df["text"]: # for each sentence in the column text
            tokens = tokenize(str(s)) # split into characters
            sentences.append(tokens) # adding the character list to our collection

    print(f"Training FastText on {len(sentences)} sentences "
          f"(dim={args.dim}, epochs={args.epochs}, sg={args.sg})...")

    # Train the FastText model on all sentences
    model = FastText(
        sentences = sentences, # the list of tokenised sentences
        vector_size = args.dim,       
        window = args.window,         
        min_count = 1, # include every character, even if it appears only once
        epochs = args.epochs, 
        sg = int(args.sg), # 1 = skip-gram, 0 = CBOW
        workers = args.workers, 
    )
    
    # write the trained model to disk
    model.save(args.output_model)  
    print(f"Model saved: {args.output_model}")


if __name__ == "__main__":
    main()
