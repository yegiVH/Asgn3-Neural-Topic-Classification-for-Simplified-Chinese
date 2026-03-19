"""
evaluate.py — Load a saved classifier and print accuracy + confusion matrix.

Evaluates on one or more .npz embedding files (produced by sentence_embeddings.py).
Results are printed to the terminal.

Usage example
-------------
    python scripts/evaluate.py \
        --model      models/classifier.pt \
        --label_map  models/label_map.json \
        --embeddings embeddings/train.npz embeddings/dev.npz embeddings/test.npz
"""

import argparse   # command-line argument parsing
import json       # reading the label->index mapping saved during training

import numpy as np   # numerical arrays
import torch         # PyTorch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definition (must match train_classifier.py exactly)
# ---------------------------------------------------------------------------

class FeedForwardClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        # Identical architecture to train_classifier.py so weights can be loaded correctly
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),   # first hidden layer
            nn.ReLU(),                           # non-linearity
            nn.Dropout(dropout),                 # dropout for regularisation
            nn.Linear(hidden_size, hidden_size), # second hidden layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes), # output: one score per class
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(path: str):
    data = np.load(path, allow_pickle=True)              # load compressed numpy file
    return data["embeddings"].astype(np.float32), data["labels"]


def print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    """Pretty-print a confusion matrix with row/column headers."""
    col_w = max(len(l) for l in labels) + 2             # width of the label column
    num_w = max(4, len(str(cm.max()))) + 1               # width of each number cell

    # Print column header row
    header = " " * col_w + "".join(f"{l:>{num_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    # Print each row: true label on the left, then prediction counts
    for i, row_label in enumerate(labels):
        row = f"{row_label:<{col_w}}" + "".join(f"{cm[i, j]:>{num_w}}" for j in range(len(labels)))
        print(row)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)    # start with an n x n matrix of zeros
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1                   # increment the cell at (true label, predicted label)
    return cm


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, device, X: np.ndarray, y_int: np.ndarray,
             label_names: list[str], split_name: str) -> None:
    model.eval()                        # disable dropout for deterministic predictions
    with torch.no_grad():               # no gradient needed during evaluation
        logits = model(torch.from_numpy(X).to(device))  # forward pass for all sentences at once
    preds = logits.argmax(dim=1).cpu().numpy()  # pick the class with the highest score for each sentence

    correct = (preds == y_int).sum()    # count how many predictions match the true labels
    total   = len(y_int)
    acc     = correct / total           # accuracy = correct / total
    chance  = 1.0 / len(label_names)   # random-guess baseline

    print(f"\n{'='*60}")
    print(f"Split: {split_name}")
    print(f"{'='*60}")
    print(f"Accuracy : {correct}/{total} = {acc:.4f} ({acc*100:.1f}%)")
    print(f"Chance   : {chance:.4f} ({chance*100:.1f}%)  [{len(label_names)} classes]")
    print(f"Above chance: {'YES' if acc > chance else 'NO'}")

    cm = confusion_matrix(y_int, preds, len(label_names))  # build the confusion matrix
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print_confusion_matrix(cm, label_names)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved feed-forward classifier on sentence embedding files."
    )
    parser.add_argument("--model",      required=True, help="Path to saved model weights (.pt).")
    parser.add_argument("--label_map",  required=True, help="Path to label->index JSON (from train_classifier.py).")
    parser.add_argument("--embeddings", nargs="+", required=True,
                        help="One or more .npz files to evaluate on.")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size used during training (default: 128).")
    parser.add_argument("--dropout",     type=float, default=0.0,
                        help="Dropout during eval is 0 by default (model.eval() handles this).")
    args = parser.parse_args()

    # Load the label map: maps topic names to integers, e.g. {"sports": 5, ...}
    with open(args.label_map, encoding="utf-8") as f:
        label_map: dict[str, int] = json.load(f)
    # Reverse the map to get a list of label names in the correct order: [label_0, label_1, ...]
    idx_to_label = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes  = len(label_map)

    # Read the first embedding file just to find out the vector dimension
    first_X, _ = load_npz(args.embeddings[0])
    input_dim   = first_X.shape[1]     # e.g. 100 if trained with --dim 100

    # Recreate the model architecture and load the saved weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FeedForwardClassifier(input_dim, args.hidden_size, num_classes, dropout=0.0).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))  # load saved weights
    model.eval()   # switch to evaluation mode (disables dropout)
    print(f"Loaded model from {args.model}  (input_dim={input_dim}, classes={num_classes})")

    # Evaluate on each provided embedding file (train, dev, test, etc.)
    for path in args.embeddings:
        X, y_str = load_npz(path)
        y_int    = np.array([label_map[l] for l in y_str], dtype=np.int64)  # convert labels to ints
        split    = path   # use the file path as a human-readable split name
        evaluate(model, device, X, y_int, idx_to_label, split)

    print()


if __name__ == "__main__":
    main()
