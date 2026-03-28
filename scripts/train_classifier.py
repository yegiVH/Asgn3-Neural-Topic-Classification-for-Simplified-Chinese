"""
train_classifier.py : Train a feed-forward PyTorch classifier on sentence embeddings.

---- Architecture
Input (dim)  ->  Linear -> ReLU -> Dropout
             ->  Linear -> ReLU -> Dropout
             ->  Linear (num_classes)
             ->  CrossEntropyLoss (includes softmax internally)


-----Usage example
    python scripts/train_classifier.py \
        --train_embeddings embeddings/train.npz \
        --dev_embeddings   embeddings/dev.npz   \
        --epochs 20 \
        --batch_size 32 \
        --hidden_size 128 \
        --dropout 0.3 \
        --lr 1e-3 \
        --output_model models/classifier.pt \
        --output_labels models/label_map.json \
        --plot models/training_curve.png

-----Required arguments
    --train_embeddings : .npz file produced by sentence_embeddings.py (training split)
    --dev_embeddings : .npz file for validation (dev split)
    --output_model : Where to save the trained model weights (.pt)
    --output_labels : Where to save the label->index mapping (.json)

-----Optional arguments
    --epochs : Training epochs (default: 20)
    --batch_size : Mini-batch size (default: 32)
    --hidden_size : Units per hidden layer (default: 128)
    --dropout : Dropout probability (default: 0.3)
    --lr : Learning rate (default: 1e-3)
    --plot : Path to save the training curve .png (skipped if not provided)
"""

import argparse         
import json             
import os          
import matplotlib
matplotlib.use("Agg") # use non-interactive backend so plots save without a display
import matplotlib.pyplot as plt
import numpy as np    
import torch 
import torch.nn as nn # neural network building blocks
from torch.utils.data import DataLoader, TensorDataset  # batching utilities


# Model
class FeedForwardClassifier(nn.Module):
    """Two hidden layers with ReLU activations and dropout."""

    def __init__(self, input_dim: int, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        # Build the network as a sequential chain of layers
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_size),# first hidden layer: embeddings -> hidden units
            nn.ReLU(), # non-linearity: replace negative values with 0
            nn.Dropout(dropout),# randomly zero some outputs to prevent overfitting
            
            # Layer 2
            nn.Linear(hidden_size, hidden_size), # second hidden layer: hidden -> hidden
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output Layer
            nn.Linear(hidden_size, num_classes), # output layer: hidden -> one score per class
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)# pass input through all layers in order



# Helper functions
# ------------------------------------
def load_npz(path: str):
    """Return (embeddings, labels) from a .npz file."""
    data = np.load(path, allow_pickle=True) # load the compressed numpy file
    return data["embeddings"].astype(np.float32), data["labels"]


def build_label_map(labels: np.ndarray) -> dict[str, int]:
    """Map unique label strings to integer indices (sorted for reproducibility)."""
    unique = sorted(set(labels.tolist())) # get all distinct topic names, sorted
    return {label: idx for idx, label in enumerate(unique)}  # {"entertainment": 0, ...}


def encode_labels(labels: np.ndarray, label_map: dict[str, int]) -> np.ndarray:
    # Convert string labels like "sports" to integers like 5
    return np.array([label_map[l] for l in labels], dtype=np.int64)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1) # pick the class with the highest score
    return (preds == targets).float().mean().item() # fraction of correct predictions


# -----------------------------------------------------------------------
# Training loop
def train(args):
    # Load sentence embeddings and their labels from the .npz files
    train_X, train_y_str = load_npz(args.train_embeddings)
    dev_X, dev_y_str = load_npz(args.dev_embeddings)

    label_map = build_label_map(train_y_str)
    train_y = encode_labels(train_y_str, label_map) # convert training labels to integers
    dev_y = encode_labels(dev_y_str, label_map) 

    print(f"Classes ({len(label_map)}): {list(label_map.keys())}")
    print(f"Train: {len(train_X)} samples   Dev: {len(dev_X)} samples   Dim: {train_X.shape[1]}")

    # wrapping numpy arrays into PyTorch datasets so we can iterate over batches
    train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    dev_dataset = TensorDataset(torch.from_numpy(dev_X), torch.from_numpy(dev_y))

    # DataLoaders shuffle training data each epoch and cut it into mini-batches
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Creating the neural network and moving it to the chosen device
    model = FeedForwardClassifier( # the class I defined up there
        input_dim = train_X.shape[1],# number of input features = embedding dimension
        hidden_size = args.hidden_size,
        num_classes = len(label_map),   
        dropout = args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss() #loss function for multi-class classification
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)# Adam adjusts weights automatically

    best_dev_acc = 0.0 # for tracking the best accuracy seen so far on the dev set
    history_loss: list[float] = [] # store loss per epoch for the plot
    history_dev_acc: list[float] = [] # store dev accuracy per epoch for the plot

    for epoch in range(1, args.epochs + 1):
        # --- Training phase
        model.train() # enable dropout (active during training)
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # move batch to GPU/CPU
            optimiser.zero_grad() # clear gradients from the previous batch
            logits = model(X_batch) # forward pass: compute class scores
            loss = criterion(logits, y_batch) #measure how wrong the predictions are
            loss.backward() # backprop: compute gradients
            optimiser.step() # update model weights using gradients
            total_loss += loss.item() * len(X_batch) # accumulate loss weighted by batch size

        #average loss over all training examples
        avg_loss = total_loss / len(train_dataset)

        # --- Validation phase
        model.eval() # disable dropout (no randomness during evaluation)
        all_logits, all_targets = [], []
        with torch.no_grad():# skip gradient computation to save memory and speed up
            for X_batch, y_batch in dev_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                all_logits.append(model(X_batch)) # collect predictions for each batch
                all_targets.append(y_batch)

        dev_logits = torch.cat(all_logits) # combine all batches into one tensor
        dev_targets = torch.cat(all_targets)
        dev_acc = accuracy(dev_logits, dev_targets) # compute accuracy on the dev set

        history_loss.append(avg_loss) # record for plotting
        history_dev_acc.append(dev_acc)
        print(f"Epoch {epoch:3d}/{args.epochs} loss={avg_loss:.4f} dev_acc={dev_acc:.4f}")

        # Save the model only when it achieves a new best dev accuracy
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.output_model)  # save only the weights, not the whole object

    print(f"\nBest dev accuracy: {best_dev_acc:.4f}")
    print(f"Model saved: {args.output_model}")

    # Saving label map so the evaluator can convert predicted integers back to topic names
    os.makedirs(os.path.dirname(args.output_labels) or ".", exist_ok=True)
    with open(args.output_labels, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Labels saved: {args.output_labels}")

    #training curve plot
    if args.plot:
        epochs = range(1, args.epochs + 1)
        chance = 1.0 / len(label_map)# random-guess baseline (1 divided by number of classes)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_loss = "#d62728"# red for loss
        color_acc  = "#1f77b4" # blue for accuracy

        # Left y-axis: training loss
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-entropy loss", color=color_loss)
        ax1.plot(epochs, history_loss, color=color_loss, linewidth=2, label="Train loss")
        ax1.tick_params(axis="y", labelcolor=color_loss)

        # Right y-axis: dev accuracy 
        ax2 = ax1.twinx()
        ax2.set_ylabel("Dev accuracy", color=color_acc)
        ax2.plot(epochs, history_dev_acc, color=color_acc, linewidth=2, label="Dev accuracy")
        ax2.axhline(chance, color=color_acc, linestyle="--", linewidth=1,
                    label=f"Chance ({chance:.2f})") # horizontal dashed line at chance level
        ax2.set_ylim(0, 1) # accuracy always between 0 and 1
        ax2.tick_params(axis="y", labelcolor=color_acc)

        # Combine legends from both axes into one box
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        plt.title("Training curve — loss and dev accuracy per epoch")
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig.savefig(args.plot, dpi=150) 
        plt.close(fig)# free memory
        print(f"Plot saved:   {args.plot}")


# main --------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a feed-forward classifier on FastText sentence embeddings.")
    
    parser.add_argument("--train_embeddings", required=True, help="Path to train .npz file (from sentence_embeddings.py).")
    
    parser.add_argument("--dev_embeddings", required=True, help="Path to dev .npz file.")
    
    parser.add_argument("--output_model", required=True, help="Where to save the best model weights (.pt).")
    
    parser.add_argument("--output_labels", required=True, help="Where to save the label->index JSON mapping.")
    
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20).")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32).")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size (default: 128).")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability (default: 0.3).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--plot", default=None, help="Optional path to save the training curve as a .png file.")

    args = parser.parse_args()
    train(args) # hand off to the training function


if __name__ == "__main__":
    main()
