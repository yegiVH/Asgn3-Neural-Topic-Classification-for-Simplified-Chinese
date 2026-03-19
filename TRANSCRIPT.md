# Session Transcript

Full terminal session run on **mltgpu** from the project root.

```
[user@mltgpu ~]$ cd Asgn3-Neural-Topic-Classification-for-Simplified-Chinese
[user@mltgpu Asgn3-Neural-Topic-Classification-for-Simplified-Chinese]$ mkdir -p models embeddings models_sif embeddings_sif
```

---

## Step 1 — Train FastText embeddings

```
[user@mltgpu Asgn3-Neural-Topic-Classification-for-Simplified-Chinese]$ python scripts/train_fasttext.py \
    --input_files data/train.tsv data/dev.tsv data/test.tsv \
    --dim 100 \
    --output_model models/fasttext.model

Training FastText on 1004 sentences (dim=100, epochs=5, sg=False)...
Model saved: models/fasttext.model
```

---

## Step 2 — Compute sentence embeddings (mean)

```
[user@mltgpu Asgn3-Neural-Topic-Classification-for-Simplified-Chinese]$ python scripts/sentence_embeddings.py \
    --model models/fasttext.model \
    --input_files data/train.tsv data/dev.tsv data/test.tsv \
    --output_dir embeddings/

Loading FastText model from models/fasttext.model ...
Model loaded. Vector size: 100
Embedding mode: mean
  data/train.tsv -> embeddings/train.npz  (701 sentences, dim=100)
  data/dev.tsv -> embeddings/dev.npz  (99 sentences, dim=100)
  data/test.tsv -> embeddings/test.npz  (204 sentences, dim=100)
Done.
```

---

## Step 3 — Train classifier (mean embeddings)

```
[user@mltgpu Asgn3-Neural-Topic-Classification-for-Simplified-Chinese]$ python scripts/train_classifier.py \
    --train_embeddings embeddings/train.npz \
    --dev_embeddings   embeddings/dev.npz   \
    --output_model     models/classifier.pt \
    --output_labels    models/label_map.json \
    --epochs 20 \
    --batch_size 32 \
    --plot models/training_curve.png

Classes (7): ['entertainment', 'geography', 'health', 'politics', 'science/technology', 'sports', 'travel']
Train: 701 samples   Dev: 99 samples   Dim: 100
Device: cpu
Epoch   1/20  loss=1.9179  dev_acc=0.2020
Epoch   2/20  loss=1.8561  dev_acc=0.2121
Epoch   3/20  loss=1.7934  dev_acc=0.2323
Epoch   4/20  loss=1.7312  dev_acc=0.2424
Epoch   5/20  loss=1.6748  dev_acc=0.2424
Epoch   6/20  loss=1.6201  dev_acc=0.2525
Epoch   7/20  loss=1.5689  dev_acc=0.2525
Epoch   8/20  loss=1.5234  dev_acc=0.2525
Epoch   9/20  loss=1.4823  dev_acc=0.2626
Epoch  10/20  loss=1.4451  dev_acc=0.2626
Epoch  11/20  loss=1.4102  dev_acc=0.2626
Epoch  12/20  loss=1.3798  dev_acc=0.2626
Epoch  13/20  loss=1.3524  dev_acc=0.2626
Epoch  14/20  loss=1.3273  dev_acc=0.2626
Epoch  15/20  loss=1.3040  dev_acc=0.2626
Epoch  16/20  loss=1.2839  dev_acc=0.2626
Epoch  17/20  loss=1.2651  dev_acc=0.2626
Epoch  18/20  loss=1.2487  dev_acc=0.2626
Epoch  19/20  loss=1.2338  dev_acc=0.2626
Epoch  20/20  loss=1.2201  dev_acc=0.2626

Best dev accuracy: 0.2626
Model saved:  models/classifier.pt
Labels saved: models/label_map.json
Plot saved:   models/training_curve.png
```

---

## Step 4 — Evaluate (mean embeddings)

```
[user@mltgpu Asgn3-Neural-Topic-Classification-for-Simplified-Chinese]$ python scripts/evaluate.py \
    --model     models/classifier.pt \
    --label_map models/label_map.json \
    --embeddings embeddings/train.npz embeddings/dev.npz embeddings/test.npz

Loaded model from models/classifier.pt  (input_dim=100, classes=7)

============================================================
Split: embeddings/train.npz
============================================================
Accuracy : 190/701 = 0.2710 (27.1%)
Chance   : 0.1429 (14.3%)  [7 classes]
Above chance: YES

Confusion matrix (rows = true, columns = predicted):
                    entertainmentgeographyhealthpoliticsscience/technologysportstravel
--------------------------------------------------------------------------------------
entertainment           0    0    0   24   41    0    0
geography               0    0    0   26   32    0    0
health                  0    0    0   22   55    0    0
politics                0    0    0   47   55    0    0
science/technology      0    0    0   33  143    0    0
sports                  0    0    0   40   45    0    0
travel                  0    0    0   17  121    0    0

============================================================
Split: embeddings/dev.npz
============================================================
Accuracy : 26/99 = 0.2626 (26.3%)
Chance   : 0.1429 (14.3%)  [7 classes]
Above chance: YES

Confusion matrix (rows = true, columns = predicted):
                    entertainmentgeographyhealthpoliticsscience/technologysportstravel
--------------------------------------------------------------------------------------
entertainment           0    0    0    2    7    0    0
geography               0    0    0    3    5    0    0
health                  0    0    0    2    9    0    0
politics                0    0    0    5    9    0    0
science/technology      0    0    0    4   21    0    0
sports                  0    0    0    6    6    0    0
travel                  0    0    0    5   15    0    0

============================================================
Split: embeddings/test.npz
============================================================
Accuracy : 51/204 = 0.2500 (25.0%)
Chance   : 0.1429 (14.3%)  [7 classes]
Above chance: YES

Confusion matrix (rows = true, columns = predicted):
                    entertainmentgeographyhealthpoliticsscience/technologysportstravel
--------------------------------------------------------------------------------------
entertainment           0    0    0    3   16    0    0
geography               0    0    0   10    7    0    0
health                  0    0    0    5   17    0    0
politics                0    0    0   10   20    0    0
science/technology      0    0    0   10   41    0    0
sports                  0    0    0   12   13    0    0
travel                  0    0    0    8   32    0    0
```

---

## Bonus — SIF embeddings

### Step 2 (SIF) — Compute sentence embeddings

```
[user@mltgpu Asgn3-Neural-Topic-Classification-for-Simplified-Chinese]$ python scripts/sentence_embeddings.py \
    --model models/fasttext.model \
    --input_files data/train.tsv data/dev.tsv data/test.tsv \
    --output_dir embeddings_sif/ \
    --sif \
    --sif_a 1e-3

Loading FastText model from models/fasttext.model ...
Model loaded. Vector size: 100
Embedding mode: SIF (a=0.001)
Building character frequency table over all input files...
  Vocabulary size: 2122  Total characters: 43522
Removing first principal component (global)...
  data/train.tsv -> embeddings_sif/train.npz  (701 sentences, dim=100)
  data/dev.tsv -> embeddings_sif/dev.npz  (99 sentences, dim=100)
  data/test.tsv -> embeddings_sif/test.npz  (204 sentences, dim=100)
Done.
```

### Step 3 (SIF) — Train classifier

```
[user@mltgpu Asgn3-Neural-Topic-Classification-for-Simplified-Chinese]$ python scripts/train_classifier.py \
    --train_embeddings embeddings_sif/train.npz \
    --dev_embeddings   embeddings_sif/dev.npz   \
    --output_model     models_sif/classifier.pt \
    --output_labels    models_sif/label_map.json \
    --epochs 20 \
    --batch_size 32 \
    --plot models_sif/training_curve_sif.png

Classes (7): ['entertainment', 'geography', 'health', 'politics', 'science/technology', 'sports', 'travel']
Train: 701 samples   Dev: 99 samples   Dim: 100
Device: cpu
Epoch   1/20  loss=1.9318  dev_acc=0.2525
Epoch   2/20  loss=1.8923  dev_acc=0.2525
Epoch   3/20  loss=1.8525  dev_acc=0.2525
Epoch   4/20  loss=1.8383  dev_acc=0.2323
Epoch   5/20  loss=1.8253  dev_acc=0.2323
Epoch   6/20  loss=1.8271  dev_acc=0.2222
Epoch   7/20  loss=1.8121  dev_acc=0.2323
Epoch   8/20  loss=1.8113  dev_acc=0.2424
Epoch   9/20  loss=1.8020  dev_acc=0.2424
Epoch  10/20  loss=1.8048  dev_acc=0.2525
Epoch  11/20  loss=1.8084  dev_acc=0.2525
Epoch  12/20  loss=1.7995  dev_acc=0.2525
Epoch  13/20  loss=1.7946  dev_acc=0.2525
Epoch  14/20  loss=1.7928  dev_acc=0.2525
Epoch  15/20  loss=1.7898  dev_acc=0.2525
Epoch  16/20  loss=1.7888  dev_acc=0.2626
Epoch  17/20  loss=1.8014  dev_acc=0.2929
Epoch  18/20  loss=1.7877  dev_acc=0.2727
Epoch  19/20  loss=1.7905  dev_acc=0.2929
Epoch  20/20  loss=1.7866  dev_acc=0.2929

Best dev accuracy: 0.2929
Model saved:  models_sif/classifier.pt
Labels saved: models_sif/label_map.json
Plot saved:   models_sif/training_curve_sif.png
```

### Step 4 (SIF) — Evaluate

```
[user@mltgpu Asgn3-Neural-Topic-Classification-for-Simplified-Chinese]$ python scripts/evaluate.py \
    --model     models_sif/classifier.pt \
    --label_map models_sif/label_map.json \
    --embeddings embeddings_sif/train.npz embeddings_sif/dev.npz embeddings_sif/test.npz

Loaded model from models_sif/classifier.pt  (input_dim=100, classes=7)

============================================================
Split: embeddings_sif/train.npz
============================================================
Accuracy : 192/701 = 0.2739 (27.4%)
Chance   : 0.1429 (14.3%)  [7 classes]
Above chance: YES

Confusion matrix (rows = true, columns = predicted):
                    entertainmentgeographyhealthpoliticsscience/technologysportstravel
--------------------------------------------------------------------------------------
entertainment           0    0    0   27   38    0    0
geography               0    0    0   27   31    0    0
health                  0    0    0   22   55    0    0
politics                0    0    0   51   51    0    0
science/technology      0    0    0   35  141    0    0
sports                  0    0    0   45   40    0    0
travel                  0    0    0   18  120    0    0

============================================================
Split: embeddings_sif/dev.npz
============================================================
Accuracy : 29/99 = 0.2929 (29.3%)
Chance   : 0.1429 (14.3%)  [7 classes]
Above chance: YES

Confusion matrix (rows = true, columns = predicted):
                    entertainmentgeographyhealthpoliticsscience/technologysportstravel
--------------------------------------------------------------------------------------
entertainment           0    0    0    2    7    0    0
geography               0    0    0    4    4    0    0
health                  0    0    0    4    7    0    0
politics                0    0    0    8    6    0    0
science/technology      0    0    0    4   21    0    0
sports                  0    0    0    7    5    0    0
travel                  0    0    0    5   15    0    0

============================================================
Split: embeddings_sif/test.npz
============================================================
Accuracy : 55/204 = 0.2696 (27.0%)
Chance   : 0.1429 (14.3%)  [7 classes]
Above chance: YES

Confusion matrix (rows = true, columns = predicted):
                    entertainmentgeographyhealthpoliticsscience/technologysportstravel
--------------------------------------------------------------------------------------
entertainment           0    0    0    3   16    0    0
geography               0    0    0   10    7    0    0
health                  0    0    0    5   17    0    0
politics                0    0    0   13   17    0    0
science/technology      0    0    0    9   42    0    0
sports                  0    0    0   12   13    0    0
travel                  0    0    0    8   32    0    0
```
