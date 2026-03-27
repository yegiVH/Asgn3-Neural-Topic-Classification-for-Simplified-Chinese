# Session Transcript

Full terminal session run on **mltgpu** from the project root.

```
gusvahaye@GU.GU.SE@mltgpu:~$ ls
as3  Assignment1  nltk_data
gusvahaye@GU.GU.SE@mltgpu:~$ git clone https://github.com/yegiVH/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese.git
Cloning into 'Asgn3-Neural-Topic-Classification-for-Simplified-Chinese'...
remote: Enumerating objects: 36, done.
remote: Counting objects: 100% (36/36), done.
remote: Compressing objects: 100% (30/30), done.
remote: Total 36 (delta 6), reused 36 (delta 6), pack-reused 0 (from 0)
Receiving objects: 100% (36/36), 886.32 KiB | 9.63 MiB/s, done.
Resolving deltas: 100% (6/6), done.

gusvahaye@GU.GU.SE@mltgpu:~$ cd Asgn3-Neural-Topic-Classification-for-Simplified-Chinese/

gusvahaye@GU.GU.SE@mltgpu:~/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese$ mkdir -p models embeddings models_sif embeddings_sif
```

---

## Step 1 — Train FastText embeddings

```
gusvahaye@GU.GU.SE@mltgpu:~/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese$ python scripts/train_fasttext.py \
        --input_files data/train.tsv data/dev.tsv data/test.tsv \
        --dim 100 \
        --output_model models/fasttext.model
Training FastText on 1004 sentences (dim=100, epochs=5, sg=False)...
Model saved: models/fasttext.model
```

---

## Step 2 — Compute sentence embeddings (mean)

```
gusvahaye@GU.GU.SE@mltgpu:~/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese$ python scripts/sentence_embeddings.py \
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
gusvahaye@GU.GU.SE@mltgpu:~/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese$ python scripts/train_classifier.py \
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
Classes (7): ['entertainment', 'geography', 'health', 'politics', 'science/technology', 'sports', 'travel']
Train: 701 samples   Dev: 99 samples   Dim: 100
Device: cuda
Epoch   1/20 loss=1.9160 dev_acc=0.2525
Epoch   2/20 loss=1.8755 dev_acc=0.2525
Epoch   3/20 loss=1.8698 dev_acc=0.2525
Epoch   4/20 loss=1.8555 dev_acc=0.2525
Epoch   5/20 loss=1.8506 dev_acc=0.2323
Epoch   6/20 loss=1.8394 dev_acc=0.2323
Epoch   7/20 loss=1.8376 dev_acc=0.2323
Epoch   8/20 loss=1.8307 dev_acc=0.2323
Epoch   9/20 loss=1.8263 dev_acc=0.2222
Epoch  10/20 loss=1.8196 dev_acc=0.2424
Epoch  11/20 loss=1.8240 dev_acc=0.2323
Epoch  12/20 loss=1.8327 dev_acc=0.2222
Epoch  13/20 loss=1.8314 dev_acc=0.2222
Epoch  14/20 loss=1.8201 dev_acc=0.2424
Epoch  15/20 loss=1.8198 dev_acc=0.2222
Epoch  16/20 loss=1.8235 dev_acc=0.2323
Epoch  17/20 loss=1.8145 dev_acc=0.2424
Epoch  18/20 loss=1.8234 dev_acc=0.2424
Epoch  19/20 loss=1.8141 dev_acc=0.2424
Epoch  20/20 loss=1.8099 dev_acc=0.2525

Best dev accuracy: 0.2525
Model saved: models/classifier.pt
Labels saved: models/label_map.json
Plot saved:   models/training_curve.png
```

---

## Step 4 — Evaluate (mean embeddings)

```
gusvahaye@GU.GU.SE@mltgpu:~/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese$ python scripts/evaluate.py \
        --model      models/classifier.pt \
        --label_map  models/label_map.json \
        --embeddings embeddings/train.npz embeddings/dev.npz embeddings/test.npz
/home/gusvahaye@GU.GU.SE/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese/scripts/evaluate.py:137: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(args.model, map_location=device))  # load saved weights
Loaded model from models/classifier.pt  (input_dim=100, classes=7)

============================================================
Split: embeddings/train.npz
============================================================
Accuracy : 176/701 = 0.2511 (25.1%)
Chance   : 0.1429 (14.3%)  [7 classes]
Above chance: YES

Confusion matrix (rows = true, columns = predicted):
                    entertainmentgeographyhealthpoliticsscience/technologysportstravel
--------------------------------------------------------------------------------------
entertainment           0    0    0    0   65    0    0
geography               0    0    0    0   58    0    0
health                  0    0    0    0   77    0    0
politics                0    0    0    0  102    0    0
science/technology      0    0    0    0  176    0    0
sports                  0    0    0    0   85    0    0
travel                  0    0    0    0  138    0    0

============================================================
Split: embeddings/dev.npz
============================================================
Accuracy : 25/99 = 0.2525 (25.3%)
Chance   : 0.1429 (14.3%)  [7 classes]
Above chance: YES

Confusion matrix (rows = true, columns = predicted):
                    entertainmentgeographyhealthpoliticsscience/technologysportstravel
--------------------------------------------------------------------------------------
entertainment           0    0    0    0    9    0    0
geography               0    0    0    0    8    0    0
health                  0    0    0    0   11    0    0
politics                0    0    0    0   14    0    0
science/technology      0    0    0    0   25    0    0
sports                  0    0    0    0   12    0    0
travel                  0    0    0    0   20    0    0

============================================================
Split: embeddings/test.npz
============================================================
Accuracy : 51/204 = 0.2500 (25.0%)
Chance   : 0.1429 (14.3%)  [7 classes]
Above chance: YES

Confusion matrix (rows = true, columns = predicted):
                    entertainmentgeographyhealthpoliticsscience/technologysportstravel
--------------------------------------------------------------------------------------
entertainment           0    0    0    0   19    0    0
geography               0    0    0    0   17    0    0
health                  0    0    0    0   22    0    0
politics                0    0    0    0   30    0    0
science/technology      0    0    0    0   51    0    0
sports                  0    0    0    0   25    0    0
travel                  0    0    0    0   40    0    0
```

---

## Bonus — SIF embeddings

### Step 2 (SIF) — Compute sentence embeddings

```
gusvahaye@GU.GU.SE@mltgpu:~/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese$ python scripts/sentence_embeddings.py \
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
gusvahaye@GU.GU.SE@mltgpu:~/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese$ python scripts/train_classifier.py \
    --train_embeddings embeddings_sif/train.npz \
    --dev_embeddings   embeddings_sif/dev.npz   \
    --output_model     models_sif/classifier.pt \
    --output_labels    models_sif/label_map.json \
    --epochs 20 \
    --batch_size 32 \
    --plot models_sif/training_curve_sif.png
Classes (7): ['entertainment', 'geography', 'health', 'politics', 'science/technology', 'sports', 'travel']
Train: 701 samples   Dev: 99 samples   Dim: 100
Device: cuda
Epoch   1/20 loss=1.9360 dev_acc=0.2525
Epoch   2/20 loss=1.8938 dev_acc=0.2525
Epoch   3/20 loss=1.8544 dev_acc=0.2525
Epoch   4/20 loss=1.8426 dev_acc=0.2323
Epoch   5/20 loss=1.8334 dev_acc=0.2222
Epoch   6/20 loss=1.8195 dev_acc=0.2323
Epoch   7/20 loss=1.8184 dev_acc=0.2323
Epoch   8/20 loss=1.8148 dev_acc=0.2323
Epoch   9/20 loss=1.8044 dev_acc=0.2424
Epoch  10/20 loss=1.8036 dev_acc=0.2424
Epoch  11/20 loss=1.7984 dev_acc=0.2525
Epoch  12/20 loss=1.8006 dev_acc=0.2525
Epoch  13/20 loss=1.8040 dev_acc=0.2525
Epoch  14/20 loss=1.8016 dev_acc=0.2525
Epoch  15/20 loss=1.7959 dev_acc=0.2525
Epoch  16/20 loss=1.7955 dev_acc=0.2525
Epoch  17/20 loss=1.7909 dev_acc=0.2626
Epoch  18/20 loss=1.7932 dev_acc=0.2929
Epoch  19/20 loss=1.7899 dev_acc=0.2929
Epoch  20/20 loss=1.7831 dev_acc=0.2929

Best dev accuracy: 0.2929
Model saved: models_sif/classifier.pt
Labels saved: models_sif/label_map.json
Plot saved:   models_sif/training_curve_sif.png
```

### Step 4 (SIF) — Evaluate

```
gusvahaye@GU.GU.SE@mltgpu:~/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese$ python scripts/evaluate.py \
    --model     models_sif/classifier.pt \
    --label_map models_sif/label_map.json \
    --embeddings embeddings_sif/train.npz embeddings_sif/dev.npz embeddings_sif/test.npz
/home/gusvahaye@GU.GU.SE/Asgn3-Neural-Topic-Classification-for-Simplified-Chinese/scripts/evaluate.py:137: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(args.model, map_location=device))  # load saved weights
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
geography               0    0    0   28   30    0    0
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
Accuracy : 54/204 = 0.2647 (26.5%)
Chance   : 0.1429 (14.3%)  [7 classes]
Above chance: YES

Confusion matrix (rows = true, columns = predicted):
                    entertainmentgeographyhealthpoliticsscience/technologysportstravel
--------------------------------------------------------------------------------------
entertainment           0    0    0    3   16    0    0
geography               0    0    0   10    7    0    0
health                  0    0    0    5   17    0    0
politics                0    0    0   13   17    0    0
science/technology      0    0    0   10   41    0    0
sports                  0    0    0   12   13    0    0
travel                  0    0    0    8   32    0    0
```
