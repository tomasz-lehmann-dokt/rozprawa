# BIQA - Bezwzorcowa ocena jakości obrazu

Predyktor MOS oparty na architekturach ConvNeXt-MLP oraz dual-Xception. Szczegóły w rozdziale 3 rozprawy.

## Struktura kodu

| Plik | Opis |
|------|------|
| `model.py` | ConvNeXt-MLP: backbone ConvNeXt + MLP dla parametrów |
| `model_xception.py` | dual-Xception: backbone Xception + MLP dla parametrów |
| `dataset.py` | Loader KonIQ-10k z podziałem train/valid/test |
| `features.py` | Ekstrakcja 9 parametrów obrazu |
| `metrics.py` | SROCC, PLCC, RMSE |
| `train.py` | Trening z AdamW, CosineAnnealingLR, mixed precision |
| `evaluate.py` | Ewaluacja na zbiorze testowym |

## Dane

**KonIQ-10k**: http://database.mmsp-kn.de/koniq-10k-database.html

Struktura po pobraniu:

```
data/
├── 1024x768/                              # obrazy JPEG
└── koniq10k_scores_and_distributions.csv  # oceny MOS
```

## Trening

```bash
cd src

# ConvNeXt-MLP
python train.py --csv_path /path/to/koniq10k_scores_and_distributions.csv \
                --images_dir /path/to/1024x768 \
                --model convnext \
                --epochs 60 \
                --batch_size 8

# dual-Xception
python train.py --csv_path /path/to/koniq10k_scores_and_distributions.csv \
                --images_dir /path/to/1024x768 \
                --model xception \
                --epochs 60 \
                --batch_size 4
```

Checkpoint zostanie zapisany w `checkpoints/`.

## Ewaluacja

```bash
cd src

python evaluate.py --checkpoint ../models/convnext_mlp_koniq.pth \
                   --csv_path /path/to/koniq10k_scores_and_distributions.csv \
                   --images_dir /path/to/1024x768 \
                   --model convnext
```

## Modele

| Plik | Opis |
|------|------|
| `convnext_mlp_koniq.pth` | ConvNeXt-MLP wytrenowany na KonIQ-10k |
| `param_stats.npz` | Statystyki normalizacji parametrów |

## Wyniki

| Model | SROCC | PLCC | RMSE |
|-------|-------|------|------|
| ConvNeXt-MLP | 0.921 | 0.932 | 5.02 |
