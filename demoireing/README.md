# Redukcja prążków mory

Attention U-Net do usuwania artefaktów mory. Szczegóły w rozdziale 4 rozprawy.

## Struktura kodu

| Plik | Opis |
|------|------|
| `model.py` | Attention U-Net z bramkami uwagi na skip connections |
| `dataset.py` | Loader UHDM/TIP2018 z RFBCS (Rotating Fixed-Budget Cross-Sampling) |
| `metrics.py` | SSIM, PSNR |
| `train.py` | Trening z Adam, StepLR, kombinowaną stratą MSE+SSIM |
| `evaluate.py` | Ewaluacja na zbiorach testowych |

## Dane

- **UHDM**: https://xinyu-andy.github.io/uhdm-page/
- **TIP2018**: https://github.com/ZhengJun-AI/MoireBenchmark

Struktura po pobraniu:

```
data/
├── UHDM/
│   ├── train/           # podkatalogi ze scenami, pliki *moire.jpg + *gt.jpg
│   └── test_origin/     # pliki *moire.jpg + *gt.jpg
└── TIP2018/
    ├── trainData/       # *source.png, *target.png
    └── testData/
```

## Trening

```bash
cd src

# Strategia RFBCS (0.1*MSE + (1-SSIM), 15500 próbek TIP2018/epokę)
python train.py --uhdm_dir /path/to/UHDM \
                --tip_dir /path/to/TIP2018 \
                --epochs 100 \
                --batch_size 32 \
                --image_size 512 \
                --mse_weight 0.1 \
                --ssim_weight 1.0 \
                --tip_budget 15500

# Pełny zbiór TIP2018 (bez RFBCS)
python train.py --uhdm_dir /path/to/UHDM \
                --tip_dir /path/to/TIP2018 \
                --epochs 100 \
                --tip_budget 135000
```

Checkpoint zostanie zapisany w `checkpoints/`.

## Ewaluacja

```bash
cd src

python evaluate.py --checkpoint ../models/attention_unet_rfbcs.pth \
                   --uhdm_dir /path/to/UHDM \
                   --tip_dir /path/to/TIP2018
```

## Modele

| Plik | Opis |
|------|------|
| `attention_unet_rfbcs.pth` | RFBCS, 0.1*MSE + (1-SSIM) |
| `attention_unet_full.pth` | Pełny zbiór, 0.1*MSE + (1-SSIM) |

## Wyniki

**Z RFBCS (cs-AUid):**

| Zbiór | SSIM | PSNR |
|-------|------|------|
| UHDM | 0.80 | 19.31 |
| TIP2018 | 0.91 | 27.46 |

**Bez RFBCS (AUid):**

| Zbiór | SSIM | PSNR |
|-------|------|------|
| UHDM | 0.82 | 19.48 |
| TIP2018 | 0.94 | 28.58 |
