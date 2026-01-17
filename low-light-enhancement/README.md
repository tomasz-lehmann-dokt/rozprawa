# Wzmacnianie obrazu w warunkach niedoświetlenia

Swin-UNet do poprawy jakości obrazów niedoświetlonych. Szczegóły w rozdziale 5 rozprawy.

## Struktura kodu

| Plik | Opis |
|------|------|
| `model.py` | Swin-UNet (transformer encoder-decoder) |
| `dataset.py` | Loader LOL/LOL-v2/SID z strategiami wyboru ekspozycji |
| `loss.py` | Funkcje straty: MSE, FN-loss, LYT, LPIPS-based |
| `metrics.py` | PSNR, SSIM |
| `train.py` | Trening z AdamW, cosine annealing |
| `evaluate.py` | Ewaluacja |

## Dane

- **LOL v1**: https://daooshee.github.io/BMVC2018website/
- **LOL v2**: https://github.com/flyywh/CVPR-2020-Semi-Low-Light
- **SID**: https://github.com/cchen156/Learning-to-See-in-the-Dark

Struktura po pobraniu:

```
datasets/
├── LOL/
│   ├── our485/
│   │   ├── low/         # niedoświetlone (trening)
│   │   └── high/        # referencyjne
│   └── eval15/
│       ├── low/         # niedoświetlone (test)
│       └── high/
├── LOL-v2/
│   ├── Real_captured/
│   │   ├── Train/Low/, Train/Normal/
│   │   └── Test/Low/, Test/Normal/
│   └── Synthetic/
│       ├── Train/Low/, Train/Normal/
│       └── Test/Low/, Test/Normal/
└── SID/
    ├── Dataset_Part1/
    │   ├── Label/       # referencyjne JPG
    │   └── <scene_id>/  # niedoświetlone JPG
    └── Dataset_Part2/
```

## Trening

```bash
cd src

# Z funkcją straty LPIPS
python train.py --datasets_dir /path/to/datasets \
                --epochs 100 \
                --batch_size 4 \
                --crop_size 256 \
                --embed_dim 512 \
                --depths 2 4 8 2 \
                --alpha_lpips 0.1 \
                --use_sid \
                --sid_selection darkest

# Bez SID
python train.py --datasets_dir /path/to/datasets \
                --epochs 100 \
                --batch_size 4
```

Checkpoint zostanie zapisany w `checkpoints/`.

## Ewaluacja

```bash
cd src

python evaluate.py --checkpoint ../models/swin_unet_lpips.pth \
                   --datasets_dir /path/to/datasets \
                   --embed_dim 512 \
                   --depths 2 4 8 2
```

## Modele

| Plik | Funkcja straty |
|------|----------------|
| `swin_unet_lpips.pth` | Smooth-L1 + LPIPS + MS-SSIM + PSNR + color + grad |
| `swin_unet_lyt.pth` | LYT (Smooth-L1 + VGG + hist + PSNR + color + MS-SSIM) |
| `swin_unet_fn.pth` | FN-loss (HVI + sRGB) |
| `swin_unet_mse.pth` | MSE |

## Wyniki

| Zbiór | PSNR | SSIM |
|-------|------|------|
| LOL-v1 | 21.77 | 0.827 |
| LOL-v2-real | 22.60 | 0.826 |
| LOL-v2-synth | 22.42 | 0.897 |
