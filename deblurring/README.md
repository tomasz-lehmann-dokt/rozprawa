# Redukcja rozmycia z komponentem MOS

Attention U-Net z percepcyjną funkcją straty opartą na estymacji MOS z modelu dual-Xception. Szczegóły w rozdziale 6 rozprawy.

## Struktura kodu

| Plik | Opis |
|------|------|
| `model.py` | Attention U-Net z bramkami uwagi na skip connections |
| `mos_model.py` | dual-Xception do estymacji MOS |
| `dataset.py` | Loader zbioru HIDE |
| `loss.py` | Kombinowana strata: λ_MSE·MSE + λ_SSIM·(1-SSIM) + λ_MOS·L_MOS |
| `features.py` | Ekstrakcja 9 parametrów obrazu dla modelu MOS |
| `metrics.py` | PSNR, SSIM |
| `train.py` | Trening z RMSProp, StepLR, early stopping |
| `evaluate.py` | Ewaluacja na zbiorze testowym |

## Dane

**HIDE**: https://github.com/joanshen0508/HA_deblur

Zbiór par obrazów rozmytych/ostrych. Rozmycie: uśrednienie 11 klatek z 240fps. Rozdzielczość: 1280×720.

Struktura po pobraniu:

```
HIDE_dataset/
├── train/              # obrazy rozmyte (trening)
├── test/
│   ├── test-close-ups/
│   └── test-long-shot/
└── GT/                 # obrazy ostre (ground truth)
```

## Funkcja straty

Zgodnie z równaniem (6.3) rozprawy:

```
L_total = λ_MSE·L_MSE + λ_SSIM·L_SSIM + λ_MOS·L_MOS
```

Komponent MOS (równanie 6.1):

```
L_MOS = 1 - σ(-MOS(ŷ) / MOS(x))
```

Optymalne wagi: λ_MSE=50.0, λ_SSIM=0.00699, λ_MOS=0.01792.

## Trening

```bash
cd src

# Pełna konfiguracja z MOS
python train.py --data_dir /path/to/HIDE_dataset \
                --mos_checkpoint ../models/dual_xception_mos.pth \
                --param_stats ../models/param_stats.npz \
                --epochs 50 \
                --batch_size 28 \
                --w_mse 50.0 \
                --w_ssim 0.00699 \
                --w_mos 0.01792

# Baseline (tylko MSE)
python train.py --data_dir /path/to/HIDE_dataset \
                --param_stats ../models/param_stats.npz \
                --w_mse 50.0 \
                --w_ssim 0.0 \
                --w_mos 0.0
```

Checkpoint zostanie zapisany w `checkpoints/`.

## Ewaluacja

```bash
cd src

python evaluate.py --data_dir /path/to/HIDE_dataset \
                   --checkpoint checkpoints/best_model.pth
```

## Modele

| Plik | Opis |
|------|------|
| `dual_xception_mos.pth` | Wytrenowany dual-Xception do estymacji MOS |
| `param_stats.npz` | Statystyki normalizacji parametrów |

## Wyniki

| Konfiguracja | SSIM | PSNR [dB] |
|--------------|------|-----------|
| Tylko MSE | 0.8665 | 24.742 |
| MSE + SSIM + MOS | 0.8702 | 24.843 |
