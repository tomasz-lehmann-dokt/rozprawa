# Rozprawa doktorska, Tomasz Lehmann, 2026

Kod źródłowy do rozprawy: *Zastosowanie lekkich architektur głębokich w percepcyjnej analizie i korygowaniu degradacji obrazu*.

## Struktura repozytorium

```
rozprawa/
├── biqa/                  # Bezwzorcowa ocena jakości obrazu (rozdział 3)
├── demoireing/            # Redukcja prążków mory (rozdział 4)
├── low-light-enhancement/ # Wzmacnianie niedoświetlenia (rozdział 5)
└── deblurring/            # Redukcja rozmycia z MOS (rozdział 6)
```

## Moduły

| Moduł | Architektura | Zbiory danych | Opis |
|-------|--------------|---------------|------|
| `biqa` | ConvNeXt-MLP, dual-Xception | KonIQ-10k | Predykcja MOS |
| `demoireing` | Attention U-Net | UHDM, TIP2018 | Usuwanie mory |
| `low-light-enhancement` | Swin-UNet | LOL, LOL-v2, SID | Poprawa niedoświetlenia |
| `deblurring` | Attention U-Net + MOS | HIDE | Usuwanie rozmycia |

## Instalacja

Wymagania zostały przetestowane na systemie z kartą NVIDIA GeForce RTX 3070 Laptop GPU oraz CUDA 12.8. Wersje pakietów PyTorch mogą różnić się w zależności od konfiguracji sprzętowej i wersji sterowników CUDA.

```bash
# Klonowanie repozytorium
git clone <repo_url>
cd rozprawa

# Utworzenie środowiska wirtualnego
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# lub: .venv\Scripts\activate  # Windows

# Instalacja zależności
pip install -r requirements.txt
```

W przypadku innej karty graficznej należy dostosować wersję PyTorch zgodnie z dokumentacją: https://pytorch.org/get-started/locally/

## Inferencja

### BIQA (predykcja MOS)

```bash
cd biqa/src
python evaluate.py --checkpoint ../models/convnext_mlp_koniq.pth \
                   --csv_path /path/to/koniq10k_scores_and_distributions.csv \
                   --images_dir /path/to/1024x768 \
                   --model convnext
```

### Demoireing

```bash
cd demoireing/src
python evaluate.py --checkpoint ../models/attention_unet_rfbcs.pth \
                   --uhdm_dir /path/to/UHDM \
                   --tip_dir /path/to/TIP2018
```

### Low-light enhancement

```bash
cd low-light-enhancement/src
python evaluate.py --checkpoint ../models/swin_unet_lpips.pth \
                   --datasets_dir /path/to/datasets
```

### Deblurring

```bash
cd deblurring/src
python evaluate.py --data_dir /path/to/HIDE_dataset \
                   --checkpoint ../models/best_model.pth
```

## DIQA

Zbiór 1000 obrazów z 10 modeli dyfuzyjnych do ewaluacji predyktorów jakości.

https://www.grafi.ii.pw.edu.pl/index.php/galerie/diffusion-images-for-quality-assessment
