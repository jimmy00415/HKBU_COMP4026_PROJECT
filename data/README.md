# Data Directory

This directory holds all datasets used by the project. **Do not commit data files — they are git-ignored.**

## Required Datasets

### 1. FER-2013 (Facial Expression Recognition)
- **Size:** ~48 MB (CSV)
- **Download:** https://www.kaggle.com/datasets/msambare/fer2013
- **Place as:** `data/fer2013/fer2013.csv`
- **Details:** 35,887 grayscale 48×48 images, 7 expression classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

### 2. Pins Face Recognition
- **Size:** ~1.5 GB
- **Download:** https://www.kaggle.com/datasets/hereisburak/pins-face-recognition
- **Place as:** `data/pins_face_recognition/<identity_folder>/<image>.jpg`
- **Details:** 105 identities, 17,534 cropped celebrity face images

### 3. CelebAMask-HQ (Optional — for face parsing & anonymizer evaluation)
- **Size:** ~6 GB
- **Download:** https://github.com/switchablenorms/CelebAMask-HQ
- **Place as:** `data/CelebAMask-HQ/`
  - `CelebA-HQ-img/` — high-res face images
  - `CelebAMask-HQ-mask-anno/` — 19-class semantic masks

## Directory Structure After Setup

```
data/
├── fer2013/
│   └── fer2013.csv
├── pins_face_recognition/
│   ├── pins_Adam Sandler/
│   ├── pins_Adriana Lima/
│   └── ...  (105 identity folders)
└── CelebAMask-HQ/          (optional)
    ├── CelebA-HQ-img/
    └── CelebAMask-HQ-mask-anno/
```

## Notes
- The preprocessing pipeline will detect, align, and crop all images to **256×256 RGB** (the canonical face-crop contract).
- Preprocessed/cached crops are stored in `cache/` (also git-ignored).
