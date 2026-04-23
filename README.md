# Optical Character Recognition via Feature Extraction

An OCR system built **from scratch** using OpenCV's feature extraction and connected component analysis — no off-the-shelf OCR libraries. Characters are recognized by matching template images against a grayscale test image using custom feature descriptors.

## How it works

**Inputs:**
- A set of template images — each representing one character to recognize
- A grayscale test image containing characters to identify

**Assumptions:**
- Characters are darker than the background
- Background color touches the image boundary
- Characters are separated by at least one background pixel
- Characters may have varying gray levels

**Pipeline:**
```
Template Characters + Grayscale Test Image
        |
Connected Component Analysis
        |
Feature Extraction (OpenCV)
        |
Template Matching
        |
Recognized Characters + Positions
```

## Key Techniques

- **Connected component labeling** — isolates individual characters from the background
- **Feature extraction** — extracts discriminative descriptors from character regions
- **Template matching** — compares extracted features against known character templates
- **Character localization** — outputs bounding regions and identity of each recognized character

## Tech Stack

`Python` · `OpenCV` · `NumPy` · `Feature Extraction` · `Connected Components`

## Setup

```bash
git clone https://github.com/mitalildeshpande/Optical-Character-Extraction-using-feature-extraction.git
cd Optical-Character-Extraction-using-feature-extraction
pip install opencv-python numpy
python ocr.py
```