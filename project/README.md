# 🚗 Car Plate Detection Pipeline

This project implements a complete AI pipeline for detecting and preprocessing car plate annotations. It includes database loading, data cleaning, feature engineering, and image preprocessing for OCR-ready plate regions.

---

## 📁 Project Structure

```
├── database/
│   └── dataset.db                 # SQLite database
├── content/                
|   ├── annotations/               # Raw annotation files (downloaded from kaggle.com/andrewmvd/car-plate-detection)
|   ├── images/                    # Raw car images (downloaded from kaggle.com/andrewmvd/car-plate-detection) 
|   ├── plates/                    # Folder for preprocessed plate images
├── scripts/
│   ├── database_connection.py     # Database connection utility
│   ├── load_data.py               # Loads data from database
│   ├── preprocess.py              # Data cleaning, normalization, blurry detection
│   ├── feature_engineering.py     # Advanced visual/spatial feature extraction
├── pipeline.py                    # Master script to run the full pipeline
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
```

---

## 🚀 Pipeline Overview

### 1. `database_connection.py`

Handles reusable database connection using SQLite.

### 2. `load_data.py`

Loads joined data from annotations and features tables into Pandas DataFrames.

### 3. `preprocess.py`

Cleans and prepares the data:

* Removes invalid entries (e.g., no plate annotations).
* Normalizes numeric features using Min-Max scaling.
* Detects blurry plates using Laplacian variance and marks them in the database.
* Generates grayscale, resized, and normalized plate crops for OCR.

### 4. `feature_engineering.py`

Adds advanced engineered features:

* `bbox_width`, `bbox_height`, `aspect_ratio`
* `area_fraction`, normalized center coordinates
* Plate margins (left/right/top/bottom)
* Saves features to both the database and `engineered_features.csv`

### 5. `pipeline.py`

Executes the entire workflow in sequence:

```bash
python pipeline.py
```

---

## 📦 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
numpy
pandas
opencv-python
scikit-learn
```

---

## 📂 Outputs

* Preprocessed plate images saved to: `content/plates/`
* Engineered features saved to:

  * `database/engineered_plate_features` (table)
  * `engineered_features.csv`

---

## 👥 Contributors

* Parsa Bukani - SID: 810102501
* Mani Hosseini - SID: 810102552
* Erfan Falahati - SID: 810102491