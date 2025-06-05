import pandas as pd
import sqlite3
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === SETTINGS ===
train_mode = True  # Set to False to skip training
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

db_path = "your_database_path.db"
label_path = "ocr_labels.csv"

# === Load labels and features ===
df_labels = pd.read_csv(label_path)

conn = sqlite3.connect(db_path)
df_features = pd.read_sql_query("""
    SELECT preprocessed_plate, blur_score, bbox_width, bbox_height,
           aspect_ratio, area_fraction
    FROM engineered_plate_features
    WHERE preprocessed_plate IS NOT NULL
""", conn)

conn.close()

df = pd.merge(df_labels, df_features, on="preprocessed_plate", how="inner")

# === Prepare data ===
features = ['blur_score', 'bbox_width', 'bbox_height', 'aspect_ratio', 'area_fraction']
X = df[features]
y = df['worth_ocr']

# === Models to train/test ===
model_defs = {
    'random_forest': RandomForestClassifier(),
    'log_reg': LogisticRegression(max_iter=1000),
    'svm': SVC(probability=True),
    'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# === Training Mode ===
if train_mode:
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    for name, model in model_defs.items():
        print(f"\n🧠 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"📊 {name} performance:")
        print(classification_report(y_test, preds))
        joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))
    print("✅ All models trained and saved.")
else:
    # === Inference Mode: Use best model (e.g. RandomForest)
    model_name = "random_forest"  # You can change this to 'xgboost', etc.
    model_path = os.path.join(model_dir, f"{model_name}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    model = joblib.load(model_path)
    df['predicted_worth_ocr'] = model.predict(X)

    # Save results
    df[['preprocessed_plate', 'predicted_worth_ocr']].to_csv("predicted_worth_ocr.csv", index=False)

    # Optional: Update DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE engineered_plate_features ADD COLUMN predicted_worth_ocr INTEGER;")
    except:
        pass

    for _, row in df.iterrows():
        cursor.execute("""
            UPDATE engineered_plate_features
            SET predicted_worth_ocr = ?
            WHERE preprocessed_plate = ?
        """, (int(row['predicted_worth_ocr']), row['preprocessed_plate']))

    conn.commit()
    conn.close()
    print("✅ Predictions saved to DB and CSV.")
