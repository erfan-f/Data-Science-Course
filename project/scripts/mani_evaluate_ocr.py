# evaluate_ocr.py

import pandas as pd
import sqlite3
import Levenshtein

db_path = "database/dataset.db"
ocr_result_path = "ocr_results.csv"

def normalize_text(text):
    replacements = {'o': '0', 'O': '0', 'l': '1', 'I': '1', ']': '1'}
    for k, v in replacements.items():
        text = text.replace(k, v).replace(k.upper(), v)
    return text

def evaluate():
    # Load OCR predictions
    df_ocr = pd.read_csv(ocr_result_path)

    # Load GT from DB
    conn = sqlite3.connect(db_path)
    df_gt = pd.read_sql_query("""
        SELECT preprocessed_plate, ocr_text
        FROM engineered_plate_features
        WHERE ocr_text IS NOT NULL AND ocr_text != '[NO TEXT]'
    """, conn)
    conn.close()

    df = pd.merge(df_ocr, df_gt, on='preprocessed_plate', suffixes=('_ocr', '_gt'))

    # Comparisons
    df['exact_match'] = df['ocr_text_ocr'] == df['ocr_text_gt']
    df['normalized_match'] = df.apply(
        lambda row: normalize_text(row['ocr_text_ocr']) == normalize_text(row['ocr_text_gt']), axis=1
    )
    df['levenshtein'] = df.apply(
        lambda row: Levenshtein.distance(row['ocr_text_gt'], row['ocr_text_ocr']), axis=1
    )

    # Report
    print("📊 Evaluation Results:")
    print(f"- Exact Match Accuracy       : {df['exact_match'].mean():.2f}")
    print(f"- Normalized Match Accuracy  : {df['normalized_match'].mean():.2f}")
    print(f"- Average Levenshtein Dist.  : {df['levenshtein'].mean():.2f}")

    df.to_csv("ocr_evaluation_detailed.csv", index=False)
    print("✅ Saved detailed results to ocr_evaluation_detailed.csv")

if __name__ == "__main__":
    evaluate()
