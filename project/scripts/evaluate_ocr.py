# evaluate_ocr.py

import pandas as pd
import Levenshtein

def normalize_text(text):
    replacements = {'o': '0', 'O': '0', 'l': '1', 'I': '1',']': '1'}
    for k, v in replacements.items():
        text = text.replace(k, v).replace(k.upper(), v)
    return text

def evaluate(ocr_result_path="ocr_results.csv", label_path="ocr_labels.csv"):
    df_ocr = pd.read_csv(ocr_result_path)
    df_labels = pd.read_csv(label_path)

    df = pd.merge(df_ocr, df_labels, on='preprocessed_plate', suffixes=('_ocr', '_label'))
    df = df[df['ocr_text_label'] != "[NO TEXT]"]  # Optional: ignore unreadable labels

    # Exact match
    df['exact_match'] = df['ocr_text_ocr'] == df['ocr_text_label']

    # Normalized match
    df['norm_match'] = df.apply(lambda row: normalize_text(row['ocr_text_ocr']) == normalize_text(row['ocr_text_label']), axis=1)

    # Levenshtein distance
    df['levenshtein'] = df.apply(lambda row: Levenshtein.distance(row['ocr_text_label'], row['ocr_text_ocr']), axis=1)

    print(f"✅ Evaluation Results:")
    print(f"- Exact Match Accuracy: {df['exact_match'].mean():.2f}")
    print(f"- Normalized Match Accuracy: {df['norm_match'].mean():.2f}")
    print(f"- Avg Levenshtein Distance: {df['levenshtein'].mean():.2f}")

    df.to_csv("ocr_evaluation_detailed.csv", index=False)

if __name__ == "__main__":
    evaluate()
