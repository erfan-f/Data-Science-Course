import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def label_plates():
    plate_dir = os.path.join('content', 'plates')  
    save_path = 'ocr_labels.csv'

    if os.path.exists(save_path):
        df_labels = pd.read_csv(save_path)
    else:
        df_labels = pd.DataFrame(columns=['preprocessed_plate', 'ocr_text', 'worth_ocr'])

    labeled = set(df_labels['preprocessed_plate'].tolist())

    plate_files = sorted([f for f in os.listdir(plate_dir) if f.endswith('.png')])

    for plate_file in plate_files:
        if plate_file in labeled:
            continue  

        img_path = os.path.join(plate_dir, plate_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not load: {plate_file}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(plate_file)
        plt.axis('off')
        plt.show()

        ocr_text = input("👉 What is the actual plate text? (or [NO TEXT] if unreadable): ").strip()
        worth_ocr = input("❓ Worth doing OCR on this? (1 = yes, 0 = no): ").strip()

        try:
            worth_ocr = int(worth_ocr)
            assert worth_ocr in [0, 1]
        except:
            print("⚠️ Invalid input for worth_ocr. Skipping.")
            continue

        df_labels.loc[len(df_labels)] = {
            'preprocessed_plate': plate_file,
            'ocr_text': ocr_text,
            'worth_ocr': worth_ocr
        }

        df_labels.to_csv(save_path, index=False)
        print(f"✅ Labeled {plate_file}\n")
        plt.close()

    print("🎉 Labeling complete.")





import os
import pandas as pd

def check_unlabeled(plate_dir='content/plates', label_path='ocr_labels.csv'):
    # Get all plate filenames
    all_files = sorted([f for f in os.listdir(plate_dir) if f.endswith('.png')])

    # Load labeled data
    if not os.path.exists(label_path):
        print("⚠️ Label file not found. No labels exist yet.")
        return

    df = pd.read_csv(label_path)
    labeled_files = set(df['preprocessed_plate'].tolist())

    # Find missing
    missing = [f for f in all_files if f not in labeled_files]

    if missing:
        print("❌ Plates missing labels:")
        for f in missing:
            print(f"- {f}")
        print(f"\nTotal missing: {len(missing)}")
    else:
        print("✅ All plates are labeled.")

if __name__ == "__main__":
    check_unlabeled()
