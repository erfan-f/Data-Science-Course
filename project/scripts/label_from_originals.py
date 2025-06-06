import os
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

# === SETTINGS ===
image_dir = "images"
xml_dir = "annotations"
output_csv = "ocr_annotations.csv"

# Create CSV if not exists
if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
else:
    df = pd.DataFrame(columns=[
        'filename', 'xmin', 'ymin', 'xmax', 'ymax',
        'ocr_text', 'worth_ocr'
    ])

already_labeled = set(df['filename'].tolist())

# === Helper to update XML ===
def add_ocr_to_xml(xml_path, ocr_text, worth_ocr):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    obj = root.find("object")
    if obj is None:
        print("⚠️ No object tag found.")
        return

    # Add <ocr_text> and <worth_ocr> under <object>
    for tag, value in [('ocr_text', ocr_text), ('worth_ocr', str(worth_ocr))]:
        existing = obj.find(tag)
        if existing is not None:
            obj.remove(existing)

        elem = ET.SubElement(obj, tag)
        elem.text = value

    tree.write(xml_path)

# === Go through all XML files ===
for xml_file in sorted(os.listdir(xml_dir)):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text.strip()
    image_path = os.path.join(image_dir, filename)
    if filename in already_labeled:
        continue

    obj = root.find("object")
    bbox = obj.find("bndbox")
    x0 = int(bbox.find("xmin").text)
    y0 = int(bbox.find("ymin").text)
    x1 = int(bbox.find("xmax").text)
    y1 = int(bbox.find("ymax").text)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image: {filename}")
        continue

    plate = image[y0:y1, x0:x1]
    plate_rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)

    # Show cropped plate
    plt.imshow(plate_rgb)
    plt.title(filename)
    plt.axis("off")
    plt.show()

    # Ask for labeling
    ocr_text = input("👉 What is the actual plate text? (or [NO TEXT]): ").strip()
    worth_ocr = input("❓ Worth OCR? (1 = yes, 0 = no): ").strip()

    try:
        worth_ocr = int(worth_ocr)
        assert worth_ocr in [0, 1]
    except:
        print("⚠️ Invalid input. Skipping...")
        continue

    # Save to CSV
    df.loc[len(df)] = {
        'filename': filename,
        'xmin': x0, 'ymin': y0, 'xmax': x1, 'ymax': y1,
        'ocr_text': ocr_text,
        'worth_ocr': worth_ocr
    }
    df.to_csv(output_csv, index=False)

    # Save to XML
    add_ocr_to_xml(xml_path, ocr_text, worth_ocr)

    print(f"✅ Labeled: {filename} (saved to CSV & XML)")
    plt.close()

print("🎉 All done. Labels saved.")





import os
import xml.etree.ElementTree as ET

# === Settings ===
xml_dir = "annotations"

missing = []

for xml_file in sorted(os.listdir(xml_dir)):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    obj = root.find("object")
    if obj is None:
        missing.append((xml_file, "missing <object>"))
        continue

    ocr_text = obj.find("ocr_text")
    worth_ocr = obj.find("worth_ocr")

    if ocr_text is None or worth_ocr is None:
        missing.append((xml_file, "missing ocr_text or worth_ocr"))

if not missing:
    print("✅ All annotations are labeled.")
else:
    print(f"❌ {len(missing)} annotations missing labels:")
    for f, reason in missing:
        print(f" - {f}: {reason}")
