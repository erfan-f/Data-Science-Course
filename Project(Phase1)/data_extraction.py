from function import parse_xml
import pickle
import os
from PIL import Image

image_dir = 'data/images'
annot_dir = 'data/annotations'

image_files = os.listdir(image_dir)
annot_files = os.listdir(annot_dir)

dataset = {}

for case in image_files:
    case_info = {
        'image': Image.open(os.path.join(image_dir, case)),
        'annotation': parse_xml(os.path.join(annot_dir, case.replace('.png', '.xml')))
    }
    dataset[case.replace('.png', '')] = case_info
    
with open('data/dataset.pkl', 'wb') as file:
        pickle.dump(dataset, file)