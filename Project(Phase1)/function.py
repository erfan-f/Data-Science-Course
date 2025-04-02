import xml.etree.ElementTree as ET

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = {
        'width': root.find('size/width').text,
        'height': root.find('size/height').text,
        'depth': root.find('size/depth').text
    }
    
    object = {
        'difficult': False if root.find('object/difficult').text == '0' else True,
        'bndbox': {
            'xmin': root.find('object/bndbox/xmin').text,
            'ymin': root.find('object/bndbox/ymin').text,
            'xmax': root.find('object/bndbox/xmax').text,
            'ymax': root.find('object/bndbox/ymax').text
        }
    }
        
    extracted_info = {
        'folder': root.find('folder').text,
        'filename': root.find('filename').text,
        'size': size,
        'object': object
    }
    
    return extracted_info    