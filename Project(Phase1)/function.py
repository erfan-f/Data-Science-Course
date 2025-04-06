import xml.etree.ElementTree as ET

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = {
        'width': root.find('size/width').text,
        'height': root.find('size/height').text,
        'depth': root.find('size/depth').text
    }
        
    objects = [{
            'difficult': False if object.find('difficult').text == '0' else True,
            'bndbox': {
                'xmin': object.find('bndbox/xmin').text,
                'ymin': object.find('bndbox/ymin').text,
                'xmax': object.find('bndbox/xmax').text,
                'ymax': object.find('bndbox/ymax').text,
            }
        } for object in root.findall('object')]
        
        
    extracted_info = {
        'folder': root.find('folder').text,
        'filename': root.find('filename').text,
        'size': size,
        'objects': objects
    }
    
    return extracted_info    