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
            'bndbox': (
                int(object.find('bndbox/xmin').text),
                int(object.find('bndbox/ymin').text),
                int(object.find('bndbox/xmax').text),
                int(object.find('bndbox/ymax').text))}  for object in root.findall('object')]
        
        
    extracted_info = {
        'folder': root.find('folder').text,
        'filename': root.find('filename').text,
        'size': size,
        'objects': objects
    }
    
    return extracted_info    