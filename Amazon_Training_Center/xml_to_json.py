import xml.etree.ElementTree as ET
import json

def read_xml(file_path):
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None
    except FileNotFoundError:
        print("XML file not found.")
        return None

def convert_xml_to_json(xml_root):
    json_data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": []
    }

    for obj in xml_root.findall('object'):
        label = obj.find('name').text
        points = []
        for pt in obj.find('polygon').findall('pt'):
            x = float(pt.find('x').text)
            y = float(pt.find('y').text)
            points.append([x, y])

        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
        json_data["shapes"].append(shape)
    
    return json_data

def save_json(data, file_path):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"JSON data saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

def main():
    xml_file_path = 'Amazon_Training_Center/default/AL5.xml'
    json_file_path = 'AL_005.json'

    xml_root = read_xml(xml_file_path)
    if xml_root is None:
        return

    json_data = convert_xml_to_json(xml_root)
    save_json(json_data, json_file_path)

if __name__ == "__main__":
    main()
