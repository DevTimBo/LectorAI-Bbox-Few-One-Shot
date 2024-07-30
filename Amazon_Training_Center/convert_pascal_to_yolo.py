# Author: Jason Pranata
import xml.etree.ElementTree as ET
import os

folder_path = 'train_dataset'

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_id):
    in_file = open(f'{folder_path}/XML/{image_id}.xml')
    out_file = open(f'{folder_path}/labels/{image_id}.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        print(f"Class: {cls}, Index: {cls_id}")  # Debug print
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]) + '\n')

classes = [
    "ad_unterschrift", "adress_aend", "ad_erzieher", "ad_neue_ad", "ad_schueler_unterschrift",
    "ad_erzieher_name", "ad_erzieher_vorname", "ad_erzieher_tel", "ad_erzieher_email",
    "ad_neue_ad_str_haus_nr", "ad_neue_ad_plz", "ad_neue_ad_stadt", "ad_schueler_datum",
    "schueler", "schueler_name", "schueler_vorname", "schueler_klasse",
    "ag", "ag_auswahl", "ag_unterschrift", "ag_schueler_datum",
    "ag_auswahl_wahl_1", "ag_auswahl_wahl_2", "ag_auswahl_wahl_3", "ag_schueler_unterschrift",
    "AL", "al_1", "al_2", "al_3", "allergien", "Sonstiges"]

image_ids = [os.path.splitext(f)[0] for f in os.listdir('default') if f.endswith('.xml')]
for image_id in image_ids:
    convert_annotation(image_id)
