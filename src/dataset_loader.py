import os
import xml.etree.ElementTree as ET
from PIL import Image

class DatasetLoader:
    def __init__(self, image_dir, annot_dir):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.samples = []

        for fname in os.listdir(image_dir):
            if fname.endswith(".png"):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(image_dir, fname)
                xml_path = os.path.join(annot_dir, f"{base}.xml")
                if os.path.exists(xml_path):
                    labels = self._parse_label(xml_path)
                    self.samples.append((img_path, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, labels, img_path

    def _parse_label(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall("object")
        labels = [obj.find("name").text.strip().lower() for obj in objects]
        return labels
