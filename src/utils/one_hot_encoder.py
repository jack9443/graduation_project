import os, json
from tqdm import tqdm
import torch

class OneHotEncoder:
    def __init__(self) -> None:
        self.label_encoder = {}
        self.label_decoder = []
        self.style_encoder = {}
        self.style_decoder = []
    
    def build_encoder(self, label_path):
        files = os.listdir(f'{label_path}')
        label_dict = {}
        for file in tqdm(files):
            annt = json.loads(open(f'{self.label_path}/{file}', 'r').read())
            for key in annt:
                if key == "top_coord" or key == "bottom_coord" or key == "image_id":
                    continue
                if key not in label_dict:
                    label_dict[key] = set()
                label_dict[key].add(annt[key])
        self.label_encoder = {}
        cnt = 0
        for label_type in label_dict:
            if label_type == 'style':
                continue
            for label_value in label_dict[label_type]:
                self.label_encoder[label_type + '/' + label_value] = cnt
                cnt += 1
        self.label_decoder = ['' for _ in range(cnt)]
        for label in self.label_encoder:
            self.label_decoder[self.label_encoder[label]] = label
        self.style_encoder = {}
        cnt = 0
        for label_type in label_dict:
            if label_type == 'style':
                for label_value in label_dict[label_type]:
                    self.style_encoder[label_value] = cnt
                    cnt += 1
        self.style_decoder = ['' for _ in range(cnt)]
        for style in self.style_encoder:
            self.style_decoder[self.style_encoder[style]] = style
        self.indices = []
        prev_type = ''
        cnt = 0
        for key in self.label_decoder:
            label_type = key.split('/')[0]
            if prev_type != label_type:
                prev_type = label_type
                self.indices.append(0)
            self.indices[-1] += 1
    
    def save_encoder(self, save_path):
        with open(f'{save_path}/one-hot-label.json', 'w') as f:
            json.dump(self.label_encoder, f, indent=2)
        with open(f'{save_path}/one-hot-label-decoder.json', 'w') as f:
            json.dump(self.label_decoder, f, indent=2)
        with open(f'{save_path}/one-hot-style.json', 'w') as f:
            json.dump(self.style_encoder, f, indent=2)
        with open(f'{save_path}/one-hot-style-decoder.json', 'w') as f:
            json.dump(self.style_decoder, f, indent=2)
    
    def load_encoder(self, load_path):
        with open(f'{load_path}/one-hot-label.json', 'r') as f:
            self.label_encoder = json.loads(f.read())
        with open(f'{load_path}/one-hot-label-decoder.json', 'r') as f:
            self.label_decoder = json.loads(f.read())
        with open(f'{load_path}/one-hot-style.json', 'r') as f:
            self.style_encoder = json.loads(f.read())
        with open(f'{load_path}/one-hot-style-decoder.json', 'r') as f:
            self.style_decoder = json.loads(f.read())
        self.indices = []
        prev_type = ''
        for key in self.label_decoder:
            label_type = key.split('/')[0]
            if prev_type != label_type:
                prev_type = label_type
                self.indices.append(0)
            self.indices[-1] += 1
    
    def encode(self, data):
        new_data = {
            'top_label': [0] * len(self.label_encoder),
            'bottom_label': [0] * len(self.label_encoder),
            }

        for label_type in data:
            if label_type in ["top_coord", "bottom_coord", "image_id", 'style']:
                continue
            if label_type[:3] == 'top':
                new_data['top_label'][self.label_encoder[label_type + '/' + data[label_type]]] = 1
            else:
                new_data['bottom_label'][self.label_encoder[label_type + '/' + data[label_type]]] = 1
        new_data['style'] = self.style_encoder[data['style']]
        return new_data

    def decode(self, data):
        labels = {
            'top_label': [],
            'bottom_label': []
            }
        for idx in range(len(data['top_label'])):
            if data['top_label'][idx] == 1:
                labels['top_label'].append(self.label_decoder[idx])
        for idx in range(len(data['bottom_label'])):
            if data['bottom_label'][idx] == 1:
                labels['bottom_label'].append(self.label_decoder[idx])
        labels['style'] = self.style_decoder[data['style']]
        return labels
    
    def decode_tensor(self, data, cloth_type):
        decoded = []
        for idx, x in enumerate(data):
            label = self.label_decoder[idx]
            if x and (label[:label.find('_')] == cloth_type or cloth_type == 'all'):
                decoded.append(self.label_decoder[idx])
        return decoded

    def label_argmax(self, label):
        max_pred = torch.zeros(sum(self.indices), dtype=torch.bool)
        n_label = len(self.indices)
        idx = 0
        for i in range(n_label):
            max_pred[idx + label[idx:idx + self.indices[i]].argmax(0)] = 1
            idx += self.indices[i]
        return max_pred
    
    def transform_label(self, label, cloth_type):
        result = []
        if cloth_type == 'top':
            cnt = 0
            for x in self.indices[:7]:
                result.append(label[cnt:cnt + x].index(True))
                cnt += x
        if cloth_type == 'bottom':
            cnt = sum(self.indices[:7])
            for x in self.indices[7:]:
                result.append(label[cnt:cnt + x].index(True))
                cnt += x
        return result
    
    def decode_label(self, label, cloth_type):
        result = []
        if cloth_type == 'top':
            cnt = 0
            for idx, x in enumerate(self.indices[:7]):
                result.append(self.label_decoder[cnt + label[idx]])
                cnt += x
        if cloth_type == 'bottom':
            cnt = sum(self.indices[:7])
            for idx, x in enumerate(self.indices[7:]):
                result.append(self.label_decoder[cnt + label[idx]])
                cnt += x
        return result