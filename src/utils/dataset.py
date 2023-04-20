import json
import torch
import random
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image, ImageOps
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader

class KFashionDS(Dataset):
    def __init__(self, path, preload=False):
        super().__init__()
        self.image_zip = ZipFile(f'{path}/image.zip')
        self.label_zip = ZipFile(f'{path}/label.zip')
        self.id_list = [x.filename[:-4] for x in self.image_zip.infolist()]
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.preload = preload
        if preload:
            self.image_list = []
            self.annt_list = []
            for file_id in tqdm(self.id_list, desc="Preloading"):
                self.image_list.append(self.image_zip.open(f'{file_id}.jpg'))
                with self.label_zip.open(f'{file_id}.json') as f:
                    annt = json.loads(f.read())
                self.annt_list.append(annt)

    def __getitem__(self, index):
        if self.preload:
            img = self.image_list[index]
            img = Image.open(img).convert('RGB')
            annt = self.annt_list[index]
        else:
            file_id = self.id_list[index]
            temp = self.image_zip.open(f'{file_id}.jpg')
            img = Image.open(temp).convert('RGB')
            annt = json.loads(self.label_zip.open(f'{file_id}.json').read())
        style = annt['style']
        flipped = random.randint(0,1)
        if flipped == 0:
            label = [
                [annt['top_coord']['x'], annt['top_coord']['y'], annt['top_coord']['width'], annt['top_coord']['height'], 1],
                [annt['bottom_coord']['x'], annt['bottom_coord']['y'], annt['bottom_coord']['width'], annt['bottom_coord']['height'], 0]
            ]
        else:
            img = ImageOps.mirror(img)
            label = [
                [1 - annt['top_coord']['x'], annt['top_coord']['y'], annt['top_coord']['width'], annt['top_coord']['height'], 1],
                [1 - annt['bottom_coord']['x'], annt['bottom_coord']['y'], annt['bottom_coord']['width'], annt['bottom_coord']['height'], 0]
            ]
        label[0] += annt['top_label'][:82]
        label[1] += annt['bottom_label'][:82]
        img = self.transform(img)
        return img, torch.tensor(label), style

    def collate_fn(self, batch):
        images, labels, styles = list(zip(*batch))
        targets = []
        for idx, label in enumerate(labels):
            targets.append(torch.cat((torch.tensor([idx]), label[0])))
            targets.append(torch.cat((torch.tensor([idx]), label[1])))
        images = torch.stack([image for image in images])
        targets = torch.stack(targets)
        return images, targets, torch.tensor(styles)

    def __len__(self):
        return len(self.id_list)


if __name__ == '__main__':
    fashionDS = KFashionDS('data/train')
    train_dl = DataLoader(fashionDS, 2, True, collate_fn=fashionDS.collate_fn)
    for images, labels, styles in train_dl:
        print(images)
        print(labels)
        print(styles)
        break