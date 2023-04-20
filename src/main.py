import time
import json
import option
import torch
import torch.optim as optim
import utils.sample as sample
import utils.dataset as dataset

from tqdm import tqdm
from model import YOLOv3, Recommendation
from torch.utils.data import DataLoader
from utils.one_hot_encoder import OneHotEncoder


def main():
    args = option.get_args()

    if args.mode == 'train':
        train(args)

    elif args.mode == 'test':
        test(args)

    else:
        print('wrong mode!')


def test(args):
    print('>>> test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv3().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    enc = OneHotEncoder()
    enc.load_encoder('src/utils')

    rec_model = Recommendation(enc)
    rec_model.load('model')

    valid_fashionDS = dataset.KFashionDS(path='data/valid', preload=False)
    valid_loader = DataLoader(valid_fashionDS, args.batch_size, shuffle=True, collate_fn=valid_fashionDS.collate_fn)
    
    with torch.no_grad():
        for images, targets, _ in valid_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            start_time = time.time()
            
            output, _ = model(images, targets)
            print(model.yolo_detection.metrics)
            end_time = time.time()

            plz = images.detach().cpu()
            targets = targets.detach().cpu()
            output = output.detach().cpu()
            sample.show_sample(plz, output, './result', targets[:, 1:], enc, rec_model)
            break


    print(f'>>> test done ({end_time-start_time}s)')

def train(args):
    print('>>> train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('>>> train recommendation model')
    enc = OneHotEncoder()
    enc.load_encoder('src/utils')
    rec_model = Recommendation(enc)
    rec_model.train('data/train')
    rec_model.save('model')

    print('>>> train detection model')
    model = YOLOv3().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    batch_size = args.batch_size

    train_fashionDS = dataset.KFashionDS(path='data/train', preload=False)
    valid_fashionDS = dataset.KFashionDS(path='data/valid', preload=False)

    train_loader = DataLoader(train_fashionDS, batch_size, shuffle=True, collate_fn=train_fashionDS.collate_fn)
    valid_loader = DataLoader(valid_fashionDS, batch_size, shuffle=True, collate_fn=valid_fashionDS.collate_fn)

    optimizer = optim.Adam(model.parameters())
    if checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('>>> optimizer is loaded!')

    start_epochs = checkpoint['epochs'] + 1

    train_metrics = []
    valid_metrics = []

    for epoch in range(start_epochs, start_epochs + args.epochs):
        # Train
        model.train()

        train_n = 0
        train_metric = {
            'loss_box': 0,
            'loss_conf': 0,
            'loss_label': 0,
            'loss_total': 0,
            'acc_is_top': 0,
            'acc_top': 0,
            'acc_bottom': 0,
            'acc_total': 0,
        }

        for i, (images, targets, _) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            targets = targets.to(device)

            _, loss = model(images, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_n += images.size(0)
            for key in train_metric:
                train_metric[key] += model.yolo_detection.metrics[key] * images.size(0)

            if i % args.log_freq == 0:
                print(epoch, i, model.yolo_detection.metrics)

        for key in train_metric:
            train_metric[key] /= train_n
        train_metrics.append(train_metric)

        # Valid
        model.eval()

        valid_n = 0
        valid_metric = {
            'loss_box': 0,
            'loss_conf': 0,
            'loss_label': 0,
            'loss_total': 0,
            'acc_is_top': 0,
            'acc_top': 0,
            'acc_bottom': 0,
            'acc_total': 0,
        }

        with torch.no_grad():
            for images, targets, _ in tqdm(valid_loader):
                images = images.to(device)
                targets = targets.to(device)

                _, _ = model(images, targets)

                valid_n += images.size(0)
                for key in valid_metric:
                    valid_metric[key] += model.yolo_detection.metrics[key] * images.size(0)

        for key in valid_metric:
            valid_metric[key] /= valid_n
        valid_metrics.append(valid_metric)

        print(f'>>> {epoch} epochs train\n{train_metric}')
        print(f'>>> {epoch} epochs valid\n{valid_metric}')

        checkpoint = {
            'epochs': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': {'train': train_metric, 'valid': valid_metric}
        }

        torch.save(checkpoint, f'model/yolo_{epoch}.pt')

    result_metric = {'train':train_metrics, 'valid': valid_metrics}
    with open('result/metrics.json', 'w') as fp:
        json.dump(result_metric, fp)
    print('>>> train done')


if __name__ == '__main__':
    main()