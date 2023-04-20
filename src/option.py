import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='train / test')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size of batch')
    parser.add_argument('--log_freq', type=int, default=1,
                        help='Frequency to log and save model')
    parser.add_argument('--checkpoint', type=str, default='model/yolo_ver2_50.pt',
                        help='trained YOLO checkpoint path')
    return parser.parse_args()


if __name__ == '__main__':
    print(get_args())