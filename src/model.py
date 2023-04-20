import json
import torch
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from zipfile import ZipFile
from utils.one_hot_encoder import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

def make_conv(in_channels, out_channels, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            make_conv(channels, channels // 2, kernel_size=1, padding=0),
            make_conv(channels// 2, channels , kernel_size=3)
        )

    def forward(self, x):
        return x + self.block(x)


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.body = nn.Sequential(
            make_conv(3, 32, kernel_size=3),
            make_conv(32, 64, kernel_size=3, stride=2),
            ResidualBlock(channels=64),
            make_conv(64, 128, kernel_size=3, stride=2),
            ResidualBlock(channels=128),
            ResidualBlock(channels=128),
            make_conv(128, 256, kernel_size=3, stride=2),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            make_conv(256, 512, kernel_size=3, stride=2),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            make_conv(512, 1024, kernel_size=3, stride=2),
            ResidualBlock(channels=1024),
            ResidualBlock(channels=1024),
            ResidualBlock(channels=1024),
            ResidualBlock(channels=1024),
        )

    def forward(self, x):
        return self.body(x)


class YOLODetection(nn.Module):
    def __init__(self, anchors, img_size, n_top, n_bottom):
        super(YOLODetection, self).__init__()
        self.anchors = anchors
        self.n_anchor = len(anchors)
        self.img_size = img_size
        self.n_top = n_top
        self.n_bottom = n_bottom

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.threshold = 0.5
        self.no_obj_weight = 10
        self.box_weight = 1
        self.conf_weight = 1
        self.label_weight = 1

        self.metrics = {}

    def forward(self, x, target):
        device = torch.device('cuda' if x.is_cuda else 'cpu')
        self.anchors = self.anchors.to(device)
        n_batch = x.size(0)
        n_grid = x.size(2)
        stride = self.img_size / n_grid

        # x : [n_batch, n_anchor * (5 + 1 + self.n_top + self.n_bottom), n_grid, n_grid]
        # --> [n_batch, n_anchor, n_grid, n_grid, 5 + 1 + self.n_top + self.n_bottom]
        pred = x.view(n_batch, self.n_anchor, 5 + 1 + self.n_top + self.n_bottom, n_grid, n_grid)\
                .permute(0, 1, 3, 4, 2).contiguous()

        # predicted grid
        pred_cx = torch.sigmoid(pred[..., 0])
        pred_cy = torch.sigmoid(pred[..., 1])
        pred_w = pred[..., 2]
        pred_h = pred[..., 3]

        # grid offset
        grid_offset = torch.arange(n_grid, dtype=torch.float, device=device).repeat(n_grid, 1)
        grid_x = grid_offset.view([1, 1, n_grid, n_grid])
        grid_y = grid_offset.t().view([1, 1, n_grid, n_grid])
        anchor_w = self.anchors[:, 0].view(1, -1, 1, 1)
        anchor_h = self.anchors[:, 1].view(1, -1, 1, 1)

        # [n_batch, n_anchor, n_grid, n_grid, [0:4](x, y, w, h) [4]conf [5:]class ]
        pred_grid = torch.zeros_like(pred[..., :4], device=device)
        pred_grid[..., 0] = (grid_x + pred_cx) * stride
        pred_grid[..., 1] = (grid_y + pred_cy) * stride
        pred_grid[..., 2] = anchor_w * torch.exp(pred_w) * self.img_size
        pred_grid[..., 3] = anchor_h * torch.exp(pred_h) * self.img_size
        pred_conf = torch.sigmoid(pred[..., 4])
        pred_is_top = torch.sigmoid(pred[..., 5])
        pred_class = torch.sigmoid(pred[..., 6:])

        # output form
        output = torch.cat(
            (pred_grid.view(n_batch, -1, 4),
             pred_conf.view(n_batch, -1, 1),
             pred_is_top.view(n_batch, -1, 1),
             pred_class.view(n_batch, -1, self.n_top + self.n_bottom)),
            -1)


        # if test phase
        if target is None:
            return output, 0


        # train phase
        # select best anchor to target
        batch_i = target[:, 0].long()
        t_cx, t_cy = target[:, 1:3].t() * n_grid
        t_w, t_h = target[:, 3:5].t()

        anchor_to_target = torch.stack([self.anchor_to_target(anchor, t_w, t_h) for anchor in self.anchors])
        _, best_anchor_idx = anchor_to_target.max(0)

        # set object mask
        obj_mask = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid,
                               dtype=torch.bool, device=device)
        no_obj_mask = torch.ones(n_batch, self.n_anchor, n_grid, n_grid,
                                 dtype=torch.bool, device=device)

        t_ci, t_cj = t_cx.long(), t_cy.long()
        obj_mask[batch_i, best_anchor_idx, t_cj, t_ci] = 1
        no_obj_mask[batch_i, best_anchor_idx, t_cj, t_ci] = 0

        for i, anchor_ious in enumerate(anchor_to_target.t()):
            no_obj_mask[batch_i[i], anchor_ious > self.threshold, t_cj[i], t_ci[i]] = 0

        # set target grid
        target_cx = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid,
                                dtype=torch.float, device=device)
        target_cy = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid,
                                dtype=torch.float, device=device)
        target_w = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid,
                               dtype=torch.float, device=device)
        target_h = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid,
                               dtype=torch.float, device=device)

        target_cx[batch_i, best_anchor_idx, t_cj, t_ci] = t_cx - t_cx.floor()
        target_cy[batch_i, best_anchor_idx, t_cj, t_ci] = t_cy - t_cy.floor()
        target_w[batch_i, best_anchor_idx, t_cj, t_ci] = torch.log(t_w / self.anchors[best_anchor_idx][:, 0] + 1e-16)
        target_h[batch_i, best_anchor_idx, t_cj, t_ci] = torch.log(t_h / self.anchors[best_anchor_idx][:, 1] + 1e-16)

        # set is_top
        target_is_top = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid,
                                    dtype=torch.float, device=device)
        target_is_top[batch_i, best_anchor_idx, t_cj, t_ci] = target[:, 5]

        # set top / bottom mask
        is_top_mask = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, self.n_top + self.n_bottom,
                                  dtype=torch.bool, device=device)
        is_bottom_mask = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, self.n_top + self.n_bottom,
                                    dtype=torch.bool, device=device)

        top_mask = torch.zeros(self.n_top + self.n_bottom, dtype=torch.bool, device=device)
        bottom_mask = torch.zeros(self.n_top + self.n_bottom, dtype=torch.bool, device=device)

        top_mask[:self.n_top] = True
        bottom_mask[self.n_top:] = True

        for i, is_top in enumerate(target_is_top[batch_i, best_anchor_idx, t_cj, t_ci].long().bool()):
            if is_top:
                is_top_mask[batch_i[i], best_anchor_idx[i], t_cj[i], t_ci[i]] = top_mask
            else:
                is_bottom_mask[batch_i[i], best_anchor_idx[i], t_cj[i], t_ci[i]] = bottom_mask

        exist_top = is_top_mask.sum() != 0
        exist_bottom = is_bottom_mask.sum() != 0

        # set target class
        target_class = torch.zeros(n_batch, self.n_anchor, n_grid, n_grid, self.n_top + self.n_bottom,
                                   dtype=torch.float, device=device)
        target_class[batch_i, best_anchor_idx, t_cj, t_ci] = target[:, 6:]

        # target conf
        target_conf = obj_mask.float()

        # bounding box loss
        loss_cx = self.mse_loss(pred_cx[obj_mask], target_cx[obj_mask])
        loss_cy = self.mse_loss(pred_cy[obj_mask], target_cy[obj_mask])
        loss_w = self.mse_loss(pred_w[obj_mask], target_w[obj_mask])
        loss_h = self.mse_loss(pred_h[obj_mask], target_h[obj_mask])
        loss_box = loss_cx + loss_cy + loss_w + loss_h

        # confidence loss --> obj + no_obj * weight
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], target_conf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], target_conf[no_obj_mask])
        loss_conf = loss_conf_obj + loss_conf_no_obj * self.no_obj_weight

        # class loss --> is_top + top + bottom
        loss_label = self.bce_loss(pred_is_top[obj_mask], target_is_top[obj_mask])

        if exist_top:
            loss_top_class = self.bce_loss(pred_class[is_top_mask], target_class[is_top_mask])
            loss_label += loss_top_class
        if exist_bottom:
            loss_bottom_class = self.bce_loss(pred_class[is_bottom_mask], target_class[is_bottom_mask])
            loss_label += loss_bottom_class

        # total loss
        loss_box = loss_box * self.box_weight
        loss_conf = loss_conf * self.conf_weight
        loss_label = loss_label * self.label_weight
        loss_total = loss_box + loss_conf + loss_label

        # metric phase
        n_obj = obj_mask.sum()

        acc_is_top = 0
        acc_is_top += (target_is_top[obj_mask][pred_is_top[obj_mask] > 0.5] == 1).sum() / n_obj
        acc_is_top += (target_is_top[obj_mask][pred_is_top[obj_mask] < 0.5] == 0).sum() / n_obj

        acc_top, acc_bottom = self.get_acc(pred_class[is_top_mask], target_class[is_top_mask],
                                                     pred_class[is_bottom_mask], target_class[is_bottom_mask],
                                                     device)
        acc_total = (acc_top + acc_bottom) / 2

        self.metrics = {
            'loss_box': loss_box.detach().cpu().item(),
            'loss_conf': loss_conf.detach().cpu().item(),
            'loss_label': loss_label.detach().cpu().item(),
            'loss_total': loss_total.detach().cpu().item(),
            'acc_is_top': acc_is_top.detach().cpu().item(),
            'acc_top': acc_top.detach().cpu().item(),
            'acc_bottom': acc_bottom.detach().cpu().item(),
            'acc_total': acc_total.detach().cpu().item(),
        }

        return output, loss_total


    # calc iou between anchor and target box
    def anchor_to_target(self, anchor, t_w, t_h):
        a_w, a_h = anchor
        inter = torch.min(a_w, t_w) * torch.min(a_h, t_h)
        union = (t_w * t_h) + (a_w * a_h) - inter + 1e-16
        return inter / union


    # get acc
    def get_acc(self, pred_top_1, target_top_1, pred_bottom_1, target_bottom_1, device):
        # top: [color14 sub-color14 sleeve5 length3 category6 fit4 print7]
        # bottom: [color14 category5 fit5 length5]

        pred_top = pred_top_1.view(-1, self.n_top)
        target_top = target_top_1.view(-1, self.n_top)
        pred_bottom = pred_bottom_1.view(-1, self.n_bottom)
        target_bottom = target_bottom_1.view(-1, self.n_bottom)

        n_top = pred_top.shape[0]
        n_bottom = pred_bottom.shape[0]

        max_pred_top = torch.zeros_like(pred_top, dtype=torch.bool, device=device)
        max_pred_bottom = torch.zeros_like(pred_bottom, dtype=torch.bool, device=device)

        indice_top = [14, 14, 5, 3, 6, 4, 7]
        indice_bottom = [14, 5, 5, 5]

        b_top = torch.arange(n_top, dtype=torch.long, device=device)
        b_bottom = torch.arange(n_bottom, dtype=torch.long, device=device)

        idx = 0
        n_label_top = len(indice_top)
        for i in range(n_label_top - 1):
            max_pred_top[b_top, idx + pred_top[:, idx:idx+indice_top[i]].argmax(1)] = 1
            idx += indice_top[i]

        idx = 0
        n_label_bottom = len(indice_bottom)
        for i in range(n_label_bottom - 1):
            max_pred_bottom[b_bottom, idx + pred_bottom[:, idx:idx+indice_bottom[i]].argmax(1)] = 1
            idx += indice_bottom[i]

        acc_top = ((max_pred_top & target_top.bool()).sum() / 7) / n_top
        acc_bottom = ((max_pred_bottom & target_bottom.bool()).sum() / 4) / n_bottom

        return acc_top, acc_bottom


class YOLOv3(nn.Module):
    def __init__(self, img_size=416, n_top=53, n_bottom=29):
        super(YOLOv3, self).__init__()
        self.anchors = torch.tensor([[0.37181956, 0.42933849],
                                    [0.56479795, 0.66844948],
                                    [0.28394044, 0.60433433]])
        self.last_out_channels = len(self.anchors) * (4 + 1 + 1 + n_top + n_bottom)

        self.darknet53 = Darknet53()
        self.detection_block = self.make_detection_block(1024, 512)
        self.yolo_detection = YOLODetection(self.anchors, img_size, n_top, n_bottom)

    def forward(self, x, target=None):
        x = self.darknet53(x)
        x = self.detection_block(x)
        output, loss = self.yolo_detection(x, target)

        return output, loss

    def make_detection_block(self, in_channels, out_channels):
        return nn.Sequential(
            make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            make_conv(out_channels, out_channels * 2, kernel_size=3),
            make_conv(out_channels * 2, out_channels, kernel_size=1, padding=0),
            make_conv(out_channels, out_channels * 2, kernel_size=3),
            make_conv(out_channels * 2, out_channels, kernel_size=1, padding=0),
            make_conv(out_channels, out_channels * 2, kernel_size=3),
            nn.Conv2d(out_channels * 2, self.last_out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

class Recommendation:
    def __init__(self, enc: OneHotEncoder):
        self.enc = enc
        pass
    def load(self, path):
        with open(f'{path}/top_dt.pickle', 'rb') as f:
            self.top_dt = pickle.load(f)
        with open(f'{path}/bottom_dt.pickle', 'rb') as f:
            self.bottom_dt = pickle.load(f)
    
    def save(self, path):
        with open(f'{path}/top_dt.pickle', 'wb') as f:
            pickle.dump(self.top_dt, f)
        with open(f'{path}/bottom_dt.pickle', 'wb') as f:
            pickle.dump(self.bottom_dt, f)
    
    def train(self, data_path: str):
        top_df = []
        bottom_df = []
        with ZipFile(f'{data_path}/label.zip') as zf:
            infos = zf.infolist()
            temp = zf.open(infos[0]).read()
            temp = json.loads(temp)
            top_label = self.enc.decode_tensor(temp['top_label'], 'top')
            bottom_label = self.enc.decode_tensor(temp['bottom_label'], 'bottom')
            top_column = [x[:x.find('/')] for x in top_label]
            bottom_column = [x[:x.find('/')] for x in bottom_label]
            for member in tqdm(infos, desc=f'Data'):
                temp = zf.open(member).read()
                temp = json.loads(temp)
                top_label = temp['top_label']
                bottom_label = temp['bottom_label']
                top_label_line = []
                bottom_label_line = []
                cnt = 0
                for x in self.enc.indices[:7]:
                    top_label_line.append(top_label[cnt:cnt + x].index(1))
                    cnt += x
                for x in self.enc.indices[7:]:
                    bottom_label_line.append(bottom_label[cnt:cnt + x].index(True))
                    cnt += x
                top_df.append(top_label_line)
                bottom_df.append(bottom_label_line)
        top_df = pd.DataFrame(top_df, columns=top_column)
        bottom_df = pd.DataFrame(bottom_df, columns=bottom_column)
        top_train = np.array(pd.DataFrame(top_df, columns=['top_color', 'top_sub_color', 'top_sleeve', 'top_length', 'top_category', 'top_fit', 'top_print']))
        bottom_train = np.array(pd.DataFrame(bottom_df, columns=['bottom_color', 'bottom_category', 'bottom_fit', 'bottom_length']))
        self.top_dt = []
        for idx in range(7):
            self.top_dt.append(DecisionTreeClassifier().fit(bottom_train, top_train[:, idx]))
        self.bottom_dt = []
        for idx in range(4):
            self.bottom_dt.append(DecisionTreeClassifier().fit(top_train, bottom_train[:, idx]))
    
    def test(self, input_cloth, cloth_type):
        if cloth_type == 'bottom':
            output = np.zeros(7, dtype = int)
            for idx in range(7):
                output[idx] = self.top_dt[idx].predict([input_cloth])
            output = self.enc.decode_label(output, 'top')
        elif cloth_type == 'top':
            cloth_type = 'bottom'
            output = np.zeros(4, dtype = int)
            for idx in range(4):
                output[idx] = self.bottom_dt[idx].predict([input_cloth])
            output = self.enc.decode_label(output, 'bottom')
        return output


def initial_model_creation():
    model = YOLOv3()

    darknet_model = torch.load('model/my_trained_darknet_171.pt')
    model.darknet53.body.load_state_dict(darknet_model['darknet53_body_state_dict'])

    checkpoint = {
        'epochs': 0,
        'state_dict': model.state_dict(),
        'optimizer': None,
        'metrics': None
    }

    torch.save(checkpoint, 'model/yolo_0.pt')


if __name__ == '__main__':
    initial_model_creation()