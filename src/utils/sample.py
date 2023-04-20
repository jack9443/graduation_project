import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import torchvision.transforms as transforms
from PIL import Image
import math

def show_sample(images, outputs, root_path, targets=None, enc=None, rec_model=None, box_cnt=1):
    batch = len(images)
    fig = plt.figure()
    fig.set_size_inches(4 * 6, math.ceil(batch / 4) * 10)
    for idx in range(batch):
        sub_plot = fig.add_subplot(math.ceil(batch / 4), 4, idx + 1)
        image = images[idx].permute(1, 2, 0)
        output = outputs[idx]
        target = None if targets is None else targets[idx * 2: idx * 2 + 2]
        sub_plot.imshow(image)
        if target != None:
            for box in target:
                color = 'yellow'
                box_patch = patches.Rectangle(
                    (
                        box[0] * 416 - box[2] * 213, 
                        box[1] * 416 - box[3] * 213,
                    ),
                    box[2] * 416,
                    box[3] * 416,
                    edgecolor=color, fill=False
                )
                sub_plot.add_patch(box_patch)
        top_label = output[output[:,5] > 0.5, 6:]
        top_output = output[output[:,5] > 0.5]
        top_label = top_label[top_output[:, 4].sort(descending=True)[1]]
        top_output = top_output[top_output[:, 4].sort(descending=True)[1]]
        top_label = enc.label_argmax(top_label[0])
        rec_output_bottom = rec_model.test(enc.transform_label(list(top_label), 'top'), 'top')
        for box in top_output[:box_cnt]:
            # title.append(f'top box   : {box[:5]}')
            color = 'red'
            box_patch = patches.Rectangle(
                (
                    box[0] - box[2] / 2, 
                    box[1] - box[3] / 2,
                ),
                box[2],
                box[3],
                edgecolor=color, fill=False
            )
            sub_plot.add_patch(box_patch)
        bottom_label = output[output[:,5] <= 0.5, 6:]
        bottom_output = output[output[:,5] <= 0.5]
        bottom_label = bottom_label[bottom_output[:, 4].sort(descending=True)[1]]
        bottom_output = bottom_output[bottom_output[:, 4].sort(descending=True)[1]]
        bottom_label = enc.label_argmax(bottom_label[0])
        rec_output_top = rec_model.test(enc.transform_label(list(bottom_label), 'bottom'), 'bottom')
        for box in bottom_output[:box_cnt]:
            # title.append(f'bottom box: {box[:5]}')
            color = 'blue'
            box_patch = patches.Rectangle(
                (
                    box[0] - box[2] / 2, 
                    box[1] - box[3] / 2,
                ),
                box[2],
                box[3],
                edgecolor=color, fill=False
            )
            sub_plot.add_patch(box_patch)
        sub_plot.set_xlabel('\n'.join(['Recommendation'] + rec_output_top + rec_output_bottom))
        if target != None:
            decoded_output = enc.decode_tensor(top_label, 'top') + enc.decode_tensor(bottom_label, 'bottom')
            answer = enc.decode_tensor(target[0, 5:], 'top') + enc.decode_tensor(target[1, 5:], 'bottom')
            title = [f'{answer[idx]} | {decoded_output[idx]}' for idx in range(len(decoded_output))]
            sub_plot.set_title('\n'.join(title))
    fig.tight_layout()
    plt.savefig(f'{root_path}/test_by_validset.jpg')

def test_image(root_path, image_name, model, enc, device):
    s_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416))
    ])
    r_transform = transforms.Resize((416, 416))
    image_path = f'{root_path}/data/test/{image_name}'
    image = Image.open(image_path).convert('RGB')
    image = s_transform(image)
    image = image.unsqueeze(dim=0)
    image = image.to(device)
    output, _ = model(image)
    image = Image.open(image_path).convert('RGB')
    image = r_transform(image)
    
    image = image
    plt.figure(figsize=(5, 8))
    plt.imshow(image)
    output = output[0]
    top_label = output[output[:,5] > 0.5, 6:]
    top_output = output[output[:,5] > 0.5]
    top_label = top_label[top_output[:, 4].sort(descending=True)[1]]
    top_output = top_output[top_output[:, 4].sort(descending=True)[1]]
    top_label = enc.label_argmax(top_label[0])
    for box in top_output[:1]:
        # title.append(f'top box   : {box[:5]}')
        color = 'red'
        box_patch = patches.Rectangle(
            (
                box[0] - box[2] / 2, 
                box[1] - box[3] / 2,
            ),
            box[2],
            box[3],
            edgecolor=color, fill=False
        )
        plt.gca().add_patch(box_patch)
        plt.text(box[0] - box[2] / 2, (box[1] - box[3] / 2), str(box[4])).set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
    bottom_label = output[output[:,5] <= 0.5, 6:]
    bottom_output = output[output[:,5] <= 0.5]
    bottom_label = bottom_label[bottom_output[:, 4].sort(descending=True)[1]]
    bottom_output = bottom_output[bottom_output[:, 4].sort(descending=True)[1]]
    bottom_label = enc.label_argmax(bottom_label[0])
    for box in bottom_output[:1]:
        # title.append(f'bottom box: {box[:5]}')
        color = 'blue'
        box_patch = patches.Rectangle(
            (
                box[0] - box[2] / 2, 
                box[1] - box[3] / 2,
            ),
            box[2],
            box[3],
            edgecolor=color, fill=False
        )
        plt.gca().add_patch(box_patch)
        plt.text(box[0] - box[2] / 2, (box[1] - box[3] / 2), str(box[4])).set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
    decoded_output = enc.decode_tensor(top_label, 'top') + enc.decode_tensor(bottom_label, 'bottom')
    title = [f'{decoded_output[idx]}' for idx in range(len(decoded_output))]
    plt.title('\n'.join(title))
    plt.savefig(f'{root_path}/result/result_{image_name}')