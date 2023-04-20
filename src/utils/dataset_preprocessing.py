import os, json, sys

from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile
from one_hot_encoder import OneHotEncoder

def get_annotation_list(data_path):
    print('Checking label directory')
    
    if not os.path.exists(f'{data_path}/label'):
        print('Make label directory')
        os.mkdir(f'{data_path}/label')
    
    if not os.path.exists(f'{data_path}/라벨링데이터.zip'):
        print('Fail to load labeling data(라벨링데이터.zip)')
        return
    
    print('Read Annotation Zipfile')
    files = []
    with ZipFile(f'{data_path}/라벨링데이터.zip') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting'):
            member.filename = member.filename.encode('cp437').decode('euc-kr')
            files.append(member.filename)
    return files

class json_filter:
    def __init__(self):
        self.translate_dict = json.loads(open('translated.json', 'r', encoding='utf-8').read())
        self.top_print_filter = ['dot', 'floral', 'none-print', 'stripe', 'check', 'lettering', 'graphic']
        self.top_label_trans = {
            '색상': 'color', '서브색상': 'sub_color', '넥라인': 'neck_line', 
            '소매기장': 'sleeve', '소재': 'material', '프린트': 'print', '카테고리': 'category', 
            '핏': 'fit', '기장': 'length', '디테일': 'detail', '옷깃': 'collar'
        }
        self.bottom_label_trans = {
            '색상': 'color', '서브색상': 'sub_color', '소재': 'material', '프린트': 'print', 
            '카테고리': 'category', '핏': 'fit', '기장': 'length', '디테일': 'detail'
        }
    
    def formatter(self, json_data, raw_name):
        style_name = raw_name.split('/')[0]
        new_json = dict.fromkeys(['top_'+x for x in self.top_label_trans.values()] + ['bottom_'+x for x in self.bottom_label_trans.values()])
        new_json['top_coord'] = {}
        new_json['top_coord']['x'] = int(json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['X좌표'])
        new_json['top_coord']['y'] = int(json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['Y좌표'])
        new_json['top_coord']['width'] = int(json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['가로'])
        new_json['top_coord']['height'] = int(json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['세로'])
        new_json['bottom_coord'] = {}
        new_json['bottom_coord']['x'] = int(json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['X좌표'])
        new_json['bottom_coord']['y'] = int(json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['Y좌표'])
        new_json['bottom_coord']['width'] = int(json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['가로'])
        new_json['bottom_coord']['height'] = int(json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['세로'])
        new_json['image_id'] = json_data['이미지 정보']['이미지 식별자']
        top_label = json_data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['상의'][0]
        bottom_label = json_data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['하의'][0]
        for key in top_label:
            if type(top_label[key]) == list:
                new_json[f'top_{self.translate_dict[key]}'] = [self.translate_dict[x] for x in top_label[key]]
            else:
                new_json[f'top_{self.translate_dict[key]}'] = self.translate_dict[top_label[key]]
        for key in bottom_label:
            if type(bottom_label[key]) == list:
                new_json[f'bottom_{self.translate_dict[key]}'] = [self.translate_dict[x] for x in bottom_label[key]]
            else:
                new_json[f'bottom_{self.translate_dict[key]}'] = self.translate_dict[bottom_label[key]]
        new_json['style'] = self.translate_dict[style_name]
        return new_json
    
    def coord_filter(self, json_data):
        rect = json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']
        if rect['상의'] == [{}] or rect['하의'] == [{}]:
            return False, "No Coord"
        return True, json_data
    
    def label_filter(self, data):
        new_data = {}
        if data['top_color'] is None:
            return False, "No top_color"
        else :
            new_data['top_color'] = data['top_color']
        if data['top_sub_color'] is None:
            new_data['top_sub_color'] = data['top_color']
        else:
            new_data['top_sub_color'] = data['top_sub_color']
        if data['top_sleeve'] is None or data['top_sleeve'] == 'none':
            return False, "No top_sleeve"
        else:
            new_data['top_sleeve'] = data['top_sleeve']
        if data['top_length'] is None:
            return False, "No top_length"
        else:
            new_data['top_length'] = data['top_length']
        if data['top_category'] is None:
            return False, "No top_category"
        else:
            new_data['top_category'] = data['top_category']
        if data['top_fit'] is None:
            return False, "No top_fit"
        else:
            new_data['top_fit'] = data['top_fit']
        if data['top_print'] is None:
            return False, 'No top_print'
        for x in data['top_print']:
            if x not in self.top_print_filter:
                return False, f"top_print: {x}"
        new_data['top_print'] = data['top_print'][0]

        if data['bottom_color'] is None:
            return False, "No bottom_color"
        else:
            new_data['bottom_color'] = data['bottom_color']
        
        if data['bottom_category'] is None:
            return False, "No bottom_category"
        else:
            new_data['bottom_category'] = data['bottom_category']
        
        if data['bottom_fit'] is None:
            return False, "No bottom_fit"
        else:
            new_data['bottom_fit'] = data['bottom_fit']
        
        if data['bottom_length'] is None:
            return False, "No bottom_length"
        else:
            new_data['bottom_length'] = data['bottom_length']

        new_data['top_coord'] = data['top_coord']
        new_data['bottom_coord'] = data['bottom_coord']
        new_data['image_id'] = data['image_id']
        new_data['style'] = data['style']
        return True, new_data
    
    def filter(self, json_data, raw_name):    
        tf, json_data = self.coord_filter(json_data)
        if not tf:
            return tf, json_data
        if '기타' in raw_name:
            return False, "Style Filter"
        json_data = self.formatter(json_data, raw_name)
        tf, json_data = self.label_filter(json_data)
        if not tf:
            return tf, json_data
        return True, json_data

def filter_annotation_list(data_path, files):
    annt_filter = json_filter()
    filtered = {}
    with ZipFile(f'{data_path}/라벨링데이터.zip') as zf:
        for file in tqdm(files, desc='Filtering'):
            filename = file.encode('euc-kr').decode('cp437')
            if '.json' in filename:
                temp = zf.open(filename, 'r')
                fileid = filename[filename.find('/') + 1:-5]
                annt = json.loads(temp.read())
                tf, new_data = annt_filter.filter(annt, file)
                if tf:
                    filtered[fileid] = new_data
    return filtered

def resize_img(file_id, image, annt):
    merged_left = min(annt['top_coord']['x'], annt['bottom_coord']['x'])
    merged_right = max(annt['top_coord']['x'] + annt['top_coord']['width'], annt['bottom_coord']['x'] + annt['bottom_coord']['width'])
    merged_top = min(annt['top_coord']['y'], annt['bottom_coord']['y'])
    merged_bottom = max(annt['top_coord']['y'] + annt['top_coord']['height'], annt['bottom_coord']['y'] + annt['bottom_coord']['height'])

    padding = round(min(merged_right - merged_left, merged_bottom - merged_top) * 0.1)

    padded_left = merged_left - padding
    padded_right = merged_right + padding
    padded_top = merged_top - padding
    padded_bottom = merged_bottom + padding

    squared_size = max(padded_right - padded_left, padded_bottom - padded_top)

    x_offset = (squared_size - (padded_right - padded_left)) / 2
    y_offset = (squared_size - (padded_bottom - padded_top)) / 2

    top_box_left = annt['top_coord']['x'] - padded_left + x_offset
    top_box_top = annt['top_coord']['y'] - padded_top + y_offset
    top_box_width = annt['top_coord']['width']
    top_box_width_ratio = top_box_width / squared_size
    top_box_height = annt['top_coord']['height']
    top_box_height_ratio = top_box_height / squared_size
    top_box_center = (
        (top_box_left + top_box_width / 2) / squared_size, 
        (top_box_top + top_box_height / 2) / squared_size
    )

    bottom_box_left = annt['bottom_coord']['x'] - padded_left + x_offset
    bottom_box_top = annt['bottom_coord']['y'] - padded_top + y_offset
    bottom_box_width = annt['bottom_coord']['width']
    bottom_box_width_ratio = bottom_box_width / squared_size
    bottom_box_height = annt['bottom_coord']['height']
    bottom_box_height_ratio = bottom_box_height / squared_size
    bottom_box_center = (
        (bottom_box_left + bottom_box_width / 2) / squared_size, 
        (bottom_box_top + bottom_box_height / 2) / squared_size
    )
    try:
        crop_image = image.crop((
            padded_left - x_offset, 
            padded_top - y_offset, 
            padded_left - x_offset + squared_size,  
            padded_top - y_offset + squared_size
        ))
    except:
        print(file_id)
        print('Raw Size', image.size)
        print('Padded Coord', padded_left, padded_right, padded_top, padded_bottom)
        print('Padding and Size', padding, squared_size)
        print('Offset', x_offset, y_offset)
        print('Top Box', top_box_left, top_box_top, top_box_width, top_box_height)
        print('Bottom Box', bottom_box_left, bottom_box_top, bottom_box_width, bottom_box_height)
        sys.exit(1)

    resize_image = crop_image.resize((416, 416), Image.LANCZOS)

    new_data = {
        'top_coord': {
            'x': top_box_center[0],
            'y': top_box_center[1],
            'width': top_box_width_ratio,
            'height': top_box_height_ratio,
        },
        'bottom_coord': {
            'x': bottom_box_center[0],
            'y': bottom_box_center[1],
            'width': bottom_box_width_ratio,
            'height': bottom_box_height_ratio,
        },
    }

    return resize_image, new_data

def resize_images(data_path, files):
    if not os.path.exists(f'{data_path}/image'):
        print('Make image directory')
        os.mkdir(f'{data_path}/image')
    enc = OneHotEncoder(data_path + '/label')
    enc.load_encoder()
    print(len(files))
    cnt = 0
    
    for zip in os.listdir(data_path):
        if '원천데이터' not in zip:
            continue
        with ZipFile(f'{data_path}/{zip}') as zf:
            infos = zf.infolist()
            for member in tqdm(infos, desc=f'Data({zip}) Extraction'):
                name = member.filename.encode('cp437').decode('euc-kr')
                file_id = name[name.find('/') + 1:name.find('.')]
                if file_id == '':
                    continue
                if file_id in files:
                    cnt += 1
                    member.filename = name
                    temp = zf.open(member)
                    img = Image.open(temp).convert('RGB')
                    annt = files[file_id]
                    if 'top_label' in annt:
                        continue
                    new_data = {}
                    img, coord = resize_img(file_id, img, annt)
                    new_data.update(enc.encode(annt))
                    new_data.update(coord)
                    with open(f'{data_path}/label/{file_id}.json', 'w') as f:
                        json.dump(new_data, f, indent=2)
                    img.save(f'{data_path}/image/{file_id}.jpg', 'JPEG')
    print(cnt)

if __name__ == '__main__':
    train_data_path = 'data/train'
    valid_data_path = 'data/valid'
    files = get_annotation_list(valid_data_path)
    files = filter_annotation_list(valid_data_path, files)
    resize_images(valid_data_path, files)
    #training_data_path = input('Enter training dataset path: ')
    #extract_annotation(training_data_path)
    #clear_raw_annotation(training_data_path)
    #edited_annotation(training_data_path)
    #resize_images(training_data_path)