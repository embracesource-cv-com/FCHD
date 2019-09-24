from torch.utils.data import Dataset
import numpy as np
import cv2
import re
import os


class HeadDataset(Dataset):
    def __init__(self, dataset_dir, annots_path, transform):
        self.dataset_dir = dataset_dir
        self.data_list = self.parser(annots_path)
        self.transform = transform

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        img_path = data_info['img_path']
        boxes = data_info['boxes']
        img = cv2.imread(img_path)
        img = img.transpose((2, 0, 1))
        sample = self.transform({'img': img, 'boxes': boxes})
        return sample

    def parser(self, annots_path):
        data_list = []
        with open(annots_path, 'r') as fp:
            for line in fp.readlines():
                if ':' in line:
                    img_name = re.search(r'"(.*)"', line).group(1)
                    img_path = os.path.join(self.dataset_dir, img_name)
                    coords_list = [float(i) for i in re.findall(r'\d+\.\d+', line)]
                    assert len(coords_list) % 4 == 0
                    counts = len(coords_list) // 4
                    boxes = np.array(coords_list).reshape(-1, 4)
                    boxes = boxes[:, [1, 0, 3, 2]]  # x1,y1,x2,y2 -> y1,x2,y2,x2
                    data_list.append({'img_path': img_path, 'counts': counts, 'boxes': boxes})
        return data_list

    def __len__(self):
        return len(self.data_list)
