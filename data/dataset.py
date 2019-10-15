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
        img = img.transpose((2, 0, 1))  # (H,W,C) -> (C,H,W)
        sample = self.transform({'img': img, 'boxes': boxes})
        return sample

    def parser(self, annots_path):
        """
        Parsing the annotation file
        :param annots_path: str, the absolute path of the annotation file
        :return: list of dict
        """
        data_list = []

        with open(annots_path, 'r') as fp:
            for line in fp.readlines():
                if ':' in line:
                    # extract image name
                    img_name = re.search(r'"(.*)"', line).group(1)
                    img_path = os.path.join(self.dataset_dir, img_name)

                    # extract coordinates
                    coords_list = re.findall(r'\d+\.\d+', line)
                    coords_list = list(map(float, coords_list))
                    assert len(coords_list) % 4 == 0, 'The number of coordinates must be divisible by 4.'

                    # convert to numpy array
                    counts = len(coords_list) // 4
                    boxes = np.array(coords_list).reshape(counts, 4)
                    boxes = boxes[:, [1, 0, 3, 2]]  # x1,y1,x2,y2 -> y1,x1,y2,x2

                    data_list.append({'img_path': img_path, 'counts': counts, 'boxes': boxes})

        return data_list

    def __len__(self):
        return len(self.data_list)
