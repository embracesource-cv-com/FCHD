import numpy as np
from PIL import Image, ImageDraw
import cv2


def draw_box(img, box, color='red', thickness=4, use_normalized_coordinates=False):
    img_pil = Image.fromarray(img).convert('RGB')
    _draw(img_pil, box, color, thickness, use_normalized_coordinates)
    np.copyto(img, np.array(img_pil))


def _draw(img, box, color, thickness, use_normalized_coordinates):
    ymin, xmin, ymax, xmax = box
    draw = ImageDraw.Draw(img)
    im_width, im_height = img.size
    if use_normalized_coordinates:
        left, right, top, bottom = xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height
    else:
        left, right, top, bottom = xmin, xmax, ymin, ymax
    draw.line([(left, top), (left, bottom), (right, bottom),
              (right, top), (left, top)], width=thickness, fill=color)


def check_raw_data(data_info):
    path = data_info['img_path']
    counts = data_info['counts']
    boxes = data_info['boxes']
    img = cv2.imread(path)
    img_cp = np.copy(np.asarray(img, dtype=np.uint8))
    for i in range(counts):
        draw_box(img_cp, boxes[i])

    cv2.imshow('image', img_cp)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def check_transformed_data(img, boxes):
    img = img.transpose((1, 2, 0))
    img_cp = np.copy(np.asarray(img, dtype=np.uint8))
    for i in range(len(boxes)):
        draw_box(img_cp, boxes[i])

    cv2.imshow('image', img_cp)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
