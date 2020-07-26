# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v4 style detection model on image and video
"""

import os, colorsys, random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from keras import backend as K
from keras.models import load_model
from keras.utils import multi_gpu_model

from yolo4.model import yolo_eval, Mish

# 数据集识别80类别
CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck',
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
            29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
            48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'sofa', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
            62: 'tvmonitor', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
            68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

anchors = np.array([12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401], dtype=np.float)

class YOLO4(object):
    def __init__(self,model_path, score):
        self.model_path = model_path
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.gpu_num = 1
        self.score = score
        self.iou = 0.5
        self.class_names = CLASSES 
        self.anchors = np.array(anchors).reshape(-1, 2)
        self.model_image_size = (608, 608)  # 根据H5模型固定大小
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes, self.colors = self._generate()

    # 随机颜色
    def _random_colors(self,N):
        hsv_tuples = [(1.0 * x / N, 1., 1.) for x in range(N)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(100)
        random.shuffle(colors)
        random.seed(None)
        return colors

    # 图像尺寸调整
    def _letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    # 生成默认
    def _generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.yolo_model = load_model(model_path, custom_objects={'Mish': Mish}, compile=False)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        colors = self._random_colors(len(CLASSES))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, len(self.class_names), self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes, colors

    # 识别预测图片
    def detect_image(self, image):
        # 图像调整
        boxed_image = self._letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype=np.float)
        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        # 识别
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
            
        # 数据段值
        return_boxes = []
        return_scores = []
        return_class_names = []
        return_class_color = []
        for i, class_id in enumerate(out_classes):
            y1, x1, y2, x2 = np.array(out_boxes[i], dtype=np.int32)
            return_boxes.append([x1, y1, (x2 - x1), (y2 - y1)])
            return_scores.append(out_scores[i])
            return_class_names.append(self.class_names[class_id])
            return_class_color.append(self.colors[class_id])

        return return_boxes, return_scores, return_class_names, return_class_color

    def close_session(self):
        self.sess.close()
