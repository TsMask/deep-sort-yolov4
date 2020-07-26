import os
import argparse
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
import tensorflow as tf

from yolo4.model import yolo_eval, yolo4_body

# 执行参数 python convertToH5.py --input_size 608 --min_score 0.3 --iou 0.5 --model_path model_data/yolov4.h5 --weights_path model_data/yolov4.weights
# 外部参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, default=608, help='Image input size 320 416 512 608.')
parser.add_argument('--min_score', type=float, default=0.3, help='minimum output score.')
parser.add_argument('--iou', type=float, default=0.5, help='target threshold.')
parser.add_argument('--model_path', type=str, default='model_data/yolov4.h5', help='model save type.')
parser.add_argument('--weights_path', type=str, default='model_data/yolov4.weights', help='weight file.')
ARGS = parser.parse_args()

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

class Yolo4(object):

    def __init__(self, score, iou, model_path, weights_path,input_size, gpu_num=1):
        self.score = score
        self.input_size = input_size
        self.weights_path = weights_path
        self.model_path = model_path
        self.iou = iou
        self.gpu_num = gpu_num
        self.load_yolo()

    # 加载权重
    def load_weights(self,model, weights_file):
        wf = open(weights_file, 'rb')
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(110):
            conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
            bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in [93, 101, 109]:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in [93, 101, 109]:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'
        wf.close()

    # 保存为h5模型
    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = CLASSES
        self.anchors = np.array(anchors).reshape(-1, 2)

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        self.sess = tf.compat.v1.Session()

        # Load model, or construct model and load weights.
        self.yolo4_model = yolo4_body(Input(shape=(self.input_size, self.input_size, 3)), num_anchors//3, num_classes)

        # Read and convert darknet weight
        self.load_weights(self.yolo4_model, self.weights_path)

        self.yolo4_model.save(self.model_path)

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(
            self.yolo4_model.output, 
            self.anchors, 
            len(self.class_names), 
            self.input_image_shape, 
            score_threshold=self.score
        )
        print('Dome.')

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':

    yolo4_model = Yolo4(ARGS.min_score, ARGS.iou, ARGS.model_path, ARGS.weights_path, ARGS.input_size)

    yolo4_model.close_session()
