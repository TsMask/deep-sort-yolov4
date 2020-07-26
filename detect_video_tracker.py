#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2
import argparse
import numpy as np
from PIL import Image
from yolo4.yolo import YOLO4

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections

# 执行参数 python detect_video_tracker.py --video test.mp4 --min_score 0.3 --model_yolo model_data/yolov4.h5 --model_feature model_data/mars-small128.pb
# 外部参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='test.mp4', help='data mp4 file.')
parser.add_argument('--min_score', type=float, default=0.3, help='displays the lowest tracking score.')
parser.add_argument('--model_yolo', type=str, default='model_data/yolo4.h5', help='Object detection model file.')
parser.add_argument('--model_feature', type=str, default='model_data/market1501.pb', help='target tracking model file.')
ARGS = parser.parse_args()

box_size = 2        # 边框大小
font_scale = 0.4    # 字体比例大小

if __name__ == '__main__':
    # Deep SORT 跟踪器
    encoder = generate_detections.create_box_encoder(ARGS.model_feature, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", ARGS.min_score, None)
    tracker = Tracker(metric)

    # 载入模型
    yolo = YOLO4(ARGS.model_yolo, ARGS.min_score)

    # 读取视频
    video = cv2.VideoCapture(ARGS.video)

    # 输出保存视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_out = cv2.VideoWriter("outputVideo.mp4", fourcc, fps, size)

    # 视频是否可以打开，进行逐帧识别绘制
    while video.isOpened:
        # 视频读取图片帧
        retval, frame = video.read()
        if retval != True:
            # 任务完成后释放所有内容
            video.release()
            video_out.release()
            cv2.destroyAllWindows()
            print("没有图像！尝试使用其他视频")
            break

        prev_time = time.time()

        # 图片转换识别
        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, scores, classes, colors = yolo.detect_image(image)

        # 特征提取和检测对象列表
        features = encoder(frame, boxes)
        detections = []
        for bbox, score, classe, color, feature in zip(boxes, scores, classes, colors, features):
            detections.append(Detection(bbox, score, classe, color, feature))

        # 运行非最大值抑制
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.score for d in detections])
        indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
        detections = [detections[i] for i in indices]

        # 追踪器刷新
        tracker.predict()
        tracker.update(detections)

        # 遍历绘制跟踪信息
        track_count = 0
        track_total = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: continue
            y1, x1, y2, x2 = np.array(track.to_tlbr(), dtype=np.int32)
            # cv2.rectangle(frame, (y1, x1), (y2, x2), (255, 255, 255), box_size//4)
            cv2.putText(
                frame, 
                "No. " + str(track.track_id),
                (y1, x1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, 
                (255, 255, 255),
                box_size//2,
                lineType=cv2.LINE_AA
            )
            if track.track_id > track_total: track_total = track.track_id
            track_count += 1

        # 遍历绘制检测对象信息
        totalCount = {}
        for det in detections:
            y1, x1, y2, x2 = np.array(det.to_tlbr(), dtype=np.int32)
            caption = '{} {:.2f}'.format(det.classe, det.score) if det.classe else det.score
            cv2.rectangle(frame, (y1, x1), (y2, x2), det.color, box_size)
            # 填充文字区
            text_size = cv2.getTextSize(caption, 0, font_scale, thickness=box_size)[0]
            cv2.rectangle(frame, (y1, x1), (y1 + text_size[0], x1 + text_size[1] + 8), det.color, -1)
            cv2.putText(
                frame,
                caption,
                (y1, x1 + text_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (50, 50, 50),
                box_size//2,
                lineType=cv2.LINE_AA
            )
            # 统计物体数
            if det.classe not in totalCount: totalCount[det.classe] = 0
            totalCount[det.classe] += 1

        # 跟踪统计
        trackTotalStr = 'Track Total: %s' % str(track_total)
        cv2.putText(frame, trackTotalStr, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 0, 255), 1, cv2.LINE_AA)

        # 跟踪数量
        trackCountStr = 'Track Count: %s' % str(track_count)
        cv2.putText(frame, trackCountStr, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)

        # 识别类数统计
        totalStr = ""
        for k in totalCount.keys(): totalStr += '%s: %d    ' % (k, totalCount[k])
        cv2.putText(frame, totalStr, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)

        # 绘制时间
        curr_time = time.time()
        exec_time = curr_time - prev_time
        print("识别耗时: %.2f ms" %(1000*exec_time))

        # 视频输出保存
        video_out.write(frame)
        # 绘制视频显示窗 命令行执行屏蔽呀
        # cv2.namedWindow("video_reult", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("video_reult", frame)
        # 退出窗口
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # 任务完成后释放所有内容
    video.release()
    video_out.release()
    cv2.destroyAllWindows()
