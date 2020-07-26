# Deep SORT —— YOLO V4 目标检测跟踪

## 介绍

项目采用 `YOLO V4` 算法模型进行目标检测，使用 `Deep SORT` 目标跟踪算法。

**运行环境**

- Keras==2.4.3
- tensorflow-gpu==2.2.0
- opencv-python==4.3.0.36
- image==1.5.32
- NVIDIA GPU CUDA

## 目录结构

```text
deep-sort-yolov4

┌── deep_sort                        DeepSort目标跟踪算法
│   ├── detection.py
│   ├── generate_detections.py
│   ├── iou_matching.py
│   ├── kalman_filter.py
│   ├── linear_assignment.py
│   ├── nn_matching.py
│   ├── preprocessing.py
│   ├── track.py
│   └── tracker.py
├── model_data                       模型文件数据
│   ├── market1501.pb
│   ├── mars-small128.pb
│   ├── yolov4.h5
│   ├── yolov4.weights
│   └── README.md
├── yolo4                            YOLOV4目标检测
│   ├── model.py
│   └── yolo.py
│─── convertToH5.py
│─── detect_video_tracker.py
│─── requirements.txt
│─── test.mp4
└─── README.md
```

## 执行

模型的权重文件要先转出模型H5文件哦

```shell
# 安装依赖
pip install -r requirements.txt

# 模型权重 `yolov4.weights` 转 `yolo4.h5`
python convertToH5.py --input_size 608 --min_score 0.3 --iou 0.5 --model_path model_data/yolov4.h5 --weights_path model_data/yolov4.weights

# 执行视频目标检测跟踪
python detect_video_tracker.py --video test.mp4 --min_score 0.3 --model_yolo model_data/yolov4.h5 --model_feature model_data/mars-small128.pb
```

## 推荐

- [Object-Detection-and-Tracking](https://github.com/yehengchen/Object-Detection-and-Tracking)
- [Deep Sort](https://github.com/nwojke/deep_sort)
- [关于 Deep Sort 的一些理解](https://zhuanlan.zhihu.com/p/80764724)
