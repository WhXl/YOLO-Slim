#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image


return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./hw_weight=0.7513/yolov3_coco.pb"
#video_path      = "/home/ser2/Projects/Mobilenet--yolov3/docs/AC123103.avi"
video_path      = 0
num_classes     = 20
input_size      = 416
graph           = tf.Graph()
#graph           = tf.import_graph_def(graph, name='')
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

cv2.namedWindow("result", cv2.WINDOW_NORMAL)

with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(video_path)
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
    #
    # vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

    tm = []
    while True:
        try:
            return_value, frame = vid.read()
        except IOError:
            raise ValueError("IOError")
        if return_value:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.5)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        exec_time1 = exec_time * 1000
        tm.append(exec_time1)

        pre =np.floor(1000 / np.mean(tm[1:]))
        print("fps is:%s"%(pre))
        result = np.asarray(image)
        cv2.putText(result, '{:.2f}fps'.format(pre), (40, 40), 0,
                                 fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break



