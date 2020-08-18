# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time

from pathlib import Path
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

import os

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
cyan = (255, 255, 0)
magenta = (255, 0, 255)


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def removeFaceAra(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rect = detect(gray, cascade)

    return rect


def findMaxArea(contours):
    max_contour = None
    max_area = -1

    for contour in contours:
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        # if (w * h) * 0.4 > area:
        #     continue
        #
        # if w > h:
        #     continue

        if area > max_area:
            max_area = area
            max_contour = contour

    # if max_area < 10000:
    #     max_area = -1

    return max_area, max_contour


def distanceBetweenTwoPoints(start, end):
    x1, y1 = start
    x2, y2 = end

    return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))


def calculateAngle(A, B):
    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    C = np.dot(A, B)

    angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
    return angle


def getFingerPosition(max_contour, img_result, debug):
    points1 = []

    # STEP 6-1
    M = cv2.moments(max_contour)

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    max_contour = cv2.approxPolyDP(max_contour, 0.02 * cv2.arcLength(max_contour, True), True)
    hull = cv2.convexHull(max_contour)

    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0]))

    if debug:
        cv2.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
        for point in points1:
            cv2.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

    # STEP 6-2
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)

    if defects is None:
        return -1, None

    points2 = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

        if angle < 90:
            if start[1] < cy:
                points2.append(start)

            if end[1] < cy:
                points2.append(end)

    if debug:
        cv2.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
        for point in points2:
            cv2.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

    # STEP 6-3
    points = points1 + points2
    points = list(set(points))

    # STEP 6-4
    new_points = []
    for p0 in points:

        i = -1
        for index, c0 in enumerate(max_contour):
            c0 = tuple(c0[0])

            if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                i = index
                break

        if i >= 0:
            pre = i - 1
            if pre < 0:
                pre = max_contour[len(max_contour) - 1][0]
            else:
                pre = max_contour[i - 1][0]

            next = i + 1
            if next > len(max_contour) - 1:
                next = max_contour[0][0]
            else:
                next = max_contour[i + 1][0]

            if isinstance(pre, np.ndarray):
                pre = tuple(pre.tolist())
            if isinstance(next, np.ndarray):
                next = tuple(next.tolist())

            angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

            if angle < 90:
                new_points.append(p0)

    return 1, new_points


def process(img_bgr, img_binary, debug):
    img_result = img_bgr.copy()

    # # STEP 1
    # img_bgr = removeFaceAra(img_bgr, cascade)

    # # STEP 2
    # img_binary = make_mask_image(img_bgr)

    # # STEP 3
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, 1)
    # if debug:
    #   cv2.imshow("Binary", img_binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, 1)
    cv2.imshow('subImageMaskOnlyHand', img_binary)

    # STEP 4
    # _, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _,contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if debug:
        for cnt in contours:
            cv2.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)

    # STEP 5
    max_area, max_contour = findMaxArea(contours)

    if max_area == -1:
        # print("return only img!")
        return img_result, None, -1

    if debug:
        cv2.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)

    # STEP 6
    ret, points = getFingerPosition(max_contour, img_result, debug)

    # STEP 7
    if ret > 0 and len(points) > 0:
        for point in points:
            cv2.circle(img_result, point, 20, magenta, 5)

    return img_result, points, ret


def GestureMapping(coordinateList, videoName):

    # f = open("E:/oldPerson/Experiment/YOLOv3_TensorFlow/YOLOv3_TensorFlow_2/ResultTextFile/" + videoName.split('.')[0] + ".txt",'a')
    f = open("./ResultFile/TextFile/" + videoName.split('.')[0] + ".txt",'a')

    f.write('='*1000 + '\n')
    f.write("<CoordinateList>\n")
    f.write('[')
    for coordinate in coordinateList:
        f.write('(' +str(coordinate[0]) + ',' + str(coordinate[1]) + ')')
    f.write(']\n')

    if len(coordinateList) < 10:  # 좌표가 10개보다 없으면 그냥 메칭되는 제스처가 없다고 판단
        f.write("No Matching Gesture\n")
        f.close()
        return "No Matching Gesture"

    # 1단계: 매 프레임별 delta x, delta y구하기 (앞에것 - 뒤에것)

    deltaXList = []
    deltaYList = []

    for i in range(0, len(coordinateList) - 1):  # (앞시간 - 뒷 시간)으로 증분을 함 => 부호에 따른 의사결정에 영향을 준다!
        deltaXList.append(coordinateList[i][0] - coordinateList[i + 1][0])
        deltaYList.append(coordinateList[i][1] - coordinateList[i + 1][1])

    print(deltaXList)
    print(deltaYList)

    toalDeltaX, totalDeltaY = 0, 0

    for i in range(0, len(deltaYList)):
        toalDeltaX = toalDeltaX + deltaXList[i]
        totalDeltaY = totalDeltaY + deltaYList[i]

    print(toalDeltaX)
    print(totalDeltaY)
    f.write('toalDeltaX: '+ str(toalDeltaX) + '\n')
    f.write('totalDeltaY: ' + str(totalDeltaY) + '\n')

    if abs(toalDeltaX) > abs(totalDeltaY):  # 좌우이동 => x을 보면 된다.
        if toalDeltaX > 0:
            f.write("Previous Page!\n")
            f.close()
            return "Previous Page!"
        else:
            f.write("Next Page!\n")
            f.close()
            return "Next Page!"
    else:  # 상하 이동
        if totalDeltaY > 0:
            f.write("Scroll Up!\n")
            f.close()
            return "Scroll Up!"
        else:
            f.write("Scroll down!\n")
            f.close()
            return "Scroll down!"

    # #1단계 : x값 변화율과 y값 변화율 구하기 => 쉽게 평균변화량으로 구해보자
    # startPoint = coordinateList[0]
    # endPoint = coordinateList[len(coordinateList)-1]
    #
    # if abs(startPoint[0] - endPoint[0]) > abs(startPoint[1] - endPoint[1]) : #좌우이동 => x을 보면 된다.
    #     if startPoint[0] > endPoint[0]: #왼쪽으로 이동
    #         return "Next Page!"
    #     else:
    #         return "Previous Page!"
    # else:   #상하 이동
    #     if startPoint[1] > endPoint[1]: #위로 이동
    #         return "Scroll Up!"
    #     else:
    #         return "Scroll Down!"


fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=200, detectShadows=0)
# -----------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
parser.add_argument("input_video", type=str,
                    help="The path of the input video.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to save the video detection results.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

vid = cv2.VideoCapture(args.input_video)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))
args.save_video = True

videoName = args.input_video.split('/')[-1]
print(videoName)

if args.save_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # videoWriter = cv2.VideoWriter('C:/Users/super/Desktop/YOLOv3_TensorFlow_2/TestResult/'+ videoName + '.mp4', fourcc, video_fps,
    #                               (video_width, video_height))
    videoWriter = cv2.VideoWriter("./ResultFile/VideoFile/" + videoName.split('.')[0] + '.mp4', fourcc, video_fps,
                                  (video_width, video_height))


with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3,
                                    nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    pmode = 'start'  # 모드에는 start(시작) / oneHand(그리기-손가락 하나) / fiveHand 3가지가 있음
    coordinateList = []
    gestureStr = 'start test'
    for i in range(video_frame_cnt):  # 매 프레임 마다
        ret, img_ori = vid.read()
        frame = img_ori  # 배경제거 알고리즘으로 들어갈 프레임을 여기서 미리 할당
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        fgmask = fgbg.apply(blur, learningRate=0)
        # fgmask = fgbg.apply(blur, learningRate=-1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, 2)
        cv2.imshow('mask', fgmask)
        # 배경제거 알고리즘을 통한 이진화  이미지 얻기

        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        start_time = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        end_time = time.time()

        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori / float(args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori / float(args.new_size[1]))

        h, w, temp = img_ori.shape

        if len(scores_) >= 1:  # 뭔가 손이라고 하나 이상 발견이 되었을 때
            maxScore = scores_[0]
            maxScorePos = 0
            for j in range(0, len(scores_)):
                if maxScore < scores_[j]:
                    maxScore = scores_[j]
                    maxScorePos = j
            # maxScorePos => 한 프레임에 대해 가장 손일 가능성이 큰 box영역 인덱스를 유일하게 결정

            if maxScore > 0.1:
                x0, y0, x1, y1 = boxes_[maxScorePos]
                # if x0 - w / 10 > 0:
                #     x0 = x0 - w / 10
                if y0 - h / 7 > 0:
                    y0 = y0 - h / 7
                # if x1 + w / 10 < w:
                #     x1 = x1 + w / 10
                # 아래쪽으로 바운딩 박스를 확장할 필요는 없다. => 어차피 손가락은 위에 있기 때문이다.
                # if y1 + h / 8 < h:
                #     y1 = y1 + h / 8

                # print("hand Detection: " + str(maxScore))
                plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[maxScorePos]],
                             color=color_table[labels_[maxScorePos]])
                # 여기까지가 yolo로 손 검출

                # 손 검출이 되었을 때만 여기서부터 이진화 이미지를 손 box 영역만큼 잘라서 손가락 인식해야 한다.
                sub_fgmask = fgmask[int(y0):int(y1), int(x0):int(x1)]

                subimg_finger_result, points, ret = process(frame[int(y0):int(y1), int(x0):int(x1)], sub_fgmask,
                                                            debug=False)
                cv2.imshow('subImageMaskFinger', subimg_finger_result)

                # subimage에서 얻은 손가락 점을 원래 이미지 좌표로 변환해야함
                finalPoints = []
                if ret > 0 and len(points) > 0:
                    for point in points:
                        x = int(x0) + point[0]
                        y = int(y0) + point[1]
                        finalPoints.append((x, y))
                        cv2.circle(img_ori, (x, y), 20, magenta, 5)
                # 여기까지 영상에서 손가락 추출 완료

                if len(finalPoints) >= 3:  # 5손가락일 때에는 누적좌표 확인 => 제스처 메핑 => 누적 좌표 비우기
                    print(coordinateList)
                    if pmode is 'oneHand' and len(coordinateList) >= 10:  # 누적좌표가 10개이상 있다면
                        gestureStr = GestureMapping(coordinateList,videoName)
                        print(gestureStr)
                    pmode = 'fiveHand'
                    coordinateList = []


                elif len(finalPoints) >= 1:  # 1손가락일 때에는 손가락들중 가장 높은 위치에 있는 손가락 하나를 최종 1개 손가락 점으로 인식 => 그 손가락 위치 누적
                    # print("one")
                    if pmode is not 'start':
                        gestureStr = 'Drawing Gesture'
                        maxHeighPoint = finalPoints[0]
                        maxHeigh = maxHeighPoint[1]
                        for finalPoint in finalPoints:
                            if maxHeigh > finalPoint[1]:  # y좌표가 더 큰 손가락이 있다면
                                maxHeighPoint = finalPoint
                                maxHeigh = maxHeighPoint[1]
                        # maxHeighPoint가 최종 1개 손가락이 됨
                        # print("maxHeighPoint!")
                        cv2.circle(img_ori, maxHeighPoint, 20, white, 5)  # 최종적으로 인식되는 손가락 좌표
                        coordinateList.append([maxHeighPoint[0], maxHeighPoint[1]])
                        pmode = 'oneHand'

        # 누적좌표가 있을 때에는 그 누적좌표들을 연결한 선을 그려주자
        if len(coordinateList) >= 2:
            # print("Drawing!")
            pts = np.array(coordinateList, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # print(pts)
            cv2.polylines(img_ori, [pts], False, blue, 2)

        cv2.putText(img_ori,
                    '{:.2f}ms'.format((end_time - start_time) * 1000) + '           Gesture Info: ' + gestureStr,
                    (40, 40), 0, fontScale=1,
                    color=green, thickness=2)
        cv2.imshow('result', img_ori)
        if True:
            videoWriter.write(img_ori)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    if True:
        videoWriter.release()
