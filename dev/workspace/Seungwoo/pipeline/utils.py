import math
import heapq
import numpy as np
import pandas as pd
import cv2
from copy import copy
import matplotlib.pyplot as plt
import json
import requests
def pre_predict(reader, img, debug=False):
    url = "http://sanghyun.ddns.net:8000/items/"
    req = {"instances": img.tolist()}  # [H, W, C]

    response = requests.post(url=url, data=json.dumps(req))
    results = reader.readtext(img)
    res_eval = eval(response.text)
    portion = list(map(dgr2pct, res_eval["dgr"]))
    keypoint = [
        [x_center, y_center, x_left, y_left, x_right, y_right]
        for [[x_center, y_center], [x_left, y_left], [x_right, y_right], _] in res_eval[
            "grp"
        ]
    ]
    return conclude(img, results, keypoint, portion, debug=debug)

def predict(reader, img, debug=False):
    url = "http://sanghyun.ddns.net:8000/items/"
    req = {"instances": img.tolist()}  # [H, W, C]

    response = requests.post(url=url, data=json.dumps(req))
    results = reader.readtext(img, text_threshold = 0.3, paragraph = True, y_ths = 0.05, width_ths =0.3)
    res_eval = eval(response.text)
    portion = list(map(dgr2pct, res_eval["dgr"]))
    keypoint = [
        [x_center, y_center, x_left, y_left, x_right, y_right]
        for [[x_center, y_center], [x_left, y_left], [x_right, y_right], _] in res_eval[
            "grp"
        ]
    ]
    return conclude(img, results, keypoint, portion, debug=debug)


def conclude(image, results, keypoint, portion, threshold=0, debug=False):
    legend, legend_points, nopie = checklegend(image, keypoint, debug=debug)
    if debug:
        print(f"범례 유무 : {legend}")
        print("Legend Points")
        print(legend_points)
        print()
        print("OCR 결과")
        print(results)
        print()
        plt.imshow(nopie)
    ocr = []
    for r in results: #(좌표, 값)
        if (set(str(r[1]))-set(['0','1','2','3','4','5','6','7','8','9','.',' ', '%']) == set()):
            print('number: ',r[1])
            continue
        print(r[1])


        ocr.append([get_left_center(r[0]), r[1]])

    # 범례 있는 차트
    if legend:
        pq = []
        for text in ocr:
            for point_num, points in enumerate(legend_points):
                dist = get_distance(text[0][0], text[0][1], points[0], points[1])
                heapq.heappush(pq, [dist, text[1], point_num])

        info = [None for i in range(len(portion))]

        while pq:
            cur = heapq.heappop(pq)
            if info[cur[2]] is not None:
                continue
            info[cur[2]] = cur[1]

        return pd.DataFrame(portion, info)
    else:
        return match(results, keypoint, portion, threshold=threshold)


def checklegend(image, keypoint, debug=False):
    legend = False
    image = np.array(image)
    copy_image = copy(image)

    gray = cv2.cvtColor(copy_image, cv2.COLOR_BGR2GRAY)
    detected_circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=100,
        param2=80,
        minRadius=40,
        maxRadius=350,
    )

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(copy_image, (a, b), r + 10, (255, 255, 255), -1)
    else:
        print("No pie detected")
        # 예외처리 필요?
    lst = []
    for pt in keypoint:
        COG = (int((pt[0] + pt[2] + pt[4]) / 3), int((pt[1] + pt[3] + pt[5]) / 3))
        color = copy(image[COG[1], COG[0], :])

        mask_image = copy(copy_image)
        mask = cv2.inRange(mask_image, color, color)
        colormap = cv2.bitwise_and(mask_image, mask_image, mask=mask)
        pixels = cv2.countNonZero(mask)
        x = np.nonzero(colormap)[1].mean()
        y = np.nonzero(colormap)[0].mean()

        if np.isnan(x) or np.isnan(y):
            # TODO : 예외처리
            lst.append(get_arc_center(pt))
        else:
            legend = True
            lst.append((x, y))
    return legend, lst, copy_image


def integrate(boxes):
    """
    박스 가까운 애들끼리 합치기
    """
    return boxes


def get_center(boxes):
    boxes = np.array(boxes)
    return boxes.mean(axis=0).tolist()


def get_left_center(boxes):
    boxes = [boxes[0], boxes[-1]]
    boxes = np.array(boxes)
    return boxes.mean(axis=0).tolist()


def match(ocr, keypoint, portion, threshold=0.5):
    labels = []
    for text in ocr:
        if (set(str(text[1]))-set(['0','1','2','3','4','5','6','7','8','9','.',' ', '%']) == set()):
            continue


        labels.append([get_center(text[0]), text[1]])

    ## Check
    assert len(keypoint) == len(portion)

    pq = []
    for label in labels:
        for point_num, points in enumerate(keypoint):
            COA = get_arc_center(points)
            dist = get_distance(label[0][0], label[0][1], COA[0], COA[1])
            heapq.heappush(pq, [dist, label[1], point_num])

    legend = [None for i in range(len(portion))]

    while pq:
        cur = heapq.heappop(pq)
        if legend[cur[2]] is not None:
            continue
        legend[cur[2]] = cur[1]

    return pd.DataFrame(portion, legend)


def get_distance(x1, y1, x2, y2):
    return (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5


def get_arc_center(points):
    x_center, y_center, x_left, y_left, x_right, y_right = points
    assert (
        abs(
            get_distance(x_left, y_left, x_center, y_center)
            - get_distance(x_right, y_right, x_center, y_center)
        )
        < 5
    )

    radius = get_distance(x_left, y_left, x_center, y_center)
    norm_x_left, norm_x_right = (
        (x_left - x_center) / radius,
        (x_right - x_center) / radius,
    )

    # y가 역방향임
    norm_y_left, norm_y_right = (
        (y_center - y_left) / radius,
        (y_center - y_right) / radius,
    )

    angle = math.atan2(
        (norm_y_left + norm_y_right) / 2, (norm_x_left + norm_x_right) / 2
    )
    coord = (math.cos(angle) * radius + x_center, -math.sin(angle) * radius + y_center)

    # print((norm_y1 + norm_y2)/2, (norm_x1 + norm_x2)/2)
    # print(angle)
    return list(map(int, coord))


def dgr2pct(dgr):
    return round(dgr / 360 * 100, 1)