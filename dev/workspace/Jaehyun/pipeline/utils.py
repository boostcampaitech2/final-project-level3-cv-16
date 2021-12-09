import math
import heapq
import numpy as np
import pandas as pd

def integrate(boxes):
    """
    박스 가까운 애들끼리 합치기
    """
    return boxes

def get_center(boxes):
    boxes = np.array(boxes)
    return boxes.mean(axis=0).tolist()

def match(ocr, keypoint, portion, threshold=0.5):
    labels = []
    for text in ocr:
        if '%' in text[1]:
            continue
        if text[2] < threshold:
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
    return ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)

def get_arc_center(points):
    x_center, y_center, x_left, y_left, x_right, y_right = points
    assert abs(get_distance(x_left,y_left, x_center, y_center) - get_distance(x_right, y_right, x_center, y_center)) < 5

    radius = get_distance(x_left,y_left, x_center, y_center)
    norm_x_left, norm_x_right = (x_left - x_center)/radius, (x_right - x_center)/radius

    # y가 역방향임
    norm_y_left, norm_y_right = (y_center - y_left)/radius, (y_center - y_right)/radius

    angle = math.atan2((norm_y_left + norm_y_right)/2, (norm_x_left + norm_x_right)/2)
    coord = (math.cos(angle)*radius + x_center, -math.sin(angle)*radius + y_center)

    # print((norm_y1 + norm_y2)/2, (norm_x1 + norm_x2)/2)
    # print(angle)
    return list(map(int, coord))

def dgr2pct(dgr):
    return round(dgr/360*100, 1)
