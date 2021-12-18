import math
import numpy as np

def get_point(points, threshold):
    count = 0
    points_clean = []
    for point in points:
        if point['score'] > threshold:
            count += 1
            points_clean.append(point)
    return points_clean

def cal_dis(a, b):
    return math.sqrt(math.pow(a['bbox'][0]-b['bbox'][0], 2)+math.pow(a['bbox'][1]-b['bbox'][1], 2))

def cross(a):
    center_x = center['bbox'][0]
    center_y = center['bbox'][1]
    left_x = a['bbox'][0]
    left_y = a['bbox'][1]
    x1 = left_x - center_x
    y1 = left_y - center_y
    theta_y = math.degrees(math.acos((-y1 / math.sqrt(x1 * x1 + y1 * y1))))
    if x1 < 0:
        theta_y = 360 - theta_y
    return theta_y

def pair_one(center_point, key_points):
    global center
    center = center_point
    key_points = sorted(key_points, key=cross, reverse=True)
    groups = []
    for i in range(len(key_points)):
        score = (center_point['score'] + key_points[i]['score'] + key_points[(i+1)%len(key_points)]['score'])/3
        groups.append([tuple(center_point['bbox'][0:2]), tuple(key_points[i]['bbox'][0:2]), tuple(key_points[(i+1)%len(key_points)]['bbox'][0:2]), score])
    return groups

def pair_multi(center_points, key_points, r, threshold):
    global center
    center = center_points[0]
    key_points = sorted(key_points, key=cross, reverse=True)
    groups = []
    for i in range(len(key_points)):
        key_point = key_points[i]
        chk = 100
        for j in range(len(center_points)):
            r_ = cal_dis(key_point, center_points[j])
            if abs(r_-r) <= threshold and abs(r_ -r) < chk:
                tar_center = center_points[j]
                chk = abs(r_ -r)
        r_ = cal_dis(key_points[(i+1)%len(key_points)], tar_center)
        if abs((r_ - r)) <= threshold:
            score = (tar_center['score'] + key_points[i]['score'] + key_points[(i + 1) % len(key_points)][
                'score']) / 3
            groups.append([tuple(tar_center['bbox'][0:2]), tuple(key_points[i]['bbox'][0:2]), tuple(key_points[(i+1)%len(key_points)]['bbox'][0:2]), score])
    return groups

def estimatie_r(centers, keys):
    all_r = []
    for center in centers:
        for key_point in keys:
            all_r.append(cal_dis(center, key_point))
    r_dict = {}
    for r in all_r:
        if round(r) not in r_dict.keys():
            r_dict[round(r)] = [r]
        else:
            r_dict[round(r)].append(r)
    r_record = []
    len_value = 0
    for key, value in r_dict.items():
        if len(value) > len_value:
            r_record = value
            len_value = len(value)
            r_key = key
    
    i=1
    while len(r_record) < len(keys):
        if r_key-i in r_dict.keys():
            r_record.extend(r_dict[r_key-i])
        if r_key+i in r_dict.keys():
            r_record.extend(r_dict[r_key+i])
        i+=1
    r = np.mean(r_record)
    r_record.sort()
    threshold = abs(r_record[-1] - r_record[0]) +0.0125*abs(len(keys) - len(r_record))
    return r, threshold

def check_key(k, keys, centers):
    key = keys[k]
    left_key = keys[(k-1)%len(keys)]
    right_key = keys[(k+1)%len(keys)]
    flag = False
    for center in centers:
        r = cal_dis(key, center)
        rl = cal_dis(left_key, center)
        rr = cal_dis(right_key, center)
        if abs((rl-r)/r) < 0.1 or abs((rr-r)/r) < 0.1:
            flag = True
            break
    return flag
    
def check_center(keys, center):
    flag = False
    for i in range(len(keys)):
        rl = cal_dis(keys[i], center)
        rr = cal_dis(keys[(i+1)%len(keys)], center)
        if abs((rl-rr)/rr) < 0.1:
            flag = True
            break
    return flag

def filter(centers, keys):
    global center
    center = centers[0]
    keys = sorted(keys, key=cross, reverse=True)
    for i in range(len(keys)-1, -1, -1):
        if not check_key(i, keys, centers):
            keys.remove(keys[i])
    for i in range(len(centers)-1, -1, -1):
        if not check_center(keys, centers[i]):
            centers.remove(centers[i])
    return centers, keys

def ekey(x):
    return x[0]

def GroupPie(tls_raw, brs_raw):
    centers = []
    for temp in tls_raw.values():
        for point in temp:
            bbox = [point[2], point[3]]
            bbox = [float(e) for e in bbox]
            category_id = int(point[1])
            score = float(point[0])
            centers.append({'bbox': bbox, 'category_id': category_id, 'score': score})
    keys = []
    for temp in brs_raw.values():
        for point in temp:
            bbox = [point[2], point[3]]
            bbox = [float(e) for e in bbox]
            category_id = int(point[1])
            score = float(point[0])
            keys.append({'bbox': bbox, 'category_id': category_id, 'score': score})
    centers = get_point(centers, 0.20)
    keys = get_point(keys, 0.20)
    if len(centers) > 0 and len(keys) > 0:
        centers, keys = filter(centers, keys)
        if len(centers) == 1:
            groups = pair_one(centers[0], keys)
        if len(centers) > 1:
            r, threshold = estimatie_r(centers, keys)
            groups = pair_multi(centers, keys, r, threshold)
        data_rs = []
        for group in groups:
            center_x = group[0][0]
            center_y = group[0][1]
            left_x = group[2][0]
            left_y = group[2][1]
            right_x = group[1][0]
            right_y = group[1][1]
            x1 = left_x - center_x
            y1 = left_y - center_y
            x2 = right_x - center_x
            y2 = right_y - center_y
            theta = math.degrees(math.acos(
                max(min((x1 * x2 + y1 * y2) / math.sqrt(x1 * x1 + y1 * y1) / math.sqrt(x2 * x2 + y2 * y2), 1), -1)))
            cross = x1 * (y2) - x2 * (y1)
            if cross < 0: # 180도 넘으면
                theta = 360 - theta
            theta_y = math.degrees(math.acos((-y1 / math.sqrt(x1 * x1 + y1 * y1))))
            if x1 < 0:
                theta_y = 360 - theta_y
            data_rs.append([theta_y, theta, group])
        data_rs.sort(key=ekey)
        theta_rtn, group_rtn = [], []
        for datum in data_rs:
            theta_rtn.append(datum[1])
            group_rtn.append(datum[2])
        return theta_rtn, group_rtn

    else:
        return ["data_pure is not found"],["groups is not found"]
