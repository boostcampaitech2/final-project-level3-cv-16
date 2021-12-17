# keypoints operation

def get_flattened_keypoints(
    keypoints_group
):
    '''
    [[[x_center, y_center], [x_ccw, y_ccw], [x_cw, y_cw], confidence], ...]
        -> [[x_center, y_center, x_ccw, y_ccw, x_cw, y_cw], ...]
    '''
    flattened_keypoints = []
    for points_group_and_confidence in keypoints_group:
        points_group = points_group_and_confidence[:-1]
        keyvals = []
        for xy in points_group:
            keyvals.extend(xy)
        flattened_keypoints.append(keyvals)
    return flattened_keypoints