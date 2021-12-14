import numpy as np
from numpy import linalg as LA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_distance(x1, y1, x2, y2):
    point_1 = np.array([x1, y1])
    point_2 = np.array([x2, y2])
    return LA.norm(point_1 - point_2)

def get_arc_center(points):
    x_center, y_center, x_left, y_left, x_right, y_right = points
    assert abs(get_distance(x_left,y_left, x_center, y_center) - get_distance(x_right, y_right, x_center, y_center)) < 5

    radius = get_distance(x_left,y_left, x_center, y_center)
    norm_x_left, norm_x_right = (x_left - x_center)/radius, (x_right - x_center)/radius

    # y가 역방향임
    norm_y_left, norm_y_right = (y_center - y_left)/radius, (y_center - y_right)/radius

    angle = np.arctan2((norm_y_left + norm_y_right)/2, (norm_x_left + norm_x_right)/2)
    coord = (np.cos(angle)*radius + x_center, -1 * np.sin(angle)*radius + y_center)

    return list(map(int, coord))


def plot_model_results(image_arr, parsed_response):

    
    H, W, _ = parsed_response["im_shape"]
    degree_list = parsed_response["dgr"]
    group_list = parsed_response["grp"]

    unit = 8
    figsize = (1, H / W) if H >= W else (W / H, 1)
    figsize = (unit * figsize[0], unit * figsize[1])
    
    fig = plt.figure(figsize=figsize)
    plt.imshow(image_arr)
    for degree, group in zip(degree_list, group_list):
        
        center_x, center_y = group[0]
        ccw_x, ccw_y = group[2]
        cw_x, cw_y = group[1]
        
        plt.plot([center_x, ccw_x], [center_y, ccw_y], '--r', linewidth=3)
        plt.plot([center_x, cw_x], [center_y, cw_y], '--r', linewidth=3)
        
        plt.plot(center_x, center_y, '*b', markersize=int(unit * (30 / 8)))
        plt.plot(ccw_x, ccw_y, '.r', markersize=int(unit * (40 / 8)))
        plt.plot(cw_x, cw_y, '.r', markersize=int(unit * (40 / 8)))

        arc_center = get_arc_center(
            [center_x, center_y, ccw_x, ccw_y, cw_x, cw_y]
        )
        text_x = (arc_center[0] + center_x)/2
        text_y = (arc_center[1] + center_y)/2

        bbox_config = {
            'boxstyle': 'round',
            'ec': (0.8, 0.8, 0.8),
            'fc': (1.0, 1.0, 1.0)
        }
        text_str = f"{float(degree)/360*100:.2f}%"
        if degree < 180:
            plt.text(
                text_x, text_y, 
                text_str, fontsize=int(unit * (30 / 8)),
                bbox = bbox_config
            )
        else:
            text_x = 2 * center_x - text_x
            text_y = 2 * center_y - text_y
            plt.text(
                text_x, text_y, 
                text_str, fontsize=int(unit * (30 / 8)),
                bbox = bbox_config
            )

    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(
        left = 0, bottom = 0, right = 1, top = 1,
        hspace = 0, wspace = 0
    )

    fig.canvas.draw()
    image_bunary = fig.canvas.renderer._renderer
    image_arr = np.array(image_bunary)
    
    return image_arr