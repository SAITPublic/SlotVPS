import os
import os.path
import matplotlib
matplotlib.use('AGG') # or PDF, SVG, PS
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import numpy as np
import cv2
import json


def draw_line_chart(x, ys, labels, x_label, y_label, title, rotation, fontsize, save_path):
    colors = ['#E6B0AA', '#D98880', '#CD6155', '#A93226', '#D7BDE2', '#C39BD3', '#AF7AC5', '#884EA0',
              '#A9CCE3', '#7FB3D5', '#5499C7', '#2471A3', '#A3E4D7', '#76D7C4', '#48C9B0', '#17A589',
              '#FAD7A0', '#F8C471', '#F5B041', '#D68910', '#AEB6BF', '#85929E', '#5D6D7E', '#2E4053']
    for i in range(len(ys)):
        plt.plot(x, ys[i], 'o-', color=colors[i], label=labels[i])
    for each_x, last_y in zip(x, ys[3]):
        plt.text(each_x, last_y, str(last_y)[:5], fontsize=8.5)
    plt.xlabel(x_label)
    if fontsize is not None:
        plt.xticks(rotation=rotation, fontsize=fontsize)
    else:
        plt.xticks(rotation=rotation)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def draw_line_charts(x, yses, labels, rotation, fontsize, output_dir):
    ys_vpq, ys_vsq, ys_vrq, ys_errp = yses[:]
    all_pq_labels, all_sq_labels, all_rq_labels, all_errp_labels = labels[:]
    json.dump(ys_vpq[0: len(ys_vpq): 3], open(os.path.join(output_dir, 'vpq_all.json'), 'w'))
    json.dump(ys_vpq[1: len(ys_vpq): 3], open(os.path.join(output_dir, 'vpq_things.json'), 'w'))
    draw_line_chart(x, ys_vpq[0: len(ys_vpq): 3], all_pq_labels[0: len(ys_vpq): 3], x_label='video_id',
                    y_label='vpq_all', rotation=rotation, fontsize=fontsize,
                    title='vpq_all_fig', save_path=os.path.join(output_dir, 'vpq_all_fig.png'))
    draw_line_chart(x, ys_vpq[1: len(ys_vpq): 3], all_pq_labels[1: len(ys_vpq): 3], x_label='video_id',
                    y_label='vpq_things', rotation=rotation, fontsize=fontsize,
                    title='vpq_things_fig', save_path=os.path.join(output_dir, 'vpq_things_fig.png'))
    draw_line_chart(x, ys_vpq[2: len(ys_vpq): 3], all_pq_labels[2: len(ys_vpq): 3], x_label='video_id',
                    y_label='vpq_stuff', rotation=rotation, fontsize=fontsize,
                    title='vpq_stuff_fig', save_path=os.path.join(output_dir, 'vpq_stuff_fig.png'))
    draw_line_chart(x, ys_vsq[0: len(ys_vsq): 3], all_sq_labels[0: len(ys_vsq): 3], x_label='video_id',
                    y_label='vsq_all', rotation=rotation, fontsize=fontsize,
                    title='vsq_all_fig', save_path=os.path.join(output_dir, 'vsq_all_fig.png'))
    draw_line_chart(x, ys_vsq[1: len(ys_vsq): 3], all_sq_labels[1: len(ys_vsq): 3], x_label='video_id',
                    y_label='vsq_things', rotation=rotation, fontsize=fontsize,
                    title='vsq_things_fig', save_path=os.path.join(output_dir, 'vsq_things_fig.png'))
    draw_line_chart(x, ys_vsq[2: len(ys_vsq): 3], all_sq_labels[2: len(ys_vsq): 3], x_label='video_id',
                    y_label='vsq_stuff', rotation=rotation, fontsize=fontsize,
                    title='vsq_stuff_fig', save_path=os.path.join(output_dir, 'vsq_stuff_fig.png'))
    draw_line_chart(x, ys_vrq[0: len(ys_vrq): 3], all_rq_labels[0: len(ys_vrq): 3], x_label='video_id',
                    y_label='vrq_all', rotation=rotation, fontsize=fontsize,
                    title='vrq_all_fig', save_path=os.path.join(output_dir, 'vrq_all_fig.png'))
    draw_line_chart(x, ys_vrq[1: len(ys_vrq): 3], all_rq_labels[1: len(ys_vrq): 3], x_label='video_id',
                    y_label='vrq_things', rotation=rotation, fontsize=fontsize,
                    title='vrq_things_fig', save_path=os.path.join(output_dir, 'vrq_things_fig.png'))
    draw_line_chart(x, ys_vrq[2: len(ys_vrq): 3], all_rq_labels[2: len(ys_vrq): 3], x_label='video_id',
                    y_label='vrq_stuff', rotation=rotation, fontsize=fontsize,
                    title='vrq_stuff_fig', save_path=os.path.join(output_dir, 'vrq_stuff_fig.png'))
    draw_line_chart(x, ys_errp[0: len(ys_errp): 3], all_errp_labels[0: len(ys_errp): 3], x_label='video_id',
                    y_label='errp_all', rotation=rotation, fontsize=fontsize,
                    title='errp_all_fig', save_path=os.path.join(output_dir, 'errp_all_fig.png'))
    draw_line_chart(x, ys_errp[1: len(ys_errp): 3], all_errp_labels[1: len(ys_errp): 3], x_label='video_id',
                    y_label='errp_things', rotation=rotation, fontsize=fontsize,
                    title='errp_things_fig', save_path=os.path.join(output_dir, 'errp_things_fig.png'))
    draw_line_chart(x, ys_errp[2: len(ys_errp): 3], all_errp_labels[2: len(ys_errp): 3], x_label='video_id',
                    y_label='errp_stuff', rotation=rotation, fontsize=fontsize,
                    title='errp_stuff_fig', save_path=os.path.join(output_dir, 'errp_stuff_fig.png'))


def draw_feature_maps(features, save_name, block_num=9, figsize=(100, 100)):
    plt.figure(figsize=figsize)
    layer_viz = features[0, :, :, :]
    layer_viz = layer_viz.cpu().data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == block_num:  # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(int(np.sqrt(block_num)), int(np.sqrt(block_num)), i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    plt.savefig(save_name + ".png")
    # plt.show()
    plt.close()


def dist_print(obj):
    if dist.get_rank() == 0:
        print(obj)


def save_color_map(im_color, file_name, apply_color_map=True, clip=True):
    if clip:
        im_color = np.clip(im_color, 0, 1)
    im_color = np.uint8(255 * im_color)
    if apply_color_map:
        im_color = cv2.applyColorMap(im_color, cv2.COLORMAP_JET)
    cv2.imwrite(file_name, im_color)
