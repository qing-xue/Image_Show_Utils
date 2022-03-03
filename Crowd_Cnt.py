import os
import math
import h5py
import numpy as np
from PIL import Image
from matplotlib import cm

from show_crop import (
    crop_rect, draw_rect, plot_images, image_add_text
)
from crop import BoxesProposal


DIR_RESULT = r'./result_map'
DIR_CSR = DIR_RESULT + r'/baseline_h5'
DIR_GT = DIR_RESULT + r'/ground_truth_h5'
DIR_Our = DIR_RESULT + r'/partA_avg_size3_h5'
DIR_BIG = DIR_RESULT + r'/images'

# 文件夹下的文件名一致，只遍历其一
IMGS_GT = sorted([x for x in os.listdir(DIR_GT) if x.endswith('h5')])
IMGS_BIG = sorted([x for x in os.listdir(DIR_BIG) if x.endswith('jpg')])

# 输出文件夹
OUTPUT_DIR_CSR = DIR_CSR + '_rect'
OUTPUT_DIR_GT = DIR_GT + '_rect'
OUTPUT_DIR_Our = DIR_Our + '_rect'
OUTPUT_DIR_BIG = DIR_BIG + '_rect'
OUTPUT_DIR_333 = DIR_RESULT + r'/333'

output_dirs = [OUTPUT_DIR_CSR, OUTPUT_DIR_GT, OUTPUT_DIR_Our, OUTPUT_DIR_BIG, OUTPUT_DIR_333]
for x in output_dirs:
    if not os.path.exists(x):
        os.makedirs(x)


def get_h5(file_path):
    gt_file = h5py.File(file_path, 'r')
    target = np.asarray(gt_file['density'])
    return target


def criterion(box):
    """用何种评估方式"""
    return float(np.sum(box))


if __name__ == '__main__':
    # 枚举筛选框
    boxProposal = BoxesProposal(box_w=9, box_h=9, stride_w=1, stride_h=1)

    # 遍历图片：针对 .h5 和 PIL Image 类型要做不同处理！
    for i, img_name in enumerate(IMGS_GT):
        im_GT = get_h5(os.path.join(DIR_GT, img_name))
        im_CSR = get_h5(os.path.join(DIR_CSR, img_name))
        im_Ours = get_h5(os.path.join(DIR_Our, img_name))
        im_Big = Image.open(os.path.join(DIR_BIG, IMGS_BIG[i]))

        # 返回：(i, j, i + box_w, j + box_h)，width 在前
        patches = boxProposal.propose(im_GT)
        img_name = img_name.split('.')[0] + '.png'

        imgs = [im_GT, im_CSR, im_Ours]
        PLOT_TITLES = ['GT_' + img_name, 'CSR_' + img_name, 'Ours_' + img_name]
        # width, height = im_GT.size  # PIL image
        height, width = im_GT.shape

        # 根据条件筛选框
        c_patches = []
        for box in patches:
            box_GT = im_GT[box[1]:box[3], box[0]:box[2]]  # height 在前
            box_CSR = im_CSR[box[1]:box[3], box[0]:box[2]]
            box_Ours = im_Ours[box[1]:box[3], box[0]:box[2]]

            cnt_gt, cnt_csr, cnt_our = list(map(criterion, (box_GT, box_CSR, box_Ours)))
            # TODO: magic number
            if cnt_gt < 10 or abs(cnt_csr - cnt_our) < 5:
                continue

            c1 = math.fabs(cnt_gt - cnt_csr)
            c2 = math.fabs(cnt_gt - cnt_our)

            if c1 > c2:
                # print('Find box', box, c1, ' < ', c2)
                # TODO: 注意顺序！！！
                c_patches.append((box, c1, cnt_gt, cnt_csr, cnt_our))

        # assert len(c_patches) > 1, 'Not Found!'
        if len(c_patches) < 1:
            continue

        # 绘图
        py_list = sorted(c_patches, key=lambda k: k[1])  # k[1]=c1
        draw_patch = py_list[0][0]  # TODO: 注意顺序！！
        cnt_gt = py_list[0][2]
        cnt_csr = py_list[0][3]
        cnt_our = py_list[0][4]

        print('Image', img_name, 'Find box', draw_patch)
        print('\tCount GT: {}, Count CSR: {}, Count Ours: {}'.format(cnt_gt, cnt_csr, cnt_our))

        draw_imgs = []
        for i, im in enumerate(imgs):
            im2 = draw_rect(Image.fromarray(np.uint8(cm.jet(im) * 255)), draw_patch)
            draw_imgs.append(im2)
        plot_images(draw_imgs, PLOT_TITLES, save_dir=os.path.join(OUTPUT_DIR_333, img_name))

        # 保存绘制好的图片
        draw_big = draw_rect(im_Big, [x * 8 for x in draw_patch], width=4)  # TODO: magic number
        draw_big.save(os.path.join(OUTPUT_DIR_BIG, img_name))

        # TODO: 注意顺序
        draw_imgs[0].save(os.path.join(OUTPUT_DIR_GT, img_name))
        draw_imgs[1].save(os.path.join(OUTPUT_DIR_CSR, img_name))
        draw_imgs[2].save(os.path.join(OUTPUT_DIR_Our, img_name))



