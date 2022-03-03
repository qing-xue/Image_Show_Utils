import numpy as np


class BoxesProposal():
    """对一张图的候选框区域进行提取"""
    def __init__(self, box_w=96, box_h=96, stride_w=96, stride_h=96, epsilon=10):
        self.box_w = box_w
        self.box_h = box_h
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.epsilon = epsilon

    def propose(self, arr):
        """get patches.

        :param epsilon: 右下方边界容忍值，低于之则直接丢弃
        :return: 返回截取的 patches 对应于矩阵的坐标
        """
        box_w = self.box_w
        box_h = self.box_h
        stride_w = self.stride_w
        stride_h = self.stride_h
        epsilon = self.epsilon

        height, width = arr.shape  # numpy array 与 PIL Image 返回顺序不一样！
        if width < box_w or height < box_h:
            return

        patches_idx = []
        iw = np.arange(0, width - box_w + 1, stride_w)
        jh = np.arange(0, height - box_h + 1, stride_h)
        for i in iw:
            for j in jh:
                box = (i, j, i + box_w, j + box_h)
                patches_idx.append(box)
        # repair x and y orientation's boundary
        if width - box_w - iw[-1] > epsilon:
            for j in jh:
                box = (width - box_w, j, width, j + box_h)
                patches_idx.append(box)
        if height - box_h - jh[-1] > epsilon:
            for i in iw:
                box = (i, height - box_h, i + box_w, height)
                patches_idx.append(box)
        # need only once
        if width - box_w - iw[-1] > epsilon and height - box_h - jh[-1] > epsilon:
            box = (width - box_w, height - box_h, width, height)
            patches_idx.append(box)

        return patches_idx