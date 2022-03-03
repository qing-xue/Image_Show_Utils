from PIL import Image
from PIL import ImageDraw, ImageFont
from pylab import *
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['figure.dpi'] = 300                              # 分辨率
plt.rcParams['savefig.dpi'] = 300                             # 分辨率
plt.rcParams.update({'font.size': 6})                         # 图题大小
subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.65)  # 子图间隔

# 绘制的框的起点与终点
X0, Y0 = 100, 120
X1, Y1 = X0 + 40, Y0 + 40

# 需要对比的图片
PROJECT_PATH = r'./samples'
FILENAMES = [
    PROJECT_PATH + r'/Value_IMG_1.jpg',
    PROJECT_PATH + r'/GT_IMG_1.png',
    PROJECT_PATH + r'/CSR_IMG_1.png',
    PROJECT_PATH + r'/Ours_IMG_1.png'
]
PLOT_TITLES = FILENAMES  # 每张图片的图题


def crop_rect(im, xys=(1, 1, 160, 160)):
    w, h = im.size
    left, top, right, bottom = xys
    assert left >= 0 and top >= 0 and right <= w and bottom <= h
    block = im.crop((left, top, right, bottom))
    return block


def image_add_text(img, text, left, top, text_color=(255, 0, 0), text_size=5):
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式 这里的 SimHei.ttf 需要有这个字体
    fontStyle = ImageFont.truetype(size=text_size, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, text_color, font=fontStyle)
    return img


def plot_images(imgs, titles=(), save_dir=None):
    """ 自定义行列比较麻烦 """
    N, M = 1, len(imgs)           # 行数和列数

    for i, im in enumerate(imgs):
        plt.subplot(N, M, i + 1)

        if len(titles) > i:
            plt.title(titles[i])
        else:
            plt.title(str(i + 1))

        plt.imshow(im)
        plt.xticks([])            # 消除每张图片自己单独的横纵坐标
        plt.yticks([])

    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()


def draw_rect(im, xys=(1, 1, 160, 160), width=1):
    im2 = im.copy()
    a = ImageDraw.ImageDraw(im2)  # 用 a 来表示右侧这段
    x0, y0, x1, y1 = xys
    a.rectangle((x0, y0, x1, y1), outline='red', width=width)
    return im2


if __name__ == '__main__':
    # 读取图片
    imgs = []
    for filename in FILENAMES:
        im = Image.open(filename)
        imgs.append(im)
        print('Image {}, \tShape ({},{})'.format(filename, *im.size))
    plot_images(imgs, PLOT_TITLES)

    # 查看绘制的小块
    draw_imgs = []
    for i, im in enumerate(imgs):
        im2 = draw_rect(im, (X0, Y0, X1, Y1))
        draw_imgs.append(im2)
    plot_images(draw_imgs, PLOT_TITLES)

    # 显示抠取的小块
    blocks = []
    for i, im in enumerate(imgs):
        blocks.append(crop_rect(im, (X0, Y0, X1, Y1)))
    blocks = tuple(blocks)
    plot_images(blocks, PLOT_TITLES)
