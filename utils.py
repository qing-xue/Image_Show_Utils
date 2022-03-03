import os
from PIL import Image


def down_bicubic(data_root='D:/workplace/dataset/Dehaze/O-HAZE/train/B', output_size=(500, 500), down_scale=0.25):
    """将文件夹内的图片进行下采样并保存
      @output_size: (Weight, Height)
    """
    parent, now = data_root.rsplit('/', 1)
    out_dir = '{}/{}_Bicubic_Down'.format(parent, now)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dirs = os.listdir(data_root)

    for file in dirs:
        if not file.endswith(('.jpg', '.png', '.jpeg')):
            continue

        img = Image.open(os.path.join(data_root, file))

        if output_size is not None:
            img_down = img.resize(output_size, Image.BICUBIC)
        elif down_scale is not None:
            img_down = img.resize((round(down_scale * img.size[0]), round(down_scale * img.size[1])), Image.BICUBIC)

        print(os.path.join(out_dir, file))
        img_down.save(os.path.join(out_dir, file))


def clear_output_dir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)



if __name__ == '__main__':
    down_bicubic(
        data_root=r'D:/workplace/dataset/JPEGImages',
        output_size=(500, 500)
    )