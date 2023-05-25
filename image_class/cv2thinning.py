import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def imageresize(img, new_x=512, new_y=512):
    """
    :param img: 768*576
    :param new_x: 512
    :param new_y: 512
    :return: new_img: 512*512
    """
    left = int(img.size[0]/2 - new_x/2)
    upper = int(img.size[1]/2 - new_y/2)
    right = left + new_x
    lower = upper + new_y
    cropped = img.crop((left, upper, right, lower))

    new_img = np.asarray(cropped)  # pil2cv转换代码
    return new_img


def main(name):
    # 1.导入图片
    img_org = cv2.imread('./moer/new_moer{}-mid5.png'.format(str(name)), cv2.IMREAD_GRAYSCALE)
    img_org2 = cv2.medianBlur(img_org, 9)
    # 2.二值化处理
    ret2, img_bin = cv2.threshold(img_org2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img_bin[img_bin == 255] = 1
    # img_bin = (1 - img_bin)*255
    # 4.细化处理 采用cv细化函数   （效果较好）
    img_thinning = cv2.ximgproc.thinning(img_bin, thinningType=cv2.ximgproc.THINNING_GUOHALL)

    # 5.显示结果
    plt.subplot(131), plt.imshow(img_org, cmap='gray'), plt.title('img_org{}'.format(str(name)), fontsize=10), plt.axis(
        'off')
    plt.subplot(132), plt.imshow(img_bin, cmap='gray'), plt.title('img_bin', fontsize=10), plt.axis('off')
    plt.subplot(133), plt.imshow(img_thinning, cmap='gray'), plt.title('img_thinning', fontsize=10), plt.axis('off')
    resizeimg = Image.fromarray(img_thinning)
    resizeimg = imageresize(resizeimg)
    # cv2.imwrite('./moer/newcv_skeleton{}.png'.format(str(name)), resizeimg)
    plt.show()


if __name__ == "__main__":
    n = 3
    main(n)