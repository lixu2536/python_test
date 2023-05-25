import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology


def imageresize(img, new_x=700, new_y=500):
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


def value2_bone(img, min=0, max=255):

    # 图像二值化处理，
    ret, binary = cv2.threshold(img, min, max, cv2.THRESH_BINARY)   # 1:110  2:115  3:130
    # ret2, binary2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("The threshold is " + str(ret))

    binary[binary == 255] = 1
    binary = 1 - binary   # 目标1，背景0

    skeleton = morphology.skeletonize(binary)  # 骨架提取
    skeleton = skeleton.astype(np.uint8) * 255
    return skeleton


def main(name):
    # 读取去噪后的图片，进行高斯处理一次
    image = cv2.imread('./moer/new_moer{}-mid5.png'.format(str(name)), cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('image', fontsize=10), plt.axis('off')
    # plt.subplot(232), plt.imshow(binary1, cmap='gray'), plt.title('th1', fontsize=10), plt.axis('off')
    # plt.subplot(233), plt.imshow(binary2, cmap='gray'), plt.title('th2', fontsize=10), plt.axis('off')

    # 二值化，提取骨架，返回直线图
    skeleton1 = value2_bone(image, 120)
    skeleton2 = value2_bone(image)

    plt.subplot(232), plt.imshow(skeleton1, cmap='gray'), plt.title('skeleton1', fontsize=10), plt.axis('off')
    plt.subplot(233), plt.imshow(skeleton2, cmap='gray'), plt.title('skeleton2', fontsize=10), plt.axis('off')
    cv2.imwrite("./moer/cv-skeleton{}.png".format(str(name)), skeleton2)  # 保存骨架2提取后的图片
    cv2.imwrite("./moer/skeleton{}.png".format(str(name)), skeleton1)  # 保存骨架1提取后的图片 效果较好

    img = Image.open("./moer/skeleton{}.png".format(str(name)))
    img = imageresize(img)
    cv2.imwrite("./moer/new_skeleton{}.png".format(str(name)), img)  # 保存裁剪后图片
    plt.subplot(236), plt.imshow(img, cmap='gray'), plt.title('newresult', fontsize=10), plt.axis('off')
    plt.show()


if __name__ == "__main__":
    n = 3
    main(n)