import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def imageresize(img, new_x=750, new_y=560):
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


def cv_main(name):
    # 读入图像，并先进行裁剪边缘，由于原先边缘存在黑边
    image = Image.open("./moer/moer{}.png".format(name))
    image = imageresize(image)
    # image = cv2.imread("./moer/moer2.png", cv2.IMREAD_GRAYSCALE)

    # 使用opencv自带函数进行：领域平均、中值滤波去噪声，并对比结果。
    imageAver3 = cv2.blur(image, (3, 3))
    imageAver5 = cv2.blur(image, (5, 5))
    imageMedian3 = cv2.medianBlur(image, 9)
    imageMedian5 = cv2.medianBlur(image, 25)
    imageMedian7 = cv2.medianBlur(image, 49)
    mid_num = 7
    cv2.imwrite('./moer/new_moer{}-mid{}.png'.format(name, mid_num), imageMedian7)

    plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('image', fontsize=10), plt.axis('off')
    plt.subplot(232), plt.imshow(imageAver3, cmap='gray'), plt.title('Aver3', fontsize=10), plt.axis('off')
    plt.subplot(233), plt.imshow(imageAver5, cmap='gray'), plt.title('Aver5', fontsize=10), plt.axis('off')
    plt.subplot(234), plt.imshow(imageMedian3, cmap='gray'), plt.title('Median3', fontsize=10), plt.axis('off')
    plt.subplot(235), plt.imshow(imageMedian5, cmap='gray'), plt.title('Median5', fontsize=10), plt.axis('off')
    plt.subplot(236), plt.imshow(imageMedian7, cmap='gray'), plt.title('Median7', fontsize=10), plt.axis('off')
    plt.show()


if __name__ == "__main__":
    n = 3
    cv_main(n)
