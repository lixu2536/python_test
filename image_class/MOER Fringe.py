import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


def transform(img):
    """
    in:空域原始图像
    out：【0】：平移后的频域图像；【1】：傅里叶变换后的频域图像
    """
    f = np.fft.fft2(img)
    # the image 'img' is passed to np.fft.fft2() to compute its 2D Discrete Fourier transform f
    # f_mag = 20 * np.log(np.abs(f))
    fshift = np.fft.fftshift(f)     # 平移:将低频平移到中心
    # center = 20 * np.log(np.abs(fshift))
    return f, fshift


def inversion(img):
    """
    in:滤波后的频域图像
    out：【0】：逆平移后的频域图像；【1】：傅里叶逆变换后的空域图像
    """
    # Shift the zero-frequency component back to the top-left corner of the frequency spectrum
    ifshift = np.fft.ifftshift(img)
    # if_mag = np.log(1 + np.abs(ifshift))
    # Apply the inverse Fourier transform to obtain the final filtered image
    final = np.fft.ifft2(ifshift)
    # final_mag = np.log(1+np.abs(final))

    return ifshift, final


def distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def idealFilterLP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 1
    return base


def idealFilterHP(D0, imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 0
    return base


def butterworthLP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (distance((y, x), center)/D0)**2)
    return base


def butterworthHP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (D0/distance((y, x), center))**2)
    return base


# def TLP(D0, imgShape):
#     base = np.zeros(imgShape[:2])
#     rows, cols = imgShape[:2]
#     center = (rows/2, cols/2)
#     for x in range(cols):
#         for y in range(rows):
#             base[y, x] = 1 / (1 + (D0/distance((y, x), center))**2)
#     return base


def gaussianLP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = math.exp((-distance((y, x), center)**2)/(2*(D0**2)))
    return base


def gaussianHP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - math.exp((-distance((y, x), center)**2)/(2*(D0**2)))
    return base


if __name__ == '__main__':
    # input1 = './images/Lena.bmp'
    input1 = './moer/new_moer1-mid7.png'
    input2 = './moer/new_moer2-mid5.png'
    input3 = './moer/new_moer3-mid5.png'
    # input1 = './moer/new_skeleton1.png'
    # input2 = './moer/new_skeleton2.png'
    # input3 = './moer/new_skeleton3.png'

    img1 = cv2.imread(input1, 0)
    img2 = cv2.imread(input2, 0)
    img3 = cv2.imread(input3, 0)

    idealk1 = idealFilterLP(20, img1.shape)  # filter model
    idealk2 = idealFilterLP(40, img1.shape)

    ft1, fshift1 = transform(img1)  # moer1
    ff1 = fshift1 * idealk1
    ifshift1, ift1 = inversion(ff1)

    ft2, fshift2 = transform(img2)  # moer2
    ff2 = fshift2 * idealk1
    ifshift2, ift2 = inversion(ff2)

    ft3, fshift3 = transform(img3)  # moer2
    ff3 = fshift3 * idealk1
    ifshift3, ift3 = inversion(ff2)

    plt.subplot(431), plt.imshow(img1, cmap='gray'), plt.title('Moer1'), plt.axis('off')
    plt.subplot(432), plt.imshow(np.log(1 + np.abs(ft1)), cmap='gray'), plt.title('Magnitude Spectrum'), plt.axis('off')
    plt.subplot(433), plt.imshow(np.log(1 + np.abs(fshift1)), cmap='gray'), plt.title('Centered Spectrum'), plt.axis(
        'off')
    plt.subplot(434), plt.imshow(img2, cmap='gray'), plt.title('Moer2'), plt.axis('off')
    plt.subplot(435), plt.imshow(np.log(1 + np.abs(ft2)), cmap='gray'), plt.title('Magnitude Spectrum'), plt.axis('off')
    plt.subplot(436), plt.imshow(np.log(1 + np.abs(fshift2)), cmap='gray'), plt.title('Centered Spectrum'), plt.axis(
        'off')
    plt.subplot(437), plt.imshow(img3, cmap='gray'), plt.title('Moer3'), plt.axis('off')
    plt.subplot(438), plt.imshow(np.log(1 + np.abs(ft3)), cmap='gray'), plt.title('Magnitude Spectrum'), plt.axis('off')
    plt.subplot(439), plt.imshow(np.log(1 + np.abs(fshift3)), cmap='gray'), plt.title('Centered Spectrum'), plt.axis(
        'off')

    plt.subplot(4,3,10), plt.imshow(np.log(1 + np.abs(ift1)), cmap='gray'), plt.title('moer1-20'), plt.axis('off')
    plt.subplot(4,3,11), plt.imshow(np.log(1 + np.abs(ift2)), cmap='gray'), plt.title('moer2-20'), plt.axis('off')
    plt.subplot(4,3,12), plt.imshow(np.log(1 + np.abs(ift3)), cmap='gray'), plt.title('moer3-20'), plt.axis('off')
    plt.savefig("./moer/denoise-frequency.jpg", dpi=300, bbox_inches="tight")

    plt.show()
