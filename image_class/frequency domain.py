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
    input = './images/Lena.bmp'
    # input = './moer/moer3.png'
    img = cv2.imread(input, 0)
    ft, fshift = transform(img)
    idealk1 = idealFilterLP(20, img.shape)  # ideal20
    ff1 = fshift * idealk1
    ifshift1, ift1 = inversion(ff1)
    idealk2 = idealFilterLP(40, img.shape)  # 40
    ff2 = fshift * idealk2
    ifshift2, ift2 = inversion(ff2)

    # idealk3 = idealFilterLP(60, img.shape)  # 60
    # ff3 = fshift * idealk3
    # ifshift3, ift3 = inversion(ff3)

    idealk3 = gaussianLP(40, img.shape)  # gauss--exp指数
    ff3 = fshift * idealk3
    ifshift3, ift3 = inversion(ff1)

    plt.subplot(431), plt.imshow(img, cmap='gray'), plt.title('Original-noise'), plt.axis('off')
    plt.subplot(432), plt.imshow(np.log(1 + np.abs(ft)), cmap='gray'), plt.title('Magnitude Spectrum'), plt.axis('off')
    plt.subplot(433), plt.imshow(np.log(1 + np.abs(fshift)), cmap='gray'), plt.title('Centered Spectrum'), plt.axis('off')

    plt.subplot(434), plt.imshow(idealk1, cmap='gray'), plt.title('ideal-mask-20'), plt.axis('off')
    plt.subplot(435), plt.imshow(np.log(1 + np.abs(ff1)), cmap='gray'), plt.title('ifshift-ideal'), plt.axis('off')
    plt.subplot(436), plt.imshow(np.log(1 + np.abs(ift1)), cmap='gray'), plt.title('ift final'), plt.axis('off')

    plt.subplot(437), plt.imshow(idealk2, cmap='gray'), plt.title('ideal-mask-40'), plt.axis('off')
    plt.subplot(438), plt.imshow(np.log(1 + np.abs(ff2)), cmap='gray'), plt.title('ifshift-gauss'), plt.axis('off')
    plt.subplot(439), plt.imshow(np.log(1 + np.abs(ift2)), cmap='gray'), plt.title('ift final'), plt.axis('off')

    plt.subplot(4,3,10), plt.imshow(idealk3, cmap='gray'), plt.title('gauss-mask-40'), plt.axis('off')
    plt.subplot(4,3,11), plt.imshow(np.log(1 + np.abs(ff3)), cmap='gray'), plt.title('ifshift-gauss'), plt.axis('off')
    plt.subplot(4,3,12), plt.imshow(np.log(1 + np.abs(ift3)), cmap='gray'), plt.title('ift final'), plt.axis('off')



    plt.show()
