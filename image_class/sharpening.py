import cv2
import numpy as np
import matplotlib.pyplot as plt


def Sobel(img, threshold):
    # 定义sobel算子卷积核
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows, columns, = img.shape
    new = np.zeros(img.shape)
    # 循环相乘卷积运算
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            x = sum(sum(sobel_x * img[i:i + 3, j:j + 3]))
            y = sum(sum(sobel_y * img[i:i + 3, j:j + 3]))
            new[i + 1, j + 1] = np.sqrt((x ** 2) + (y ** 2))
    # 设定阈值显示范围
    for p in range(0, rows):
        for q in range(0, columns):
            if new[p, q] < threshold:
                new[p, q] = 0
    return new


def Robert(img, threshold):
    kernel_x = np.array([[-1, 0], [0, 1]], dtype=int)
    kernel_y = np.array([[0, -1], [1, 0]], dtype=int)
    rows, columns, = img.shape
    new = np.zeros(img.shape)
    for i in range(0, rows-1):
        for j in range(0, columns-1):
            x = sum(sum(kernel_x * img[i:i + 2, j:j + 2]))
            y = sum(sum(kernel_y * img[i:i + 2, j:j + 2]))
            new[i + 1, j + 1] = np.sqrt((x ** 2) + (y ** 2))

    for p in range(0, rows):
        for q in range(0, columns):
            if new[p, q] < threshold:
                new[p, q] = 0
    return new


def Prewitt(img, threshold):
    kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    rows, columns, = img.shape
    new = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            x = sum(sum(kernel_x * img[i:i + 3, j:j + 3]))
            y = sum(sum(kernel_y * img[i:i + 3, j:j + 3]))
            new[i + 1, j + 1] = np.sqrt((x ** 2) + (y ** 2))

    for p in range(0, rows):
        for q in range(0, columns):
            if new[p, q] < threshold:
                new[p, q] = 0
    return new


def Laplacian(img, threshold):
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=int)
    rows, columns, = img.shape
    new = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            new[i + 1, j + 1] = sum(sum(kernel * img[i:i + 3, j:j + 3]))

    for p in range(0, rows):
        for q in range(0, columns):
            if new[p, q] < threshold:
                new[p, q] = 0
    return new


if __name__ == "__main__":
    img = cv2.imread('images/LenaRGB.bmp')
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度化处理图像
    imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = Sobel(imgGRAY, 10)
    robert = Robert(imgGRAY, 10)
    prewitt = Prewitt(imgGRAY, 10)
    laplacian = Laplacian(imgGRAY, 10)

    plt.subplot(231), plt.imshow(imgRGB, cmap='gray'), plt.title('LenaRGB', fontsize=10), plt.axis('off')
    plt.subplot(232), plt.imshow(sobel, cmap='gray'), plt.title('sobel', fontsize=10), plt.axis('off')
    plt.subplot(233), plt.imshow(robert, cmap='gray'), plt.title('robert', fontsize=10), plt.axis('off')
    plt.subplot(234), plt.imshow(imgGRAY, cmap='gray'), plt.title('LenaGRAY', fontsize=10), plt.axis('off')
    plt.subplot(235), plt.imshow(prewitt, cmap='gray'), plt.title('prewitt', fontsize=10), plt.axis('off')
    plt.subplot(236), plt.imshow(laplacian, cmap='gray'), plt.title('laplacian', fontsize=10), plt.axis('off')

    plt.savefig('sharpen.jpg', dpi=500, bbox_inches='tight')
    plt.show()

