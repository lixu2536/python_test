import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

# 直方图 histogram，均衡化 equalization，规定化 stipulate
def histogramFUN(inputimg):
    img_gray = cv2.imread(inputimg, cv2.IMREAD_GRAYSCALE)  # flags=0 读取为灰度图像

    equ = cv2.equalizeHist(img_gray)

    plt.subplot(221), plt.imshow(img_gray, cmap=plt.cm.gray), plt.title('img_gray'), plt.axis('off')  # 坐标轴关闭
    plt.subplot(222), plt.imshow(equ, cmap=plt.cm.gray), plt.title('cv_equ'), plt.axis('off')  # 坐标轴关闭
    plt.subplot(223), plt.hist(img_gray.ravel(), 256), plt.title('img_gray_hist')
    plt.subplot(224), plt.hist(equ.ravel(), 256), plt.title('equ_hist')
    # plt.savefig("histogram_fun.jpg", dpi=300, bbox_inches="tight")

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return equ, img_gray


def math(inputimg):
    img = cv2.imread(inputimg, cv2.IMREAD_GRAYSCALE)

    hist, bins = np.histogram(img.ravel(), 256, [0, 256])

    cdf = np.cumsum(hist)

    # 构建 Numpy 掩模数组，cdf 为原数组，当数组元素为 0 时，掩盖（计算时被忽略）。
    cdf_m = np.ma.masked_equal(cdf, 0)

    # 均衡化公式
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    # 对被掩盖的元素赋值，这里赋值为 0
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    img2 = cdf[img]

    hist2, bins2 = np.histogram(img2.ravel(), 256, [0, 256])
    # 直方图计算：对象数组，区间个数，区间min max
    # 返回两个数组：数组 hist 显示直方图的值，bin_edges 显示 bin 边缘。
    # bin_edges 的大小总是 1+(hist 的大小)，即 length(hist)+1。

    cdf2 = np.cumsum(hist2)  # 数组累加
    cdf_equal_normalized = cdf2 * max(hist2) / max(cdf2)

    # plt.hist(img2.ravel(), 256, [0, 256])
    # plt.plot(cdf_equal_normalized, "r")
    cv2.imwrite("equal.jpg", img2)

    plt.subplot(221), plt.imshow(img, cmap=plt.cm.gray), plt.title('gray'), plt.axis('off')  # 坐标轴关闭
    plt.subplot(222), plt.imshow(img2, cmap=plt.cm.gray), plt.title('math_equ'), plt.axis('off')  # 坐标轴关闭
    plt.subplot(223), plt.hist(img.ravel(), 256), plt.title('gray_hist')
    plt.subplot(224), plt.hist(img2.ravel(), 256), plt.title('math_hist')

    plt.savefig("histogram_math.jpg", dpi=300, bbox_inches="tight")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # file = './images/baboon.tif'
    # file = './grayline/beta05.png'
    file = './moer/new_moer1.png'
    # 对不同图像进行测试，观察均衡化的效果
    gray, equ = histogramFUN(file)
    #
    # cv2.imwrite('./histogram/equalization.png', equ)
    # cv2.imwrite('./histogram/original_gray.png', gray)
    # math(file)
