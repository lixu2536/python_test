import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

PATH = './picture/'
PATH2 = './transform/'
PATH3 = './grayline/'

def addnoise(inputimg):
    # 读取图片
    img = cv2.imread(inputimg, -1)
    # print(img.shape)
    # img = np.array(img / 255, dtype=float)
    addimg = np.zeros(img.shape)
    # 设置高斯分布的均值和方差
    mean = 0
    sigma = 16
    # noise叠加次数
    ave = 1

    for i in range(ave):
        # 根据均值和标准差生成符合高斯分布的噪声
        gauss = np.random.normal(mean, sigma, img.shape)
        # 给图片添加高斯噪声
        noise_img = img + gauss
        # 设置图片添加高斯噪声之后的像素值的范围
        noise_img = np.clip(noise_img, a_min=0, a_max=255)
        # noise_img = np.uint8(noise_img * 255)
        # 保存图片
        filename = PATH + 'noise_img' + '%02d'%i + '.jpg'
        cv2.imwrite(filename, noise_img)
        addimg += noise_img
        if i == ave-1:
            print(addimg)

    ave_img = addimg/ave
    var = np.var(ave_img - img)
    print(var)
    return addimg, ave_img

def masktransform(inputimg, maskimg):
    img = cv2.imread(inputimg, cv2.IMREAD_GRAYSCALE)
    maskimg = cv2.imread(maskimg, cv2.IMREAD_GRAYSCALE)
    mask = np.zeros(maskimg.shape)

    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(cols):
            if maskimg[row, col] != 0:
                mask[row, col] = 1

    outimg = mask * img
    return outimg

def pingyi(inputimg, x, y):
    img = cv2.imread(inputimg, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape[:2]
    H = np.float32([[1, 0, x],
                    [0, 1, y]])
    pingyiimg1 = cv2.warpAffine(img, H, (cols, rows))
    pingyiimg2 = cv2.warpAffine(img, H, (2*cols, 2*rows))
    # 输出图像需要反置，先列后行
    return pingyiimg1, pingyiimg2

def grayline(inputimg):
    img = cv2.imread(inputimg, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    # 图像的线性灰度变换
    img1 = np.empty((h, w), np.uint8)  # 创建空白数组
    img2 = np.empty((h, w), np.uint8)
    img3 = np.empty((h, w), np.uint8)
    img4 = np.empty((h, w), np.uint8)
    img5 = np.empty((h, w), np.uint8)
    img6 = np.empty((h, w), np.uint8)

    # Dt[i,j] = alfa*D[i,j] + beta
    alfa1, beta1 = 1, 50  # alfa=1,beta>0: 灰度值上移
    alfa2, beta2 = 1, -50  # alfa=1,beta<0: 灰度值下移
    alfa3, beta3 = 1.5, 0  # alfa>1,beta=0: 对比度增强
    alfa4, beta4 = 0.75, 0  # 0<alfa<1,beta=0: 对比度减小
    alfa5, beta5 = -0.5, 0  # alfa<0,beta=0: 暗区域变亮，亮区域变暗
    alfa6, beta6 = -1, 255  # alfa=-1,beta=255: 灰度值反转

    for i in range(h):
        for j in range(w):
            img1[i][j] = min(255, max((img[i][j] + beta1), 0))  # alfa=1,beta>0: 颜色发白
            img2[i][j] = min(255, max((img[i][j] + beta2), 0))  # alfa=1,beta<0: 颜色发黑
            img3[i][j] = min(255, max(alfa3 * img[i][j], 0))  # alfa>1,beta=0: 对比度增强
            img4[i][j] = min(255, max(alfa4 * img[i][j], 0))  # 0<alfa<1,beta=0: 对比度减小
            img5[i][j] = alfa5 * img[i][j] + beta5  # alfa<0,beta=255: 暗区域变亮，亮区域变暗
            img6[i][j] = min(255, max(alfa6 * img[i][j] + beta6, 0))  # alfa=-1,beta=255: 灰度值反转

    plt.figure(figsize=(10, 6))
    titleList = ["1. imgGray", "2. beta=50", "3. beta=-50", "4. alfa=1.5", "5. alfa=0.75", "6. alfa=-0.5"]
    imageList = [img, img1, img2, img3, img4, img5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.title(titleList[i]), plt.axis('off')
        plt.imshow(imageList[i], vmin=0, vmax=255, cmap='gray')
    plt.show()
    return img1, img2, img3, img4, img5, img6


def graypartline(inputimg):
    #分段线性灰度变换 (对比度拉伸)
    imgGray = cv2.imread(inputimg, cv2.IMREAD_GRAYSCALE)  # flags=0 读取为灰度图像
    height, width = imgGray.shape[:2]  # 图片的高度和宽度

    # constrast stretch, (r1,s1)=(rMin,0), (r2,s2)=(rMax,255)
    rMin = imgGray.min()  # 原始图像灰度的最小值
    rMax = imgGray.max()  # 原始图像灰度的最大值
    r1, s1 = rMin, 0  # (x1,y1)
    r2, s2 = rMax, 255  # (x2,y2)

    imgStretch = np.empty((width, height), np.uint8)  # 创建空白数组
    k1 = s1 / r1  # imgGray[h,w] < r1:
    k2 = (s2 - s1) / (r2 - r1)  # r1 <= imgGray[h,w] <= r2
    k3 = (255 - s2) / (255 - r2)  # imgGray[h,w] > r2
    for h in range(height):
        for w in range(width):
            if imgGray[h, w] < r1:
                imgStretch[h, w] = k1 * imgGray[h, w]
            elif r1 <= imgGray[h, w] <= r2:
                imgStretch[h, w] = k2 * (imgGray[h, w] - r1) + s1
            elif imgGray[h, w] > r2:
                imgStretch[h, w] = k3 * (imgGray[h, w] - r2) + s2

    plt.figure(figsize=(10, 3.5))
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=0.1, hspace=0.1)
    plt.subplot(131), plt.title("s=T(r)")
    x = [0, 96, 182, 255]
    y = [0, 30, 220, 255]
    plt.plot(x, y)
    plt.axis([0, 256, 0, 256])
    plt.text(105, 25, "(r1,s1)", fontsize=10)
    plt.text(120, 215, "(r2,s2)", fontsize=10)
    plt.xlabel("r, Input value")
    plt.ylabel("s, Output value")
    plt.subplot(132), plt.imshow(imgGray, cmap='gray', vmin=0, vmax=255), plt.title("Original"), plt.axis('off')
    plt.subplot(133), plt.imshow(imgStretch, cmap='gray', vmin=0, vmax=255), plt.title("Stretch"), plt.axis('off')
    plt.show()

    return imgStretch


if __name__ == "__main__":
    addimg, ave_img = addnoise("001.jpg")
    # cv2.imwrite('addnoise_img.jpg', addimg)
    # cv2.imwrite('ave_img.jpg', ave_img)
    # cv2.imwrite('Lena_noise.jpg', addimg) # 添加高斯噪声，后续作业

    # maskimg = masktransform("002.jpg", "002mask.png")
    # cv2.imwrite(PATH2 + 'maskimg.png', maskimg)

    # pingyiimg1, pingyiimg2 = pingyi("002.jpg", 50, 50)
    # cv2.imwrite(PATH2 + 'pingyiimg1.png', pingyiimg1)
    # cv2.imwrite(PATH2 + 'pingyiimg2.png', pingyiimg2)

    # grayline = grayline("girl_cap.jpg")
    # for i in range(6):
    #     cv2.imwrite(PATH3 + 'beta' +'%02d'%(i+1) + '.png', grayline[i])

    # graypart = graypartline("002.jpg")




