import cv2
import numpy as np
import matplotlib.pyplot as plt


# 添加椒盐噪声
def saltPepper(image, salt, pepper):
    height = image.shape[0]
    width = image.shape[1]
    pertotal = salt + pepper  # 总噪声占比
    noiseImage = image.copy()
    noiseNum = int(pertotal * height * width)
    for i in range(noiseNum):
        rows = np.random.randint(0, height - 1)     # 选取一个随机[low, high) 行列
        cols = np.random.randint(0, width - 1)
        if (np.random.randint(0, 100) < 0.5 * 100):
            noiseImage[rows, cols] = 255
        else:
            noiseImage[rows, cols] = 0

    # 查看噪声数量
    noise = np.asarray(noiseImage)
    salt_num = 0
    pe_num = 0
    for h in range(height):
        for w in range(width):
            if noise[h, w] == 0:
                salt_num += 1
            elif noise[h, w] == 255:
                pe_num += 1

    print(salt_num, pe_num)
    return noiseImage


# 添加均匀分布噪声
def Uniform_noise(img, mean, sigma):

    mean, sigma = 10, 100
    a = 2 * mean - np.sqrt(12 * sigma)  # a = -14.64
    b = 2 * mean + np.sqrt(12 * sigma)  # b = 54.64
    noiseUniform = np.random.uniform(a, b, img.shape)
    imgUniformNoise = img + noiseUniform
    imgUniformNoise = cv2.normalize(imgUniformNoise, None, 0, 255, cv2.NORM_MINMAX)  # 归一化为 [0,255]
    return imgUniformNoise


def Average(img):
    # 获取图片宽高信息
    height, width = img.shape[:2]
    average = np.zeros(img.shape)

    for i in range(height-1):
        for j in range(width-1):
            sumconv = (
                    int(img[i - 1][j - 1]) + int(img[i - 1][j]) + int(img[i - 1][j + 1]) +
                    int(img[i][j - 1]) +     int(img[i][j]) +     int(img[i][j + 1]) +
                    int(img[i + 1][j - 1]) + int(img[i + 1][j]) + int(img[i + 1][j + 1]))
            average[i][j] = np.uint8(sumconv / 9)
            j += 1
        i += 1

    return average


def Medain(img):
    # 获取图片宽高信息
    height, width = img.shape[:2]
    med = np.zeros(img.shape)

    for i in range(height - 1):
        for j in range(width - 1):
            num = list(range(9))
            num[0] = int(img[i - 1][j - 1])
            num[1] = int(img[i - 1][j])
            num[2] = int(img[i - 1][j + 1])
            num[3] = int(img[i][j - 1])
            num[4] = int(img[i][j])
            num[5] = int(img[i][j + 1])
            num[6] = int(img[i + 1][j - 1])
            num[7] = int(img[i + 1][j])
            num[8] = int(img[i + 1][j + 1])
            num.sort()
            med[i][j] = np.uint8(num[4])
            j += 1
        i += 1
    return med


# 显示函数
def matplotlib_multi_pic1(images):
    for i in range(len(images)):
        img = images[i]
        title = "(" + str(i + 1) + ")"
        # 行，列，索引
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title, fontsize=10)
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == "__main__":
    image = cv2.imread("girl_cap.jpg", cv2.IMREAD_GRAYSCALE)
    imageNoise = saltPepper(image, 0.05, 0.05)
    Uniform = Uniform_noise(image, 10, 100)
    # cv2.imwrite('uniform_noise.jpg', Uniform)
    average1 = Average(Uniform)
    median1 = Medain(Uniform)
    average2 = Average(imageNoise)
    median2 = Medain(imageNoise)
    plt.subplot(231), plt.imshow(Uniform, cmap='gray'), plt.title('Uniform noise', fontsize=10), plt.axis('off')
    plt.subplot(232), plt.imshow(average1, cmap='gray'), plt.title('average', fontsize=10), plt.axis('off')
    plt.subplot(233), plt.imshow(median1, cmap='gray'), plt.title('median', fontsize=10), plt.axis('off')
    plt.subplot(234), plt.imshow(imageNoise, cmap='gray'), plt.title('pepper salt noise', fontsize=10), plt.axis('off')
    plt.subplot(235), plt.imshow(average2, cmap='gray'), plt.title('average', fontsize=10), plt.axis('off')
    plt.subplot(236), plt.imshow(median2, cmap='gray'), plt.title('median', fontsize=10), plt.axis('off')
    # plt.savefig('math_two_filter-noise.jpg', dpi=500, bbox_inches='tight')

    # imageAver3 = cv2.blur(Uniform, (3, 3))
    # imageAver5 = cv2.blur(Uniform, (5, 5))
    # imageMedian3 = cv2.medianBlur(Uniform, 9)
    # imageMedian5 = cv2.medianBlur(Uniform, 25)
    # plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('image', fontsize=10), plt.axis('off')
    # plt.subplot(234), plt.imshow(Uniform, cmap='gray'), plt.title('imageNoise', fontsize=10), plt.axis('off')
    # plt.subplot(232), plt.imshow(imageAver3, cmap='gray'), plt.title('Aver3', fontsize=10), plt.axis('off')
    # plt.subplot(233), plt.imshow(imageAver5, cmap='gray'), plt.title('Aver5', fontsize=10), plt.axis('off')
    # plt.subplot(235), plt.imshow(imageMedian3, cmap='gray'), plt.title('Median3', fontsize=10), plt.axis('off')
    # plt.subplot(236), plt.imshow(imageMedian5, cmap='gray'), plt.title('Median5', fontsize=10), plt.axis('off')
    # plt.savefig('filtering_cv_Uniformnoise.jpg', dpi=500, bbox_inches='tight')

    # # 使用opencv自带函数blur进行领域平均去噪:椒盐噪声
    # imageAver3 = cv2.blur(imageNoise, (3, 3))
    # imageAver5 = cv2.blur(imageNoise, (5, 5))
    # imageMedian3 = cv2.medianBlur(imageNoise, 9)
    # imageMedian5 = cv2.medianBlur(imageNoise, 25)
    # # images = [image, imageNoise, imageAver3, imageAver5, imageMedian3, imageMedian5]
    # # matplotlib_multi_pic1(images)
    # plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('image', fontsize=10), plt.axis('off')
    # plt.subplot(234), plt.imshow(imageNoise, cmap='gray'), plt.title('imageNoise', fontsize=10), plt.axis('off')
    # plt.subplot(232), plt.imshow(imageAver3, cmap='gray'), plt.title('Aver3', fontsize=10), plt.axis('off')
    # plt.subplot(233), plt.imshow(imageAver5, cmap='gray'), plt.title('Aver5', fontsize=10), plt.axis('off')
    # plt.subplot(235), plt.imshow(imageMedian3, cmap='gray'), plt.title('Median3', fontsize=10), plt.axis('off')
    # plt.subplot(236), plt.imshow(imageMedian5, cmap='gray'), plt.title('Median5', fontsize=10), plt.axis('off')
    # plt.savefig('filtering_cv.jpg', dpi=500, bbox_inches='tight')
    plt.show()