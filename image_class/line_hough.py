from random import randint

import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

import cv2

"""
霍夫变换检测直线，参数需要多次调整。。。。
"""


def three_ways():
    SAMPLE_NUM = 100
    print("您当前的样本数目为:", SAMPLE_NUM)

    # 先预设一个结果，假定拟合的结果为 y=-6x+10
    X = np.linspace(-10, 10, SAMPLE_NUM)
    a = -6
    b = 10
    Y = list(map(lambda x: a * x + b, X))
    print("标准答案为：y={}*x+{}".format(a, b))

    # 增加噪声，制造数据
    Y_noise = list(map(lambda y: y + np.random.randn()*10, Y))
    plt.scatter(X, Y_noise)
    plt.title("data to be fitted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    A = np.stack((X, np.ones(SAMPLE_NUM)), axis=1)  # shape=(SAMPLE_NUM,2)
    b = np.array(Y_noise).reshape((SAMPLE_NUM, 1))

    print("方法列表如下:"
          "1.最小二乘法 least square method "
          "2.常规方程法 Normal Equation "
          "3.线性回归法 Linear regression")
    method = int(input("请选择您的拟合方法: "))

    Y_predict=list()
    if method == 1:
        theta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        # theta=np.polyfit(X,Y_noise,deg=1) 也可以换此函数来实现拟合X和Y_noise,注意deg为x的最高次幂，线性模型y=ax+b中，x最高次幂为1.
        # theta=np.linalg.solve(A,b) 不推荐使用
        theta = theta.flatten()
        a_ = theta[0]
        b_ = theta[1]
        print("拟合结果为: y={:.4f}*x+{:.4f}".format(a_, b_))
        Y_predict = list(map(lambda x: a_ * x + b_, X))

    elif method == 2:
        AT = A.T
        A1 = np.matmul(AT, A)
        A2 = np.linalg.inv(A1)
        A3 = np.matmul(A2, AT)
        A4 = np.matmul(A3, b)
        A4 = A4.flatten()
        a_ = A4[0]
        b_ = A4[1]
        print("拟合结果为: y={:.4f}*x+{:.4f}".format(a_, b_))
        Y_predict=list(map(lambda x:a_*x+b_,X))

    elif method == 3:
        # 利用线性回归模型拟合数据，构建模型
        model = LinearRegression()
        X_normalized = np.stack((X, np.ones(SAMPLE_NUM)), axis=1)  # shape=(50,2)
        Y_noise_normalized = np.array(Y_noise).reshape((SAMPLE_NUM, 1))  #
        model.fit(X_normalized, Y_noise_normalized)
        # 利用已经拟合到的模型进行预测
        Y_predict = model.predict(X_normalized)
        # 求出线性模型y=ax+b中的a和b，确认是否和我们的设定是否一致
        a_ = model.coef_.flatten()[0]
        b_ = model.intercept_[0]
        print("拟合结果为: y={:.4f}*x+{:.4f}".format(a_, b_))

    else:
        print("请重新选择")

    plt.scatter(X, Y_noise)
    plt.plot(X, Y_predict, c='green')
    plt.title("method {}: y={:.4f}*x+{:.4f}".format(method, a_, b_))
    plt.show()


def calc_distance(slopes, intercepts, k, b):
    """
    计算每个相邻直线之间的距离
    :param slopes: 直线斜率列表
    :param intercepts: 直线截距列表
    K : 垂线斜率
    b ：垂线截距
    :return: 相邻直线之间的距离列表
    """
    d = []
    for i in range(len(slopes)-1):
        # 计算两相邻直线,与垂线的交点坐标
        x1 = (intercepts[i+1] - b) / (k - slopes[i+1])
        y1 = k * x1 + b
        x2 = (intercepts[i] - b) / (k - slopes[i])
        y2 = k * x2 + b
        # plt.scatter([x1, x2], [y1, y2], 5, "red")   # 绘制坐标散点，3：点size，颜色
        # p1 = np.array([[x1, y1]])
        # p2 = np.array([[x2, y2]])
        # 计算两点之间的距离
        d_p = np.sqrt( (y2-y1)**2 + (x2-x1)**2 )
        d.append(d_p)
    return np.mean(d)


def Hough(name):
    img = cv2.imread("./moer/new_skeleton{}.png".format(name), cv2.IMREAD_GRAYSCALE)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
    # open to see how to use: cv2.Canny
    # http://blog.csdn.net/on2way/article/details/46851451
    # edges = cv2.Canny(img, 50, 200)
    edges = img
    plt.subplot(211), plt.title('skeletion{}'.format(name), fontsize=10), plt.imshow(edges, 'gray')
    plt.xticks([]), plt.yticks([])
    num = [75, 80, 98]  # hough参数直线选取点数  NO.1:75 NO.2:80 NO.3:98
    # hough transform   cv2.HoughLines标准变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, num[name - 1])
    zero = np.zeros(img.shape)
    L = []  # 存储直线参数，0：斜率，1：截距
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        A1 = (y2 - y1) / (x2 - x1)
        B1 = y1 - A1 * x1
        cv2.line(zero, (x1, y1), (x2, y2), (255, 0, 0), 1)
        # 绘制图像，两个端点，颜色，宽度
        L.append([A1, B1])
    print(len(lines))

    L = np.array(L)
    A = L[:, 0]
    B = L[:, 1]
    print(A)
    print(B)
    amin = np.argsort(A)
    # A_tr = A[amin[1:]]      # 将斜率小到大排序，剔除最小值（一般误差较大）
    # B_tr = B[amin[1:]]
    # A_tr = A[amin[1:-1]]  # 将斜率小到大排序，剔除最大值（一般误差较大）
    # B_tr = B[amin[1:-1]]
    A_tr = A[amin[:]]  # 将斜率小到大排序，剔除最大值（一般误差较大）
    B_tr = B[amin[:]]
    b_sort = np.argsort(B_tr)  # 根据截距赋予直线次序
    B_true = B_tr[b_sort]
    A_true = A_tr[b_sort]
    k = -1 / (np.mean(A_true))

    # plt.figure()
    # plt.xlim(0, 512), plt.ylim(0, 512)
    distances = []
    for i in range(10):
        b_chui = randint(0, img.shape[1])  # 随机选取一个截距
        x_chui = np.arange(0, img.shape[1], 0.01)  # 30和75要对应x0的两个端点，0.01为步长
        y_chui = k * x_chui + b_chui
        # plt.plot(x_chui, y_chui, 'gray')  # 绘制垂线
        distances_out = calc_distance(A_true, B_true, k, b_chui)
        print(distances_out)  # 输出 距离
        distances.append(distances_out)

    print("直线间的平均距离：{}".format(np.mean(distances)))

    # # hough transform     cv2.HoughLinesP概率直线检测；
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=60, maxLineGap=10)
    # lines1 = lines[:, 0, :]  # 提取为二维
    # zero = np.zeros(img.shape)
    # for x1, y1, x2, y2 in lines1[:]:
    #     cv2.line(zero, (x1, y1), (x2, y2), (255, 0, 0), 1)
    # print(len(lines))

    plt.subplot(212), plt.imshow(zero), plt.title('hough', fontsize=10),
    plt.xticks([]), plt.yticks([])
    plt.savefig("./moer/Hough{}.jpg".format(n), dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    n = 3
    Hough(n)

