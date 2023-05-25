import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torchvision
from PIL import Image
from random import randint

from scipy import optimize
from skimage import morphology


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
    imageMedian5 = cv2.medianBlur(image, 25)
    imageMedian7 = cv2.medianBlur(image, 49)
    mid_num = 7
    cv2.imwrite('./moer/new_moer{}-mid{}.png'.format(name, mid_num), imageMedian7)


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


def value_main(name):
    # 读取去噪后的图片，进行高斯处理一次
    image = cv2.imread('./moer/new_moer{}-mid5.png'.format(str(name)), cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 二值化，提取骨架，返回直线图
    skeleton1 = value2_bone(image, 120)
    skeleton2 = value2_bone(image)

    cv2.imwrite("./moer/cv-skeleton{}.png".format(str(name)), skeleton2)  # 保存骨架2提取后的图片
    cv2.imwrite("./moer/skeleton{}.png".format(str(name)), skeleton1)  # 保存骨架1提取后的图片 效果较好

    img = Image.open("./moer/skeleton{}.png".format(str(name)))
    img = imageresize(img, 700, 500)
    cv2.imwrite("./moer/new_skeleton{}.png".format(str(name)), img)  # 保存裁剪后图片


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
    # plt.subplot(211), plt.title('skeletion{}'.format(name), fontsize=10), plt.imshow(edges, 'gray')
    # plt.xticks([]), plt.yticks([])
    num = [75, 80, 98]  # hough参数直线选取点数  NO.1:75 NO.2:80 NO.3:98
    # hough transform   cv2.HoughLines标准变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, num[name - 1])
    zero = np.zeros(img.shape)
    plt.figure(1, figsize=(7, 5))
    plt.xlim(0, img.shape[1]), plt.ylim(0, img.shape[0])
    plt.title("img{}-Hough".format(name))
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
        # cv2.line(zero, (x1, y1), (x2, y2), (255, 0, 0), 1)
        # 绘制图像，两个端点，颜色，宽度
        xx1 = np.arange(0, img.shape[1], 0.01)  # 30和75要对应x0的两个端点，0.01为步长
        yy1 = A1 * xx1 + B1
        plt.plot(xx1, yy1)

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
        plt.plot(x_chui, y_chui, 'gray')  # 绘制垂线
        distances_out = calc_distance(A_true, B_true, k, b_chui)
        print(distances_out)  # 输出 距离
        distances.append(distances_out)

    print("直线间的平均距离：{}".format(np.mean(distances)))
    plt.show()


def index_points(points, idx):
    """
    Input:
        点云数据 points: input points data, [B, N, C]\n
        点云索引 idx: sample index data, [B, S] 示例索引数据，S为对应点的索引！
    Return:
        new_points:, indexed points data, [B, S, C];返回索引的点云数据
        返回新的点，如果idx为一个[B,D1……DN],则它会按照idx中的纬度结构将其提取成[B，D1……DN,C]
    """

    # 将所有最开始读取数据时的Tensor张量copy一份到device所指定的GPU上，之后的运算都在GPU上进行。
    B = points.shape[0]
    # 将点云数据的第一维B，赋值给B

    view_shape = list(idx.shape)
    # view_shape=[B,1]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # [1:]表示从1往后的所有进行切片，除了第0个数，len(view_shape)为2，所以list后面全为1
    # c++中可以写成：
    # for i in range(1, len(view_shape)):
    # view_shape[i] = 1

    repeat_shape = list(idx.shape)
    # repeat_shape=[B,S]，将idx的shape转化为list[B,S]
    repeat_shape[0] = 1
    # repeat_shape=[1,S]，第一维令为1

    # batch_indices = np.arange(B, dtype=np.int_).reshape(view_shape).repeat(repeat_shape)
    # .view(view_shape)=.view(B,1)
    # .repeat(repeat_shape)=.view(1,S)
    # torch.arange(B) 用于产生一个[0开始,到B)结束,(步长为step的Tensor张量, 并且可以设置Tensor的device和dtyp)
    # Tensor[0,1,2....,B-1]-->>View后变成列向量-->>repeat将列向量复制S次
    # batch_indices = [B,S]

    new_points = points[idx, :]
    # 从points当中取出每个batch_indices对应索引的数据点赋值为new_points = [B,S,N]
    # “:”意思为保留该维
    return new_points  # new_points = [S,N]


def readimg_tensor(img):
    # inimg = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    inimg = Image.open(img)
    tx = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((128, 128)),  # 《《《《《96->128 数据处理，影响处理速度
        torchvision.transforms.ToTensor()
    ])
    img = tx(inimg)
    line_idx = img.nonzero()
    # line_idx = line_idx.cpu().detach().numpy()  # tensor后，debug查看转为array

    return line_idx


def readimg(img):
    inimg = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    imgshape = inimg.shape
    line_idx = inimg.nonzero()
    n = len(line_idx[0])
    line = np.zeros([n, 2], dtype=int)
    line[:, 0] = line_idx[0]
    line[:, 1] = line_idx[1]

    return line, imgshape


def closest_point_sample(xy):
    """
    实现从点集中采样出一条直线中所有点
    :param xy: 当前需要进行采样的点集
    :return:
    line_a：直线的索引
    line_noa：当前剩余点的索引
    """
    ori_idx = randint(1, len(xy))   # 随机选取一个初始点
    # print(ori_idx)
    line_a = [ori_idx]  # 直线a初始化
    # line_a.append(ori_idx+1)
    print(line_a)

    line_all = list(range(0, len(xy)))    # 所有亮点的索引
    # print(line_all)

    line_no = filter(lambda x: x not in line_a, line_all)
    line_noa = list(line_no)    # 将直线a上的点剔除，
    print(line_noa)


    # 两个集合A（采样点）,B（原点集合） A与B最小距离的点，从B中转移到A中，实现A中点的最近点采样
    for i in range(1000):
        if len(line_noa) > 1:   # 当所有直线上的点被采样到后跳出循环
            distance = scipy.spatial.distance.cdist(xy[line_a], xy[line_noa], metric='euclidean')
            # # 两个集合的欧氏距离
            d_min = np.argsort(distance)[:, 0]      # 集合对应点的最小值列表的索引
            min_distance = np.sort(distance)[:, 0]      # 两个集合的最小值列表
            if min(min_distance) < 5:  # 添加判断，过大的距离认为不在此直线上
                mmin = np.argsort(min_distance)[0]     # 两个集合真正的最小值索引
                line_a.append(line_noa[d_min[mmin]])    # 更新line_a
                line_no = filter(lambda x: x not in line_a, line_all)
                line_noa = list(line_no)  # 将直线a上的点剔除，  （某个循环中被全选中了，造成空集）
            else:
                return line_a, line_noa
        else:
            return line_a, line_noa

    return line_a, line_noa


def print_line(line_nxy, num, path, n):
    """
    将一副图片所有直线分别写入到文件中
    :param line_nxy: 原始所有点的数据
    :param num: 要输出的直线条数
    :return: 0
    """
    line_n_xy = line_nxy
    for i in range(num):
        if (len(line_n_xy) / len(line_nxy)) > 0.01:     # 判断剩余点个数占比
            line_a_sample, line_no = closest_point_sample(line_n_xy)     # 采样模块，返回采样得到的直线点索引+剩余点
            # print(line)
            line_ = np.array(line_a_sample)  # list2numpy
            new_line = index_points(line_n_xy, line_)
            # print(new_line)

            with open("{}/img{}_line{}.txt".format(path, n, i + 1), "w") as file:
                # file.write("X\tY\n")  # 第一行
                for j in range(len(new_line)):
                    file.write("{}\t{}\n".format(new_line[j][0], new_line[j][1]))
                # file.close()
            line_n_xy = line_n_xy[line_no]  # 更新当前剩余点
        else:
            return


def read_write_line(path, n, shape):
    """
    读取直线数据，并绘制直线
    :param path: 单幅图片中的直线文件夹路径，
            n：图片的编号
    :return:
    """
    # 模型路径
    line_list = os.listdir(path)
    print(path)
    print(line_list)
    # 获取目录下的txt文件. 由于文件夹中仅有txt文件，因此两个list数据相同
    txt_name = [item for item in line_list if item.endswith('.txt') or item.endswith('.TXT')]
    print(txt_name)
    # obj_name = os.path.splitext(item)[0]
    # print(obj_name)
    line = []
    L = []  # 存储直线参数，0：斜率，1：截距
    plt.figure(1, figsize=(7, 5))
    plt.xlim(0, shape[1]), plt.ylim(0, shape[0])
    plt.title("img{}".format(n))
    for i in range(len(txt_name)):
        line_data = txt_strtonum_feed(path+"/"+txt_name[i])
        line.append(line_data)
        line_data = np.array(line_data)
        # 散点绘制
        y0 = line_data[:, 0]
        x0 = line_data[:, 1]
        plt.scatter(x0, y0, 1, "black")
        # 直线拟合与绘制
        A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]     # 优化拟合方法

        # p0 = np.array([1, 0])
        # r = leastsq(residuals, p0, args=(x0, y0))   # 最小二乘法拟合 参数：差值函数，拟合参数初始化，拟合数据
        # A1, B1 = r[0]   # 最小二乘法拟合参数

        x1 = np.arange(0, shape[1], 0.01)  # 30和75要对应x0的两个端点，0.01为步长
        y1 = A1 * x1 + B1
        plt.plot(x1, y1, label="{}: y={:.3f}x+{:.3f}".format(txt_name[i], A1, B1))
        L.append([A1, B1])
        print("L:{}".format(L))
        plt.legend(loc='upper right')   # 显示图例，label信息，右上角
    # plt.show()
    L = np.array(L)
    A = L[:, 0]
    B = L[:, 1]
    print(A)
    amin = np.argsort(A)
    # A_tr = A[amin[1:]]      # 将斜率小到大排序，剔除最小值（一般误差较大）
    # B_tr = B[amin[1:]]
    # A_tr = A[amin[1:-1]]  # 将斜率小到大排序，剔除最大值（一般误差较大）
    # B_tr = B[amin[1:-1]]
    A_tr = A[amin[:]]
    B_tr = B[amin[:]]
    b_sort = np.argsort(B_tr)   # 根据截距赋予直线次序
    B_true = B_tr[b_sort]
    A_true = A_tr[b_sort]
    k = -1/(np.mean(A_true))

    # plt.figure()
    # plt.xlim(0, 512), plt.ylim(0, 512)
    distances = []
    for i in range(10):
        b_chui = randint(0, shape[1])  # 随机选取一个截距
        x_chui = np.arange(0, shape[1], 0.01)  # 30和75要对应x0的两个端点，0.01为步长
        y_chui = k * x_chui + b_chui
        plt.plot(x_chui, y_chui, 'gray')    # 绘制垂线
        distances_out = calc_distance(A_true, B_true, k, b_chui)
        print(distances_out)  # 输出 距离
        distances.append(distances_out)

    print("直线间的平均距离：{}".format(np.mean(distances)))
    plt.show()
    return


def f_1(x, a, b):
    return a * x + b


# 直线拟合的残差计算
def residuals(p, x, y_):
    k, b = p
    return y_ - (k * x + b)


# 数值文本文件转换为双列表形式[[...],[...],[...]],即动态二维数组
# 然后将双列表形式通过numpy转换为数组矩阵形式
def txt_strtonum_feed(filename):
    data = []
    with open(filename, 'r') as f:  # with语句自动调用close()方法
        line_read = f.readline()
        while line_read:
            eachline = line_read.split()     # ##按行读取文本文件，每行数据以列表形式返回
            read_data = [ int(x) for x in eachline[0:7]]  # TopN概率字符转换为float型
            # lable = [ int(x) for x in eachline[-1]]  # lable转换为int型
            # read_data.append(lable[0])
            # read_data = list(map(float, eachline))
            data.append(read_data)
            line_read = f.readline()
        return data  # 返回数据为双列表形式


def linesample_main(name, savepoint=False):
    linenum = 10
    imgname = './moer/new_skeleton{}.png'.format(str(name))
    line_xy, img_shape = readimg(imgname)
    # line_idx = readimg_tensor(imgname)
    print(line_xy)
    print(line_xy.shape)

    folder_path = "./moer/img{}".format(name)  # 直线保存文件夹
    folder = os.path.exists(folder_path)
    if not folder:  # 判断文件夹是否存在，并创建。
        os.makedirs(folder_path)

    if savepoint:
        print_line(line_xy, linenum, folder_path, name)     # 采样+写入 新图片会覆盖采样点

    read_write_line(folder_path, name, img_shape)      # 读取点+绘制


if __name__ == "__main__":
    n = 3
    cv_main(n)
    value_main(n)
    hough = 0
    if hough:
        Hough(n)
    else:
        linesample_main(n, savepoint=False)