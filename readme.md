见：https://zhuanlan.zhihu.com/p/599033387













基于相关滤波的目标跟踪：MOSSE、SCK 、KCF
​

编辑

切换为居中
添加图片注释，不超过 140 字（可选）
1、何为相关滤波？
a）相关
首先相关滤波相关滤波，相关是什么？
相关： g=f⊗h ，其中h为相关核
具体到每个像素： g(i,j)=∑k,lf(i+k,j+l)h(k,l) 
​

编辑

切换为全宽
添加图片注释，不超过 140 字（可选）
例: 计算输出图像  g(3,3)  像素值为：
​

编辑
添加图片注释，不超过 140 字（可选）
卷积： g=f⋆h ，其中h为卷积核
具体到每个像素： g(i,j)=∑k,lf(i−k,j−l)h(k,l)=∑k,lf(k,l)h(i−k,j−l) 
​

编辑

切换为全宽
添加图片注释，不超过 140 字（可选）
例：计算输出图像g(3,3)的像素值为：
​

编辑
添加图片注释，不超过 140 字（可选）
区别： 
（1）卷积将核旋转了180度。
（2）物理意义：相关可以反应两个信号相似程度，卷积不可以。
（3）卷积满足交换律，相关不可以。
（4）卷积可以直接通过卷积定理（时域上的卷积等于频域上的乘积）来加速运算，相关不可以。
相关和卷积类似，区别在于公式里的共轭。所以有人说：相关滤波的本质就是一个尺寸特别大（跟patch一样大）的cnn卷积核。 
因为卷积公式可以加速运算，所以这里基于相关滤波的算法都是通过快速傅里叶卷积（FFT）转化到频域里面计算的。
b）图像的‘频率‘
我的理解是：图像上的滤波操作，就是对图片进行傅里叶变换，转换到频域后，对这个频域内的”图片“进行操作，然后再傅里叶变换回去。之前正好整理过。
图像的频率是表征图像中灰度变化剧烈程度的指标，是灰度在平面空间上的梯度。如：大面积的沙漠在图像中是一片灰度变化缓慢的区域，对应的频率值很低；而对于地表属性变换剧烈的边缘区域在图像中是一片灰度变化剧烈的区域，对应的频率值较高。傅里叶变换在实际中有非常明显的物理意义，设f是一个能量有限的模拟信号，则其傅里叶变换就表示f的频谱。从纯粹的数学意义上看，傅里叶变换是将一个函数转换为一系列周期函数来处理的。从物理效果看，傅里叶变换是将图像从空间域转换到频率域，其逆变换是将图像从频率域转换到空间域。换句话说，傅里叶变换的物理意义是将图像的灰度分布函数变换为图像的频率分布函数。
然后有一个很重要的卷积公式：时域上的卷积等于频域上的乘积
故，将图像和滤波器通过算法变换到频域后，直接相乘（矩阵逐元素相乘），然后再变换回时域（也就是图像的空间域）
c）相关滤波的思想及实际意义
相关滤波的思想就是学习一个滤波器h，然后与图像f进行卷积，得到相关信息图，图中值最大的点就是物体的位置
相关滤波的实际意义：把输入图像映射到一个理想响应图，这个响应图当中的最高峰与目标中心点对应起来。至于映射的实现就是通过一个滤波器对图像进行相关操作。那么相关到底是在干什么呢？其实相关操作的每一个输出点都将滤波器与输入图像当中的一个大小相当的图像块中元素的点乘累加。也就是说完成了从图像块到响应点的映射，而目标跟踪当中的目标就是以图像块的形式存在的。 
d）优点
相关滤波的一个优点是能够借助于傅里叶变换，快速计算大量的候选样本的响应值。
2、相关滤波跟踪的原理
简单的说：相关滤波算法就是建立一个相关滤波器， 使其与目标的相关响应最大。该算法最大的特点就是速度快， 这是其他方法无法比拟的。
相关滤波跟踪的基本思想就是，设计一个滤波模板，利用该模板与目标候选区域做相关运算，最大输出响应的位置即为当前帧的目标位置。
一个公式表示就是：  
其中表示响应输出，表示输入图像，表示滤波模板。利用相关定理，将相关转换为计算量更小的点积。
3、MOSSE算法
参考经典论文 MOSSE （Minimum Output Sum of Squared Error filter）这篇CVPR2010年的论文可以说是相关滤波的开山之作吧。首次将CF引入到图像中。
下载地址：https://www.cs.colostate.edu/~vision/publications/bolme_cvpr10.pdf
MOSSE算法跟踪效果：
​

编辑
添加图片注释，不超过 140 字（可选）
1）概要
目标：找到一个点的坐标，由这个点的坐标加上宽高，就可以得到跟踪的区域。
那这个区域怎么找？
整个相关滤波算法围绕的核心公式：   其中G是想要预测得到的跟踪图，F是经帧图像经FFT变换后的，H就是需要找到的滤波器模板。G是一个经过傅里叶变换后的图，然后再IFFT，逆傅里叶变换，得到g，g是什么？g的样子大概像下面这张图一样，一张三维的图，xyz轴，xy为坐标值，所以z轴中最大值位置即为目标所在位置。
​

编辑
添加图片注释，不超过 140 字（可选）
那 怎么找？
在MOSSE中通过最小化实际输出的卷积和期望输出卷积之间的方差来得到合适的滤波器，即求解  
这里给出结论，相关推理可以看MOSSE原文
推导得到的解是：  
为了增加鲁棒性，加速得到的结论：  
上面是理论，经过实验，在实际操作中:      （1）
其中η 为学习率。加入η 可以使模型更加重视最近的帧，并使先前的帧的效果随时间呈指数衰减。文中给出的实验过程中得出的最优η =0.125，可以使滤波器快速适应目标外观的变化，同时仍保持好的稳健性。 
所以，MOSSE核心思路就是先根据第一帧确定一个H，然后后面不断的更新这个H。
2）mosse算法步骤：
准备工作 ：图像预处理。
用log函数对像素值进行处理，降低对比度（contrasting lighting situation）。
进行平均值为0，范数为1的归一化。
用余弦窗口进行滤波，降低边缘的像素值。
step1：读取第一帧图像，做归一化等预处理。根据目标检测还是其他手段，给出你要跟踪的Ground truth，即（x,y,w,h)，分别为（坐标x，坐标y，目标框的宽，目标框的高）。
step2：截取出这个框得到目标区域记为  ，对进行随机重定位(rand_wrap)，对进行preprocessing(取对数，标准化，通过窗函数，对进行傅里叶变换得到  
step3：计算出第一版的  ：根据第一帧给出的ground truth的w,h，可以得到一个二维高斯函数图  ，之后再傅里叶变化得到 。 根据和可以得到  和  。最后  。
前面三步，就是得到一个初始化的过程，初步获得一个过滤器的模板  。
step4：针对当前帧，用前一个目标框的中心截取一个框，获取当前帧的，并对其进行预处理和FFT，得到  。
step5：然后根据这个和前面计算得到的 ，由公式计算得到G。然后G进行IFFT，逆傅里叶变化，得到  。取gi中最大值的index，这个位置就是第二帧图像中目标所在，即(x_c, y_c)
step6：更新  ：根据(x_c, y_c)更新当前帧的  ，并对其进行预处理和fft，得到  。然后再由和这里新的，由公式一，更新。
之后就重复这三步，直到最后一帧。可以这么理解，用上一帧的目标坐标和上一帧更新的滤波器h，在新一帧的图像上获得一个新的目标坐标。然后再用新的目标坐标更新这个滤波器h。
可以参考画的步骤图：
​

编辑

切换为居中
添加图片注释，不超过 140 字（可选）
可以参考网络上一个画的很好的流程图： 
​

编辑

切换为居中
添加图片注释，不超过 140 字（可选）

4、SCK算法
1）概要 
从MOSSE到CSK算是了解相关滤波跟踪的必经之路。CSK在MOSSE模型的基础上增加了正则项以避免过拟合，提出了分类器的概念，同时引入了循环矩阵和核函数提高运算速率。
CSK这篇是主要引进了循环矩阵生成样本，使用相关滤波器进行跟踪。而KCF引进了多通道特征，可以使用比着灰度特征更好的HOG（梯度颜色直方图）特征或者其他的颜色特征等。
ECCV2012_CSK论文下载地址：https://www.robots.ox.ac.uk/~joao/publications/henriques_eccv2012.pdf
2）SCK算法
CSK算法是对于MOSSE算法的升级和拓展，在MOSSE上引入了核技巧以及岭回归方法。
前进一：目标是什么？
MOSSE将相关转到频域通过最小二乘法来求解相关滤波器H*：
​

添加图片注释，不超过 140 字（可选）
CSK则提出用一个线性分类器来求解相关滤波器W：
​

编辑
添加图片注释，不超过 140 字（可选）
上面这个公式也叫最小二乘法，确切的名字是正则化最小二乘法（RLS），也叫做岭回归。
其中y是理想的高斯响应，n表示样本数量，f(xi)表示图像xi与滤波器w的在频域内的点积，w即为MOSSE中的相关滤波器H。L为最小二乘法的损失函数：  ，并且  。<>和  一个意思，所以本质上，  和  是一个东西。那么CSK所用的公式就只是在后面多了一个正则项  。
加正则项的目的是为了防止求得的滤波器H过拟合。通过最小二乘法求得的相关滤波器与当前帧输入的图像F的相关性是最高的，然而我们是要用求得的滤波器结果H去预测下一帧图像中目标所在的位置。因为下一帧图像不可能和当前帧的图像一模一样，所以拟合度过高反而会影响检测的准确度，所以加入正则项来减小输入图像和滤波器之间的拟合程度，使求得的滤波器H泛化能力更强。 
前进二：CSK里求解这个分类器，参考了支持向量机的解法，使用了核函数。
引入的核函数：  
根据SVM，W可以表示为样本的线性组合:  
详细推导过程可参考博客：（写的很清楚）
最后得出结论，得到：  
其中K是核函数构成的矩阵，  是矩阵K的第i行第j列。  是引入正则化的系数，  是单位矩阵，y是理想的高斯响应函数。
前进三：引入循环矩阵求解  
MOSSE算法里求相关性的操作是通过在怀疑区域里滑动卷积得到的，如果是要求密集采样（而不是随机采样）的话，要求卷积模板在这一区域从第一个像素滑动到最后一个像素，在这一区域产生大量候选窗（这些候选窗重合度还很大），最终目标匹配得到最后的输出结果。这一过程明显计算量巨大，影响跟踪速度。直接利用循环矩阵去和滤波器点乘很直观地省略了卷积模板在检测区域内滑动的过程，简化了对重合度高的候选窗的运算，提高了速度。
关于这里的循环矩阵的理解：
​

编辑

切换为全宽
添加图片注释，不超过 140 字（可选）
​

编辑

切换为全宽
添加图片注释，不超过 140 字（可选）
这里，u可以理解为原始图像，第一行代表原始的u。后面的每一行都循环移位，循环矩阵的每一行都是上一行最后一列的元素或矢量移动到第一列。可以想象一下，把图像提取的特征看成一个1xn的向量，然后这个向量构成这个矩阵的第一行，第二行的元素就是将第一行最后一个元素移到首位。循环矩阵有一个特性：循环矩阵的和、点积、求逆都是循环矩阵。
循环矩阵的好处可以参考这篇博客：
总之最后可以得到结论：  
这个结论的推导过程可见这篇博客：
前进四：推理定位过程
定位结论：  
核函数有很多种，本文使用高斯核函数，可以将数据映射到无穷维，也叫做径向基函数（Radial Basis Function 简称 RBF）：  
3）流程图：
整个流程结构和MOSSE差不多，主要变化无非两点。一、在图像采样时多做了步骤，采用循环采样，这样计算推理就更快了些。二、滤波器更新那里，考虑到正则化的问题，使得滤波器不容易产生过拟合的问题。
可以参考网络上一个画的很好的流程图： 
​

编辑

切换为居中
添加图片注释，不超过 140 字（可选）
5、KCF算法
1）概要
KCF全称为核相关滤波算法（Kernel Correlation Filter）注：不是KFC。那么这个肯德基算法是在CSK算法的基础上而来的，CSK这篇是主要引进了循环矩阵生成样本，使用相关滤波器进行跟踪。而KCF引进了多通道特征，可以使用比着灰度特征更好的HOG（梯度颜色直方图）特征或者其他的颜色特征等。
KCF论文下载地址：https://arxiv.org/abs/1404.7584
该算法有3个特点：
使用目标周围区域的循环矩阵采集正负样本，利用脊回归（岭回归）训练目标检测器，并成功的利用循环矩阵在傅里叶空间可对角化的性质将矩阵的运算转化为向量的Hadamad积，即元素的点乘，大大降低了运算量，提高了运算速度，使算法满足实时性要求。
将线性空间的脊回归通过核函数映射到非线性空间，在非线性空间通过求解一个对偶问题和某些常见的约束，同样的可以使用循环矩阵傅里叶空间对角化简化计算。
给出了一种将多通道数据融入该算法的途径。
2）KCF算法之继承与派生
KCF算法继承了SCK算法的两个特点，核技巧与岭回归。这里可以详细参考上面的CSK。
KCF算法派生的一个特点，使用比着灰度特征更好的HOG（梯度颜色直方图）特征。
那么HOG特征是什么？
HOG特征全称为Histogram of Oriented Gradients，即方向梯度直方图。这是2005年CVPR上的一篇文章。HOG特征通过计算和统计图像局部区域的梯度方向直方图来构成特征。通过提取有用的信息并扔掉多余的信息来简化图像。HOG特征将一张大小为（width×height×channels）的图像转化为一个长度为n的特征向量，例如输入图像大小为64 × 128 × 3 ，HOG特征的输出为长度3780的向量。为什么是3780，后面会介绍。
前进一：图像预处理
伽马校正：减少光度对实验的影响
灰度化：将彩色图片变成灰度图。
彩色图片也可以直接处理。不过是分别对三通道的颜色值进行梯度计算，最后选择梯度最大的那个。
预处理的操作要求图片保持1:2的横纵比
前进二：梯度图
​

编辑

切换为全宽
添加图片注释，不超过 140 字（可选）
对于像素点A，要计算水平梯度和竖直梯度，如上图，水平梯度=30-20=10,竖直梯度=64-32=32。
那么总的梯度强度值g和梯度方向  （梯度方向将会取绝对值）将按照以下公式计算：
每一个像素点都会有两个值：梯度强度/梯度方向。
前进三：计算梯度直方图
将图像分割成多个8×8的cell，在这些cell中计算梯度直方图。根据前进二的操作，可以得到一个梯度强度图和一个梯度方向图，将0-180度分成9个区间：0,20,40,…160，之后统计每个像素点所在的区间——将这个区间命名为bin，采取的原则是对每个像素点处的g值，按θ的比例将g分配给相邻的bin。如下图：
​

编辑
两个蓝色圈圈。因为蓝圈的方向是80度，大小是2，所以该点就投给80这个bin；两个红色圈圈。因为红色圈圈的方向是10，大小是4，因为10距离0点为10，距离20点为也为10，那么有一半的大小是投给0这个bin，还有一半的大小（即是2）投给20这个bin。
统计完64个点的投票数以后，每个bin就会得到一个数值，可以得到一个直方图：
​

编辑
添加图片注释，不超过 140 字（可选）
前进四：对16x16大小的Block进行归一化
标准化（Normalization）也称归一化，即将每个向量的分量除以向量的模长。Block选取示意图如下：
​

编辑
添加图片注释，不超过 140 字（可选）
前进五：得到HOG特征向量
每一个16*16大小的block将会得到36大小的vector。那么对于一个64*128大小的图像，按照上图的方式提取block，将会有7个水平位置和15个竖直位可以取得，所以一共有7*15=105个block，所以我们整合所有block的vector，形成一个大的一维vector的大小将会是36*105=3780。
前进六：HOG特征python的实现
import cv2
import numpy as np


class HOG:
    def __init__(self, winSize):
        """

        :param winSize: 检测窗口的大小
        """
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nBins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nBins)

    def get_feature(self, image):
        winStride = self.winSize
        w, h = self.winSize
        w_block, h_block = self.blockStride#w_block=4
        w = w//w_block - 1
        h = h//h_block - 1
        # 计算给定图像的HOG特征描述子，一个n*1的特征向量
        hist = self.hog.compute(img=image, winStride=winStride, padding=(0, 0))
        return hist.reshape(w, h, 36).transpose(2, 1, 0)    # 交换轴的顺序

    def show_hog(self, hog_feature):
        c, h, w = hog_feature.shape
        feature = hog_feature.reshape(2, 2, 9, h, w).sum(axis=(0, 1))
        grid = 16
        hgrid = grid // 2
        img = np.zeros((h*grid, w*grid))

        for i in range(h):
            for j in range(w):
                for k in range(9):
                    x = int(10 * feature[k, i, j] * np.cos(x=np.pi / 9 * k))
                    y = int(10 * feature[k, i, j] * np.sin(x=np.pi / 9 * k))
                    cv2.rectangle(img=img, pt1=(j*grid, i*grid), pt2=((j + 1) * grid, (i + 1) * grid),
                                  color=(255, 255, 255))
                    x1 = j * grid + hgrid - x
                    y1 = i * grid + hgrid - y
                    x2 = j * grid + hgrid + x
                    y2 = i * grid + hgrid + y
                    cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=1)
        cv2.imshow("img", img)
        cv2.waitKey(0)


def tesHog():
    img = cv2.imread("C:/Users/xyx/Desktop/tracking_study/KCF/data/football/00000007.jpg")
    img=cv2.resize(img,(64,128))
    # cv2.imshow("img1", img[:,
    # print(img.shape) :, 0])
    hog = HOG((64, 128))

    feature = hog.get_feature(img)
    print(feature.shape)

# tesHog()
3）KCF算法流程图：
可以发现，和CSK算法的流程图一样，毕竟只是用HOG代替了原来的灰度图特征。
​

编辑

切换为居中
添加图片注释，不超过 140 字（可选）
KCF滤波python实现：
import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from HOG import HOG

class Tracker:
    def __init__(self):
        # 超参数设置
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.gray_feature = False
        self.debug = False

        # 算法变量定义
        self.scale_h = 0.
        self.scale_w = 0.

        self.ph = 0
        self.pw = 0
        self.hog = HOG((self.pw, self.pw))
        self.alphaf = None
        self.x = None
        self.roi = None

    def first_frame(self, image, roi):
        """
        对视频的第一帧进行标记，更新tracer的参数
        :param image: 第一帧图像
        :param roi: 第一帧图像的初始ROI元组
        :return: None
        """
        x1, y1, w, h = roi #目标坐标
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)#转为中心点坐标

        # 确定Patch的大小，并在此Patch中提取HOG特征描述子
        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))

        # 在矩形框的中心采样、提取特征
        x = self.get_feature(image, roi)
        y = self.gaussian_peak(x.shape[2], x.shape[1])

        self.alphaf = self.train(x, y, self.sigma, self.lambdar)
        self.x = x
        self.roi = roi

    def update(self, image):
        """
        对给定的图像，重新计算其目标的位置
        :param image:
        :return:
        """
        # 包含矩形框信息的四元组(min_x, min_y, w, h)
        cx, cy, w, h = self.roi
        max_response = -1   # 最大响应值

        for scale in [0.95, 1.0, 1.05]:
            # 将ROI值处理为整数
            roi = map(int, (cx, cy, w * scale, h * scale))

            z = self.get_feature(image, roi)    # tuple(36, h, w)
            # 计算响应
            responses = self.detect(self.x, z, self.sigma)
            height, width = responses.shape
            if self.debug:
                cv2.imshow("res", responses)
                cv2.waitKey(0)
            idx = np.argmax(responses)
            res = np.max(responses)
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z

        # 更新矩形框的相关参数
        self.roi = (cx + dx, cy + dy, best_w, best_h)

        # 更新模板
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_peak(best_z.shape[2], best_z.shape[1])
        new_alphaf = self.train(best_z, y, self.sigma, self.lambdar)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate

        cx, cy, w, h = self.roi
        return cx - w // 2, cy - h // 2, w, h

    def get_feature(self, image, roi):
        """
        对特征进行采样
        :param image:
        :param roi: 包含矩形框信息的四元组(min_x, min_y, w, h)
        :return:
        """
        # 对矩形框做2.5倍的Padding处理
        cx, cy, w, h = roi
        w = int(w*self.padding)//2*2
        h = int(h*self.padding)//2*2
        x = int(cx - w//2)
        y = int(cy - h//2)

        # 矩形框所覆盖的距离
        sub_img = image[y:y+h, x:x+w, :]
        resized_img = cv2.resize(src=sub_img, dsize=(self.pw, self.ph))

        if self.gray_feature:
            feature = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(1, self.ph, self.pw)/255.0 - 0.5
        else:
            feature = self.hog.get_feature(resized_img)
            if self.debug:
                self.hog.show_hog(feature)

        # Hog特征的通道数、高度、宽度
        fc, fh, fw = feature.shape
        self.scale_h = float(fh)/h
        self.scale_w = float(fw)/w

        # 两个二维数组，前者(fh，1)，后者(1，fw)
        hann2t, hann1t = np.ogrid[0:fh, 0:fw]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))

        # 一个fh x fw的矩阵
        hann2d = hann2t * hann1t

        feature = feature * hann2d
        return feature

    def gaussian_peak(self, w, h):
        """

        :param w:
        :param h:
        :return:      一个w*h的高斯矩阵
        """
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h//2, w//2

        y, x = np.mgrid[-syh:-syh + h, -sxh:-sxh + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g

    def kernel_correlation(self, x1, x2, sigma):
        """
        核化的相关滤波操作
        :param x1:
        :param x2:
        :param sigma:   高斯参数sigma
        :return:
        """
        # 转换到傅里叶空间
        fx1 = fft2(x1)
        fx2 = fft2(x2)
        # \hat{x^*} \otimes \hat{x}'，x*的共轭转置与x'的乘积
        tmp = np.conj(fx1) * fx2
        # 离散傅里叶逆变换转换回真实空间
        idft_rbf = ifft2(np.sum(tmp, axis=0))
        # 将零频率分量移到频谱中心。
        idft_rbf = fftshift(idft_rbf)

        # 高斯核的径向基函数
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * idft_rbf
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k

    def train(self, x, y, sigma, lambdar):
        """
        原文所给参考train函数
        :param x:
        :param y:
        :param sigma:
        :param lambdar:
        :return:
        """
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)

    def detect(self, x, z, sigma):
        """
        原文所给参考detect函数
        :param x:
        :param z:
        :param sigma:
        :return:
        """
        k = self.kernel_correlation(x, z, sigma)
        # 傅里叶逆变换的实部
        return np.real(ifft2(self.alphaf * fft2(k)))
