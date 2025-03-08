"""
Paper : A generalizable framework for low-rank tensor completion with numerical priors
Source : Pattern Recognition 2024
Link : https://doi.org/10.1016/j.patcog.2024.110678
Code-Author : Liu Qian
"""
import time
import numpy as np
import pandas as pd
from numba import jit
from numba.typed import List


@jit(nopython=True)
def _getPredict(FM, i, j, k):
    return np.sum(FM[0][i, :] * FM[1][j, :] * FM[2][k, :])


@jit(nopython=True)
def _getMetrics(FM, test_df):
    RMSE = 0
    MAE = 0
    count = 0
    for row in test_df:  # for each y
        i, j, k, y = int(row[0]), int(row[1]), int(row[2]), row[3]
        y_hat = _getPredict(FM, i, j, k)
        RMSE = RMSE + pow((y - y_hat), 2)
        MAE = MAE + abs(y - y_hat)
        count = count + 1
    MAE = MAE / count
    RMSE = np.sqrt(RMSE / count)
    return MAE, RMSE


@jit(nopython=True)
def _update(FM, lr, rho, train):
    # 创建梯度累积器
    grad0 = np.zeros_like(FM[0])
    grad1 = np.zeros_like(FM[1])
    grad2 = np.zeros_like(FM[2])
    # 遍历所有训练样本
    for row in train:
        i, j, k, y = int(row[0]), int(row[1]), int(row[2]), row[3]
        pred = _getPredict(FM, i, j, k)
        # 计算泊松梯度分量
        grad_coeff = (pred - y) / (pred + 1e-10)
        # 计算各因子矩阵的梯度分量
        grad0[i, :] += grad_coeff * (FM[1][j, :] * FM[2][k, :])
        grad1[j, :] += grad_coeff * (FM[0][i, :] * FM[2][k, :])
        grad2[k, :] += grad_coeff * (FM[0][i, :] * FM[1][j, :])

    # 添加平滑项梯度
    for mat_idx in range(3):
        mat = FM[mat_idx]
        smooth_grad = np.zeros_like(mat)
        # 第一行梯度
        smooth_grad[0] = rho * (mat[0] - mat[1])
        # 中间行梯度
        for j in range(1, mat.shape[0] - 1):
            smooth_grad[j] = rho * (2 * mat[j] - mat[j - 1] - mat[j + 1])
        # 最后一行梯度
        smooth_grad[-1] = rho * (mat[-1] - mat[-2])
        # 累加平滑梯度
        if mat_idx == 0:
            grad0 += smooth_grad
        elif mat_idx == 1:
            grad1 += smooth_grad
        else:
            grad2 += smooth_grad
    # 更新后梯度
    grad0 = np.maximum(FM[0] - lr * grad0, 0)
    grad1 = np.maximum(FM[1] - lr * grad1, 0)
    grad2 = np.maximum(FM[2] - lr * grad2, 0)

    return grad0, grad1, grad2


class SPTC():
    def __init__(self, epoch, R, lr, rho):
        '''
        SPCT model
        :param epoch: 总训练轮数
        :param R: 秩
        :param lr: 学习率
        :param rho: 平滑项系数
        '''
        self.epoch = epoch
        self.R = R
        self.lr = lr
        self.rho = rho
        self.train = None
        self.val = None
        self.test = None
        self.shape = None  # 目标张量shape
        self.scale = None  # 初始化隐特征矩阵缩放系数
        self.FM = []  # 隐特征矩阵

        self.need_metrics = ['train', 'val', 'test']  # 每论训练后将要测试的指标
        self.errorList = []  # MAE RMSE历史列表
        self.minValError = [[0, np.finfo(np.float32).max], [0, np.finfo(np.float32).max]]

    def initData(self, shape, train, val, test, scale):
        '''
        初始化数据集
        '''
        self.train = train.values
        self.val = val.values
        self.test = test.values
        self.shape = shape
        self.scale = scale

    def initFromFlie(self, csvPath, frac, sep=',', scale=0.5):
        '''
        从单个文件初始化数据集
        :param csvPath: 文件路径
        :param frac: 数据集划分,eg.[0.7,0.1,0.2]
        :param sep: 数据集中分割符形式
        :param scale: 隐特征矩阵随机初始化范围[0-scale]
        '''
        dataShape, train, val, test = self._split_data(csvPath, frac, sep=sep)
        self.initData(dataShape, train, val, test, scale)

    # 初始化参数
    def _initParams(self):
        self.FM.append(np.random.uniform(0, self.scale, size=(self.shape[0], self.R)))
        self.FM.append(np.random.uniform(0, self.scale, size=(self.shape[1], self.R)))
        self.FM.append(np.random.uniform(0, self.scale, size=(self.shape[2], self.R)))
        self.errorList = []  # MAE RMSE
        self.minValError = [[0, np.finfo(np.float32).max], [0, np.finfo(np.float32).max]]

    def decompose(self, stopError, patience=5, minEpoch=10, print_info=True, info_to_file=True):
        '''
        执行训练
        :param stopError: 早停阈值
        :param patience: 早停耐心值，即几次早停
        :param minEpoch: 最小训练轮数
        :param print_info: 训练时是否打印精度信息
        :param info_to_file: 训练结束是否输出训练过程到文件
        :return: [终止轮数，训练时间]
        '''
        self._initParams()
        stop_times = patience
        if print_info:
            print(f'train_size :{self.train.shape[0]} val_size :{self.val.shape[0]} test_size :{self.test.shape[0]}')
        start = time.time()
        i = 0
        while i < self.epoch:
            if print_info:
                print(f'\nepoch :{i + 1}')
            grad0, grad1, grad2 = _update(FM=List(self.FM),
                                          lr=self.lr,
                                          rho=self.rho,
                                          train=self.train)
            self.FM[0] = grad0
            self.FM[1] = grad1
            self.FM[2] = grad2
            stop_flag = self._afterPerUpdate(i, stopError, print_info)
            i = i + 1
            # 早停
            if stop_flag:
                stop_times -= 1
                if stop_times < 1 and i > minEpoch:
                    break
            else:
                stop_times = patience
        totalTime = time.time() - start
        if info_to_file:
            self._train_info_to_file(i, totalTime)
        return i, totalTime

    def getBestTestMetrics(self):
        '''
        获取测试集最近时的迭代轮数和对应精度
        :return: [epoch minMAE epoch minRMSE]
        '''
        return [self.minValError[0][0] + 1, self.errorList[self.minValError[0][0]][4],
                self.minValError[1][0] + 1, self.errorList[self.minValError[1][0]][5]]

    def _afterPerUpdate(self, epoch, stopError, print_info):
        '''
        内部函数，用于计算每次更新后的指标 MAE/RMSE
        '''
        flag = False  # 早停标识
        trainMAE, trainRMSE, vMAE, vRMSE, tMAE, tRMSE = 0, 0, 0, 0, 0, 0
        if 'train' in self.need_metrics:
            trainMAE, trainRMSE = _getMetrics(List(self.FM), self.train)
            if print_info:
                print(f'\ntrain Metrics: MAE = {trainMAE}; RMSE = {trainRMSE};')
        if 'val' in self.need_metrics:
            vMAE, vRMSE = _getMetrics(List(self.FM), self.val)
            if print_info:
                print(f'val Metrics: MAE = {vMAE}; RMSE = {vRMSE};')
        if 'test' in self.need_metrics:
            tMAE, tRMSE = _getMetrics(List(self.FM), self.test)
            if print_info:
                print(f'test Metrics: MAE = {tMAE}; RMSE = {tRMSE};')

        if epoch > 0:
            if ((stopError > np.abs(self.errorList[epoch - 1][2] - vMAE)) and
                    (stopError > np.abs(self.errorList[epoch - 1][3] - vRMSE))):
                flag = True
        self.errorList.append([trainMAE, trainRMSE, vMAE, vRMSE, tMAE, tRMSE])
        if vMAE < self.minValError[0][1]:
            self.minValError[0] = [epoch, vMAE]
        if vRMSE < self.minValError[1][1]:
            self.minValError[1] = [epoch, vRMSE]
        return flag

    def _train_info_to_file(self, epoch, totalTime):
        '''
        内部函数，用于将训练结果输出到文件
        '''
        with open(f'{self.__class__.__name__}_{id(self)}.log', 'a+') as f:
            for item in self.errorList:
                f.writelines(f'{item[0]} {item[1]} {item[2]} {item[3]} {item[4]} {item[5]}\n')
            epochMAE, metricsMAE, epochRMSE, metricsRMSE = self.getBestTestMetrics()
            f.writelines(f"{epochMAE} {metricsMAE}\n")
            f.writelines(f"{epochRMSE} {metricsRMSE}\n")
            f.writelines(f"{epoch} {totalTime}\n")

    def _split_data(self, csvPath, frac, sep=','):
        '''
        用于分割数据集
        '''
        dataFrame = pd.read_csv(csvPath, sep=sep, header=None)
        dataShape = [0, 0, 0]
        for row in dataFrame.itertuples():  # 获取范围
            i, j, k = row[1], row[2], row[3]
            if i > dataShape[0]:
                dataShape[0] = i
            if j > dataShape[1]:
                dataShape[1] = j
            if k > dataShape[2]:
                dataShape[2] = k
        dataShape = [int(item + 1) for item in dataShape]
        dataFrame = dataFrame.iloc[np.random.permutation(dataFrame.index)].reset_index(drop=True)
        length = len(dataFrame)
        train = dataFrame.iloc[:int(length * frac[0]), :]
        val = dataFrame.iloc[int(length * frac[0]):int(length * (frac[0] + frac[1])), :]
        if len(frac) > 2:
            test = dataFrame.iloc[int(length * (frac[0] + frac[1])):length, :]
        else:
            test = pd.DataFrame()
        return dataShape, train, val, test
