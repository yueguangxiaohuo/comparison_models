import time
from numba.experimental import jitclass  # 用于创建具有指定数据类型的NumPy兼容类，以加速性能
import numpy as np
from numba.typed import List
import pandas as pd
from numba import jit, int32, float64, njit
from itertools import product
import matplotlib
matplotlib.use('TkAgg')  # 设置matplotlib使用TkAgg后端，这是一种交互式绘图后端，主要用于桌面应用程序
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局变量
rank = 20
minMAERound = 0  # 最小MAE的轮数
minRMSERound = 0  # 最小RMSE的轮数
minRound = 0  # 最小轮数
maxValue = 0  # 最大值
minValue = np.inf  # 初始值为正无穷，表示数据中的最小值，初始状态下为无穷
trainCount = 0  # 训练集数量
testCount = 0  # 测试集数量
validCount = 0  # 验证集数量
trainRound = 200  # 模型训练的最大次数

spec = [('aID', int32),
        ('bID', int32),
        ('cID', int32),
        ('value', float64),
        ('valueHat', float64)]

@jitclass(spec)
class TensorTuple(object):
    def __init__(self):
        self.aID = 0
        self.bID = 0
        self.cID = 0
        self.value = 0.0
        self.valueHat = 0.0

def initData(inputFile1, inputFile2,inputFile3,separator):
    global maxAID, maxBID, maxCID, minAID, minBID, minCID, minValue, maxValue, trainCount, testCount, validCount
    start_time = time.time()   # 开始计时，记录当前时间
    # 读取原始数据集以获取最大值和最小值
    df_original = pd.read_csv(original_file, sep=separator, names=['aID', 'bID', 'cID', 'value'],
                              dtype={'aID': int, 'bID': int, 'cID': int, 'value': float}, engine='python')
    df_original.iloc[:, :3] = df_original.iloc[:, :3].astype(int)

    # 计算原始数据集的最大值和最小值
    maxAID = df_original['aID'].max()
    maxBID = df_original['bID'].max()
    maxCID = df_original['cID'].max()
    minAID = df_original['aID'].min()
    minBID = df_original['bID'].min()
    minCID = df_original['cID'].min()
    maxValue = df_original['value'].max()
    minValue = df_original['value'].min()
    delta = maxValue - minValue
    # 读取数据
    df1 = pd.read_csv(inputFile1, sep=separator, names=['aID', 'bID', 'cID', 'value'],
                     dtype={'aID': int, 'bID': int, 'cID': int, 'value': float}, engine='python')
    df2 = pd.read_csv(inputFile2, sep=separator, names=['aID', 'bID', 'cID', 'value'],
                      dtype={'aID': int, 'bID': int, 'cID': int, 'value': float}, engine='python')
    df3 = pd.read_csv(inputFile3, sep=separator, names=['aID', 'bID', 'cID', 'value'],
                      dtype={'aID': int, 'bID': int, 'cID': int, 'value': float}, engine='python')
    df1.iloc[:, :3] = df1.iloc[:, :3].astype(int)  # 转换数据类型为int
    df2.iloc[:, :3] = df2.iloc[:, :3].astype(int)
    df3.iloc[:, :3] = df3.iloc[:, :3].astype(int)

    def dataframe_to_list(dataframe):  # 将dataframe中的每一行转换为 TensorTuple 的实例，即转换为list
        data = List()
        for row in dataframe.itertuples(index=False):  # 遍历dataframe中的每一行，不包括索引
            qtemp = TensorTuple()
            qtemp.aID = row.aID
            qtemp.bID = row.bID
            qtemp.cID = row.cID
            qtemp.value = row.value
            data.append(qtemp) # 将每一行转换为 TensorTuple 的实例，并添加到 data 列表中
        return data

    trainData = dataframe_to_list(df1)
    testData = dataframe_to_list(df2)
    validData= dataframe_to_list(df3)

    end_time = time.time()  # 记录结束时间
    iteration_time = end_time - start_time  # 计算运行时间
    print("读取数据的时间：", iteration_time)
    return trainData,validData,testData, maxAID, maxBID, maxCID, minAID, minBID, minCID,maxValue,minValue, df1.shape[0], df2.shape[0], df3.shape[0],delta

# 原始数据集
original_file = "E:\\Data\\DB\\Symmetric\\symmetric dispose\\dataset\\Soc_dataset\\soc-sign-bitcoinalpha_updated\\updated\\soc-sign-bitcoinalpha_updated_normalized_20000.txt"

@jit(nopython=True)  # 使用 Numba 加速函数，nopython=True 表示尽量避免使用 Python 对象
def get_prediction(U, S, T, i, j, k):
    p_valueHat = 0.0  # 预测值
    for r in range(rank):
        p_valueHat += U[i][r] * S[j][r] * T[k][r]
    return p_valueHat

@jit(nopython=True)
def LF_matric():
    U = np.random.rand(maxAID + 1, rank)  # 初始化U矩阵，随机生成 U 矩阵，维度为 (maxAID+1, rank)
    S = np.random.rand(maxBID + 1, rank)
    T = np.random.rand(maxCID + 1, rank)
    return U, S, T

@jit(nopython=True)
def Help_matric():  # 辅助矩阵
    grad_U = np.zeros((maxAID + 1, rank))  # 用于计算梯度的辅助矩阵
    grad_S = np.zeros((maxBID + 1, rank))
    grad_T = np.zeros((maxCID + 1, rank))
    return grad_U, grad_S, grad_T

# 训练函数（基于验证集选择最佳模型）
@jit(nopython=True)
def train(trainData, testData, validData, lambda_, eta, m, epsilon, delta):
    # 设置的误差阈值
    errorgap = 1E-5
    flag_rmse = True
    flag_mae = True
    U, S, T = LF_matric()
    best_U = U.copy()
    best_S = S.copy()
    best_T = T.copy()
    # 连续下降轮数小于误差范围切达到阈值终止训练
    threshold = 2
    minRMSE = 200.0
    minMAE = 200.0
    everyRoundRMSE = [0.0] * (trainRound + 1)
    everyRoundMAE = [0.0] * (trainRound + 1)
    everyRoundRMSE[0] = minRMSE
    everyRoundMAE[0] = minMAE

    # 使用训练集进行训练
    for t in range(1, trainRound + 1):
        grad_U, grad_S, grad_T = Help_matric()
        # 更新U矩阵
        for train_tuple in trainData:
            i = train_tuple.aID
            j = train_tuple.bID
            k = train_tuple.cID
            train_tuple.valueHat = get_prediction(U, S, T, i, j, k)
            error = train_tuple.valueHat - train_tuple.value
            for r in range(rank):
                grad_U[i][r] += error * S[j][r] * T[k][r]
        for i in range(maxAID + 1):
            for r in range(rank):
                U[i][r] -= eta * (grad_U[i][r] + lambda_ * U[i][r])
            norm_U = np.sqrt(np.sum(U[i] ** 2))
            if norm_U > m:
                U[i] = U[i] * m / norm_U

        # 更新S矩阵
        for train_tuple in trainData:
            i = train_tuple.aID
            j = train_tuple.bID
            k = train_tuple.cID
            train_tuple.valueHat = get_prediction(U, S, T, i, j, k)
            error = train_tuple.valueHat - train_tuple.value
            for r in range(rank):
                grad_S[j][r] += error * U[i][r] * T[k][r]
        for j in range(maxBID + 1):
            for r in range(rank):
                S[j][r] -= eta * (grad_S[j][r] + lambda_ * S[j][r])
            norm_S = np.sqrt(np.sum(S[j] ** 2))
            if norm_S > m:
                S[j] = S[j] * m / norm_S

        # 更新T矩阵
        for train_tuple in trainData:
            i = train_tuple.aID
            j = train_tuple.bID
            k = train_tuple.cID
            train_tuple.valueHat = get_prediction(U, S, T, i, j, k)
            error = train_tuple.valueHat - train_tuple.value
            for r in range(rank):
                grad_T[k][r] += error * U[i][r] * S[j][r]
        scale = np.sqrt(rank) * m ** 2 * delta / epsilon  # 修正后的噪声尺度
        N = np.random.laplace(0, scale, size=(maxCID + 1, rank))
        for k in range(maxCID + 1):
            for r in range(rank):
                T[k][r] -= eta * (grad_T[k][r] + lambda_ * T[k][r] + N[k][r])

        # 使用验证集评估
        square = 0.0
        abs_count = 0.0
        for valid_tuple in validData:
            i = valid_tuple.aID
            j = valid_tuple.bID
            k = valid_tuple.cID
            valid_tuple.valueHat = get_prediction(U, S, T, i, j, k)
            square += (valid_tuple.value - valid_tuple.valueHat) ** 2
            abs_count += abs(valid_tuple.value - valid_tuple.valueHat)
        everyRoundRMSE[t] = (square / validCount) ** 0.5
        everyRoundMAE[t] = abs_count / validCount
        print("round:", t, "Validation RMSE:", everyRoundRMSE[t], "Validation MAE:", everyRoundMAE[t])

        # 更新最佳参数（以验证集 RMSE 和 MAE 共同为标准）
        if everyRoundRMSE[t] < minRMSE and everyRoundMAE[t] < minMAE:
            minRMSE = everyRoundRMSE[t]
            minMAE = everyRoundMAE[t]
            minRMSERound = t
            minMAERound = t
            best_U = U.copy()
            best_S = S.copy()
            best_T = T.copy()

        # 早停逻辑
        if everyRoundRMSE[t - 1] - everyRoundRMSE[t] > errorgap:
            flag_rmse = False
            tr = 0
        if everyRoundMAE[t - 1] - everyRoundMAE[t] > errorgap:
            flag_mae = False
            tr = 0
        if flag_rmse and flag_mae:
            tr += 1
            if tr == threshold:
                minRound = t
                break
        flag_rmse = True
        flag_mae = True

    print("**************************************************************************************")
    print("rank:", rank)
    print("Validation minRMSE:", minRMSE, "Validation minRMSERound:", minRMSERound)
    print("Validation minMAE:", minMAE, "Validation minMAERound:", minMAERound)
    print("minRound:", minRound)
    # 在测试集上评估
    square = 0.0
    abs_count = 0.0
    for test_tuple in testData:
        i = test_tuple.aID
        j = test_tuple.bID
        k = test_tuple.cID
        test_tuple.valueHat = get_prediction(best_U,best_S, best_T, i, j, k)
        square += (test_tuple.value - test_tuple.valueHat) ** 2
        abs_count += abs(test_tuple.value - test_tuple.valueHat)
    test_rmse = (square / testCount) ** 0.5
    test_mae = abs_count / testCount

    return minRMSE, minMAE, test_rmse, test_mae


if __name__ == "__main__":
    separator = "::"
    lambda_ = 0.01  # 正则化参数
    eta = 0.01  # 学习率
    m = 1.0  # 最大范数
    epsilon = 0.1  # 噪声尺度
    base_path = "E:\\Data\\DB\\Symmetric\\symmetric dispose\\dataset\\Soc_dataset\\soc-sign-bitcoinotc_updated\\updated\\dataset_955"
    datasets = ["divide1", "divide2", "divide3"]
    iterations = 10  # 每次数据集重复运行的次数

    results = {}
    for dataset in datasets:
        print(f"\n=== 处理 {dataset.capitalize()} ===")
        input_file1 = f"{base_path}\\{dataset}\\train_set.txt"
        input_file2 = f"{base_path}\\{dataset}\\test_set.txt"
        input_file3 = f"{base_path}\\{dataset}\\valid_set.txt"

        trainData, validData, testData, maxAID, maxBID, maxCID, minAID, minBID, minCID, maxValue,minValue, trainCount, testCount,validCount,delta = initData(
            input_file1, input_file2, input_file3, separator)
        print("MaxAID:", maxAID + 1, "MaxBID:", maxBID + 1, "MaxCID:", maxCID + 1)
        print("trainCount:", trainCount, "testCount:", testCount, "validCount:", validCount)

        test_rmse_list = []
        test_mae_list = []

        start_time = time.time()
        for i in range(iterations):
            print(f"\n--- {dataset.capitalize()} 第 {i + 1} 次训练 ---")
            minRMSE, minMAE, test_rmse, test_mae = train(trainData, testData, validData, lambda_, eta, m, epsilon, delta)
            test_rmse_list.append(test_rmse)
            test_mae_list.append(test_mae)
            print(f"{dataset.capitalize()} 第 {i + 1} 次测试 RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}")

        avg_test_rmse = sum(test_rmse_list) / len(test_rmse_list)
        avg_test_mae = sum(test_mae_list) / len(test_mae_list)
        sd_test_rmse = np.std(test_rmse_list)
        sd_test_mae = np.std(test_mae_list)

        end_time = time.time()
        print(f"\n{dataset.capitalize()} 训练和测试耗时: {end_time - start_time:.2f} 秒")
        print(f"{dataset.capitalize()} 测试集 RMSE: {avg_test_rmse:.6f}, 标准差: {sd_test_rmse:.6f}")
        print(f"{dataset.capitalize()} 测试集 MAE: {avg_test_mae:.6f}, 标准差: {sd_test_mae:.6f}")

        results[dataset] = {
            'avg_test_rmse': avg_test_rmse,
            'avg_test_mae': avg_test_mae,
            'sd_test_rmse': sd_test_rmse,
            'sd_test_mae': sd_test_mae
        }

    # 计算三个数据集测试集结果的平均值和标准差
    test_rmse_list = [results[dataset]['avg_test_rmse'] for dataset in datasets]
    test_mae_list = [results[dataset]['avg_test_mae'] for dataset in datasets]
    final_avg_test_rmse = sum(test_rmse_list) / len(test_rmse_list)
    final_avg_test_mae = sum(test_mae_list) / len(test_mae_list)
    final_sd_test_rmse = np.std(test_rmse_list)
    final_sd_test_mae = np.std(test_mae_list)

    print("\n=== 三个数据集测试集平均结果 ===")
    for dataset in results:
        print(
            f"{dataset}: RMSE = {results[dataset]['avg_test_rmse']:.6f} ± {results[dataset]['sd_test_rmse']:.6f}, "
            f"MAE = {results[dataset]['avg_test_mae']:.6f} ± {results[dataset]['sd_test_mae']:.6f}")
    print(f"Final Average RMSE for divide1, divide2, divide3: {final_avg_test_rmse:.6f}")
    print(f"Final Average MAE for divide1, divide2, divide3: {final_avg_test_mae:.6f}")
    print(f"Final SD for RMSE: {final_sd_test_rmse:.6f}")
    print(f"Final SD for MAE: {final_sd_test_mae:.6f}")
    print(
        f"Final results: {final_avg_test_rmse:.6f} ± {final_sd_test_rmse:.6f}, {final_avg_test_mae:.6f} ± {final_sd_test_mae:.6f}")