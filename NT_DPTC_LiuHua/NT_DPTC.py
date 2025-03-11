import numpy as np
import pandas as pd
from numba import jit
import math
import datetime


# train_indices训练集,数据的格式为[k,i,j,Y]
# vaild_indices验证集,数据的格式[k,i,j,Y]


class NT_DPTC():
    def __init__(self, Parameters, shape, r):

        self.eta, self.beta1, self.beta2, self.lambd = Parameters
        K, I, J = shape
        self.I = I
        # 初始化 S, D, T
        self.U = np.random.rand(K, I, r) * 0.05
        self.V = np.random.rand(K, r, J) * 0.05
        self.S = np.random.rand(1, I, J) * 0.05

        self.P = self.U.copy()
        self.Q = self.V.copy()
        self.R = self.S.copy()

        # 一阶动量
        self.FOM_P = np.zeros([K, I, r])
        self.FOM_Q = np.zeros([K, r, J])
        self.FOM_R = np.zeros([1, I, J])

        # 二阶动量
        self.SOM_P = np.zeros([K, I, r])
        self.SOM_Q = np.zeros([K, r, J])
        self.SOM_R = np.zeros([1, I, J])

        self.count_P = np.zeros([K, I])
        self.count_Q = np.zeros([K, J])
        self.count_R = np.zeros([I, J])

        # 记录最佳的 S, D, T
        self.best_U = np.zeros([K, I, r])
        self.best_V = np.zeros([K, r, J])
        self.best_S = np.zeros([1, I, J])
        self.best_tol = 0
        self.best_loss = float('inf')
        self.best_n = float('inf')

    def train(self, train_indices, valid_indices, num_epochs=100, epsilon=0.01, select=0):
        tol = float('inf')
        last_loss = float('inf')
        loss = float('inf')
        n = 1
        conv_cout = 0
        start_time = datetime.datetime.now()
        while n < num_epochs:
            self.U = 2 * sigmoid(self.P)
            self.V = 2 * sigmoid(self.Q)
            self.S = 1 * sigmoid(self.R)
            self.P, self.Q, self.R, self.FOM_P, self.FOM_Q, self.FOM_R, self.SOM_P, self.SOM_Q, self.SOM_R, self.count_P, self.count_Q, self.count_R = \
                update(self.P, self.Q, self.R, train_indices, \
                       [self.eta, self.beta1, self.beta2, self.lambd], \
                       self.FOM_P, self.FOM_Q, self.FOM_R, self.SOM_P, self.SOM_Q, self.SOM_R, self.count_P,
                       self.count_Q, self.count_R)
            if n == 1:
                # last_loss=calculate_obj(valid_indices,self.S,self.D,self.T,self.a,self.b,self.c,self.lambd)
                rmse, mae = metrics(self.U, self.V, self.S, valid_indices)
                if select == 0:
                    last_loss = rmse
                else:
                    last_loss = mae
                self.best_loss = last_loss
                self.best_n = n
                print('验证集:迭代轮数:%d,损失函数:%f,mae:%f' % (n, last_loss, mae))
                # 保存当前最佳数据
                self._save_best()
            if n > 1:
                # loss=calculate_obj(valid_indices,self.S,self.D,self.T,self.a,self.b,self.c,self.lambd)
                rmse, mae = metrics(self.U, self.V, self.S, valid_indices)
                if select == 0:
                    loss = rmse
                else:
                    loss = mae
                tol = last_loss - loss
                print('验证集:迭代轮数:%d,损失函数:%f,mae:%f,tol:%f' % (n, loss, mae, tol))
                if (tol > epsilon):
                    conv_cout = 0
                    #  判断当前obj是否小于最佳obj
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_n = n
                        self.best_tol = tol
                        self._save_best()
                else:
                    conv_cout = conv_cout + 1
                    if conv_cout == 1:
                        end_time = datetime.datetime.now()
                        self.time_cost = (end_time - start_time).seconds
                        break
                last_loss = loss
            n = n + 1
        end_time = datetime.datetime.now()
        self.time_cost = (end_time - start_time).seconds

    def metrics(self, indices_set):
        [RMSE, MSE] = metrics(self.best_U, self.best_V, self.best_S, indices_set)
        return [RMSE, MSE]

    def get_current_info(self):
        return self.best_loss, self.best_n, self.best_tol, self.time_cost

    def _save_best(self):
        self.best_U = self.U.copy()
        self.best_V = self.V.copy()
        self.best_S = self.S.copy()


@jit(nopython=True)
def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)

@jit(nopython=True)
def update(P, Q, R, train_indices, parameters, FOM_P, FOM_Q, FOM_R, SOM_P, SOM_Q, SOM_R, count_P, count_Q, count_R):
    eta, beta1, beta2, lambd = parameters
    # 计算出完整张量用的正则项的规则,
    X = t_product(2 * sigmoid(P), 2 * sigmoid(Q)) * sigmoid(R)
    for [k, i, j, Y] in train_indices:
        k = int(k)
        i = int(i)
        j = int(j)
        X_hat = get_X_hat(2 * sigmoid(P[k, i, :]), 2 * sigmoid(Q[k, :, j]), sigmoid(R[0, i, j]))
        # print(X_hat)
        X[k, i, j] = X_hat
        # dif=0
        # if i==0:
        #     if k==0:
        #         dif=0
        #     else:
        #         dif=X[k,i,j]-X[k-1,I-1,j]
        # else:
        #     dif=X[k,i,j]-X[k,i-1,j]

        # 链式求导,损失函数对X_hat求导
        error = (X_hat - Y)
        # deri_2_f = np.clip(deri_2_f, -1000, 1000)
        # print(deri_2_f)
        P[k, i, :], FOM_P[k, i, :], SOM_P[k, i, :], count_P[k, i] = updateP(P[k, i, :], Q[k, :, j], R[0, i, j],
                                                                            [eta, beta1, beta2, lambd], FOM_P[k, i, :],
                                                                            SOM_P[k, i, :], count_P[k, i], error)
        Q[k, :, j], FOM_Q[k, :, j], SOM_Q[k, :, j], count_Q[k, j] = updateQ(P[k, i, :], Q[k, :, j], R[0, i, j],
                                                                            [eta, beta1, beta2, lambd], FOM_Q[k, :, j],
                                                                            SOM_Q[k, :, j], count_Q[k, j], error)
        R[0, i, j], FOM_R[0, i, j], SOM_R[0, i, j], count_R[i, j] = updateR(P[k, i, :], Q[k, :, j], R[0, i, j],
                                                                            [eta, beta1, beta2, lambd], FOM_R[0, i, j],
                                                                            SOM_R[0, i, j], count_R[i, j], error)
    return P, Q, R, FOM_P, FOM_Q, FOM_R, SOM_P, SOM_Q, SOM_R, count_P, count_Q, count_R


@jit(nopython=True)
def updateP(P, Q, R, parameters, FOM_P, SOM_P, count_P, error):
    eta, beta1, beta2, lambd = parameters
    sigp_2_p = 2 * sigmoid(P) * (1 - sigmoid(P))
    # rval=R[0,i,j]
    # theta=2*error*sigmoid(Q[k,:,j])*sigmoid(R[0,i,j])*sigp_2_p+lambd*2*sigmoid(P[k,i,:])*sigp_2_p+\
    #                         lambd*dif*sigmoid(Q[k,:,j])*sigmoid(R[0,i,j])*sigp_2_p+lambd*X_hat*sigmoid(Q[k,:,j])*sigmoid(R[0,i,j])*sigp_2_p
    theta = 2 * error * sigmoid(Q) * sigmoid(R) * sigp_2_p
    # deri_2_p=np.clip(deri_2_p, -1000, 1000)
    FOM_P = beta1 * FOM_P + (1 - beta1) * theta
    last_SOM_P = SOM_P
    SOM_P = beta2 * SOM_P + (1 - beta2) * np.square(theta)
    count_P = 1 + count_P
    t = count_P
    FOM_P = FOM_P / (1 - beta1 ** t)
    SOM_P = SOM_P / (1 - beta2 ** t)
    SOM_P = np.maximum(SOM_P, last_SOM_P)
    epsilon = 1e-9
    return P - eta * (FOM_P / (np.sqrt(SOM_P) + epsilon) + lambd * P), FOM_P, SOM_P, count_P


@jit(nopython=True)
def updateQ(P, Q, R, parameters, FOM_Q, SOM_Q, count_Q, error):
    eta, beta1, beta2, lambd = parameters
    # rval=R[0,i,j]
    sigq_2_q = 2 * sigmoid(Q) * (1 - sigmoid(Q))
    # theta=2*error*sigmoid(P[k,i,:])*sigmoid(R[0,i,j])*sigq_2_q+lambd*2*sigmoid(Q[k,:,j])*sigq_2_q+\
    #                         lambd*dif*sigmoid(P[k,i,:])*sigmoid(R[0,i,j])*sigq_2_q+lambd*X_hat*sigmoid(P[k,i,:])*sigmoid(R[0,i,j])*sigq_2_q
    theta = 2 * error * sigmoid(P) * sigmoid(R) * sigq_2_q
    FOM_Q = beta1 * FOM_Q + (1 - beta1) * theta
    last_SOM_Q = SOM_Q
    SOM_Q = beta2 * SOM_Q + (1 - beta2) * np.square(theta)
    count_Q = 1 + count_Q
    t = count_Q
    FOM_Q = FOM_Q / (1 - beta1 ** t)
    SOM_Q = SOM_Q / (1 - beta2 ** t)
    SOM_Q = np.maximum(SOM_Q, last_SOM_Q)
    epsilon = 1e-9
    return Q - eta * (FOM_Q / (np.sqrt(SOM_Q) + epsilon) + lambd * Q), FOM_Q, SOM_Q, count_Q


@jit(nopython=True)
def updateR(P, Q, R, parameters, FOM_R, SOM_R, count_R, error):
    eta, beta1, beta2, lambd = parameters

    sigp_2_r = sigmoid(R) * (1 - sigmoid(R))
    # theta=4*error*sigmoid(P[k,i,:])@sigmoid(Q[k,:,j])*sigp_2_r+lambd*sigmoid(R[0,i,j])*sigp_2_r+\
    #                         lambd*dif*sigmoid(P[k,i,:])@sigmoid(Q[k,:,j])*sigp_2_r+lambd*X_hat*sigmoid(P[k,i,:])@sigmoid(Q[k,:,j])*sigp_2_r
    theta = 4 * error * sigmoid(P) @ sigmoid(Q) * sigp_2_r

    FOM_R = beta1 * FOM_R + (1 - beta1) * theta
    SOM_R = beta2 * SOM_R + (1 - beta2) * np.square(theta)
    last_SOM_R = SOM_R
    count_R = 1 + count_R
    t = count_R
    FOM_R = FOM_R / (1 - beta1 ** t)
    SOM_R = SOM_R / (1 - beta2 ** t)
    SOM_R = np.maximum(SOM_R, last_SOM_R)
    epsilon = 1e-9
    return R - eta * (FOM_R / (np.sqrt(SOM_R) + epsilon) + lambd * R), FOM_R, SOM_R, count_R


@jit(nopython=True)
def t_product(A, B):
    K, I, _ = A.shape
    _, _, J = B.shape
    C = (np.random.rand(K, I, J) * 0)
    for k in range(K):
        C[k, :, :] = A[k, :, :] @ B[k, :, :]
    return C


# 获得y_hat
@jit(nopython=True)
def get_X_hat(P, Q, R):
    return P @ Q * R


@jit(nopython=True)
def metrics(U, V, S, indices):
    """
    传入两个张量和需要测试的元素下标
    """
    RMSE = 0
    MAE = 0
    test_num = len(indices)
    for [k, i, j, X] in indices:
        k = int(k)
        i = int(i)
        j = int(j)
        # Y_hat=np.sum(S[i]*D[j]*T[k])+a[i]+b[j]+c[k]+bias[k,i,j]
        X_hat = U[k, i, :] @ V[k, :, j] * S[0, i, j]
        RMSE = RMSE + np.square(X - X_hat)
        MAE = MAE + abs(X - X_hat)
    RMSE = math.sqrt(RMSE / test_num)
    MAE = MAE / test_num
    return [RMSE, MAE]

dataset = "new_iawe"
# rho:1e-05,tau1
shape = tools.loading_shape(f"./dataset/shape/{dataset}_tensor_shape.txt")
train_set = "C:\\Users\\Administrator\\Desktop\\对比实验\\NILM(新)\\iawe(1)\\iawe_15\\train_set.txt"
vaild_set = "C:\\Users\\Administrator\\Desktop\\对比实验\\NILM(新)\\iawe(1)\\iawe_15\\test_set.txt"
test_set = "C:\\Users\\Administrator\\Desktop\\对比实验\\NILM(新)\\iawe(1)\\iawe_15\\valid_set.txt"

def NT_DPTC_valid():
    # self.eta,self.beta1,self.beta2,self.lambd,self.alpha=Parameters
    def load_tensor_data(file_path):
        import pandas as pd
        data = pd.read_csv(file_path, sep='::', engine='python', header=None)
        return data.iloc[:, [0, 1, 2, 3]].values  # 返回numpy数组保持原

    # 加载具体数据集
    train_indices = load_tensor_data(train_set)  # 替换原train_set路径字符串
    valid_indices = load_tensor_data(vaild_set)
    test_indices = load_tensor_data(test_set)
    max_iteration = 150
    iterNum = 1
    RMSE_vector = np.zeros(iterNum)
    MAE_vector = np.zeros(iterNum)
    # n_vector=np.zeros(50)
    time_cost_vector = np.zeros(iterNum)
    R = 5
    Eta = [5e-3, 1e-2, 5e-2, 1e-3]
    Beta1 = [0.01, 0.1, 0.5, 0.8]
    Beta2 = [0.01, 0.1, 0.5, 0.8]
    Lambd = [0.01, 0.001, 0.0001, 0.00001]
    epsilon = 1e-6
    with open('ukdale_NT_DPTC_parameter.txt', 'w') as file:
        for eta in Eta:
            for beta1 in Beta1:
                for beta2 in Beta2:
                    for lambd in Lambd:
                        for i in range(iterNum):
                            print(f"eta:{eta},beta1:{beta1},beta2:{beta2},lamda:{lambd}")
                            model = NT_DPTC([eta, beta1, beta2, lambd], shape, R)
                            model.train(train_indices, valid_indices, num_epochs=max_iteration, epsilon=epsilon)
                            [RMSE, MAE] = model.metrics(test_indices)
                            [loss, n, tol, time_cost] = model.get_current_info()
                            print(f"RMSE: {RMSE}, MAE: {MAE},loss: {loss},n: {n},tol:{tol},time_cost:{time_cost}")
                            # 将结果写入文件
                            file.write(
                                f"eta:{eta},beta1:{beta1},beta2:{beta2},lamda:{lambd}, RMSE: {RMSE}, MAE: {MAE},loss: {loss},n: {n},tol:{tol},time_cost:{time_cost}")
                            file.write(f"\n")
                            # 将结果写入文件
                            RMSE_vector[i] = RMSE
                            MAE_vector[i] = MAE
                            # n_vector[i]=n
                            time_cost_vector[i] = time_cost
                            if i == iterNum - 1:
                                # 计算平均值
                                avg_RMSE = np.mean(RMSE_vector)
                                avg_MAE = np.mean(MAE_vector)
                                # avg_n = np.mean(n_vector)
                                avg_time = np.mean(time_cost_vector)
                                # 计算标准差
                                std_RMSE = np.std(RMSE_vector)
                                std_MAE = np.std(MAE_vector)
                                # 将结果写入文件
                                file.write(f"avg_RMSE: {avg_RMSE}, avg_MAE: {avg_MAE},avg_time:{avg_time},\
                                            std_RMSE:{std_RMSE},std_MAE:{std_MAE}")
                                file.write(f"\n")
                                print(f"avg_RMSE: {avg_RMSE}, avg_MAE: {avg_MAE},avg_time:{avg_time}")
                                print("\n")

NT_DPTC_valid()