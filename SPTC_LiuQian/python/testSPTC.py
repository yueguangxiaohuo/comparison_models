# 初始化模型
from SPTC import SPTC

model = SPTC(epoch=1000, R=10, lr=1e-5, rho=10)
model.initFromFlie(csvPath='exampleData.csv', frac=[0.7, 0.1, 0.2], scale=0.5)
model.decompose(stopError=1e-6)  # train

'''
other method
model = {'train','val','test'} 每论结束后需要验证的数据集
model.getBestTestMetrics()
'''
