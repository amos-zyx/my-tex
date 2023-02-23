import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

def Cluster(X, n_clusters):
    '''
    对数据集X 进行k 均值聚类分析，k=n_clusters
    '''
    # 建立聚类模型对象
    kmeans = KMeans(n_clusters=n_clusters, random_state=2018)
    # 训练聚类模型
    kmeans.fit(X)
    # 预测聚类模型
    pre_y = kmeans.predict(X)
    # 样本距离最近的聚类中心的总和
    inertias = kmeans.inertia_
    return pre_y, inertias

# 导入数据
print("开始导入数据, This is Tom's book")
df_x = pd.read_csv("A 题附件1-read 计数矩阵.csv", header=0)
title = np.array(df_x.columns)
X = np.array(df_x)
df_y = pd.read_excel("A 题附件1-细胞类型.xlsx", header=0, index_col=0)
y = np.array(df_y)
# 数据预处理
print("数据预处理")
X = np.delete(X, 0, axis=1)
X = X.T
y = y.T
y = y[0]
title = np.delete(title, 0)
# 主成分分析
print("主成分分析")
from sklearn.decomposition import PCA
#取60 个主成分，此时var 已经达到0.8
