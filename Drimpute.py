import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
# 聚类的数目
k = 9
with open("data/75748.csv") as f:
    ncols = len(f.readline().split(','))
    processed_count_matrix = np.loadtxt(open("data/75748.csv", "rb"), delimiter=",", skiprows=1, usecols=range(1, ncols + 1))

def DrImpute(X,k):
    #对原始数据进行对数转换
    X_norm = np.log10(X + 1)
    #计算距离矩阵
    dist_matrix = pdist(X_norm, metric='correlation')
    #转换为相似度矩阵
    sim_matrix = 1 - dist_matrix
    #计算前5%主成分并进行K-means聚类
    n_components = int(np.ceil(0.05 * X_norm.shape[0]))
    pca = PCA(n_components=n_components)
    pca.fit(X_norm)
    X_pca = pca.transform(X_norm)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_pca)
    #获取聚类结果
    clusters = kmeans.labels_
    #计算期望值
    exp_values = np.zeros(X_norm.shape)
    for i in range(k):
        idx = np.where(clusters == i)[0]
        X_cluster = X_norm[idx, :]
        for j in range(X_norm.shape[1]):
            #过滤掉0值
            non_zeros = X_cluster[:, j] != 0
            if np.sum(non_zeros) > 0:
                exp_values[idx, j] = np.mean(X_cluster[non_zeros, j])
    #填充缺失值
    nan_idx =  np.nonzero(X == 0)
    X_imp = np.copy(X)
    exp_values = 10**exp_values-1
    X_imp[nan_idx] = exp_values[nan_idx]
    return X_imp



out_matrix = DrImpute(processed_count_matrix,k)
print(out_matrix.shape)
with open('75748_dr.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(out_matrix)
