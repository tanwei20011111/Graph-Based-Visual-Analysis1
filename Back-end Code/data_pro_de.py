import networkx as nx
import pandas as pd
import torch
import torch_geometric as pyg
import numpy as np
import matplotlib.pyplot as plt

def load(i):
## Load the data


    data = pd.read_csv(f'./data/detection/detection_set{i}.csv')

    labels_true = data['Label'].values
    # ddetete the TIME column
    data = data.drop(columns=['Hour of the year','Hour of the day','Label'])
    data['power'] = data['power'] * 10e-5

    # print(data)
    return data,labels_true

def divide_data(data):

    divide_size=5

    # 假设 Data1 是您已经准备好的 DataFrame

    # 计算需要多少个完整的10行区间
    num_intervals = len(data) // divide_size
    # 计算最后一个区间需要包含的行数，这可能少于10行，也可能正好是10行
    remainder = len(data) % divide_size

    # 初始化一个列表来存储所有区间的数组
    intervals = []
    intervals_trues=[]
    # 为每个完整的区间添加数据
    for i in range(num_intervals-1):
        interval = data.iloc[i*divide_size : (i+1)*divide_size].values
        intervals.append(interval)

        intervals_true = data.iloc[(i+1)*divide_size: (i + 2) * divide_size].values
        intervals_trues.append(intervals_true)

    # 如果有剩余的数据，则将它们添加到最后一个区间
    # if remainder > 0:
    #     last_interval = data.iloc[num_intervals*10:].values
    #     intervals.append(last_interval)

    # 验证
    # 打印每个区间的形状
    # for i, interval in enumerate(intervals):
    #     print(f"Interval {i+1}: {interval.shape}")

    return intervals,intervals_trues


def corr_matr(intervals):

    # 假设 intervals 是之前步骤计算得到的，每个元素是一个区间的 NumPy 数组

    # 初始化一个列表来存储每个区间的相关系数矩阵
    correlation_matrices = []

    # 遍历每个区间
    for interval in intervals:
        # 将区间转换为 DataFrame
        df_interval = pd.DataFrame(interval)
        # 计算皮尔逊相关系数矩阵
        corr_matrix = df_interval.corr()
        # print(corr_matrix[0])
        # 存储相关系数矩阵
        correlation_matrices.append(corr_matrix)

        # (可选) 打印相关系数矩阵，或进行其他处理
        # print(corr_matrix)
    return correlation_matrices

import math
def create_graphs(intervals,corr_matrix,labels_true,intervals_trues,index):
    data_list=[]
    # 将DataFrame转换为NumPy数组

    # 获取矩阵的维度
    cont=0
    for i in range(0,len(intervals)):
        x=torch.from_numpy(intervals[i].T).float()
        y=torch.from_numpy(intervals_trues[i].T).float()
        labels=labels_true[(i+1)*x.T.shape[0]:(i+2)*x.T.shape[0]]
        ones_ratio=0
        for r in labels:
            if r == 1:
                ones_ratio+=1

        # 如果至少有80%的值为1，则设置label为1，否则为0
        label = 1 if ones_ratio/len(labels) >= 0.2 else 0

        if label==0:
            cont+=1


        matrix = corr_matrix[i].to_numpy()
        n = matrix.shape[0]
        Fg=nx.Graph()
        G=nx.Graph()
        g=nx.read_graphml('./data/graph.graphml')
        xyedges=list(g.edges)#先验知识
        xyg=[]
        for e in xyedges:
            xyg.append((int(e[0]),int(e[1])))
        for k in range(n):
            for j in range(k + 1, n):
                xyrela=0.5
                if pd.isna(matrix[k][j]):
                    rela=0
                else:
                    rela=math.fabs(matrix[k][j])
                if (k,j) in xyg:
                    rela+=0.5
                    xyrela=0.8
                if rela>=0.6:
                    G.add_edge(k,j,weight=rela)
                Fg.add_edge(k,j,weight=xyrela)
        # nx.draw(G)
        # plt.show()
            # 这里i和j是行和列的索引，matrix[i, j]是相关性系数
        edge_attributes = [rela['weight'] for _, _, rela in G.edges(data=True)]
        edge_attributes =edge_attributes + edge_attributes
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float).unsqueeze(1)

        edge_xy_attributes = [rela['weight'] for _, _, rela in Fg.edges(data=True)]
        edge_xy_attributes = edge_xy_attributes + edge_xy_attributes
        edge_xy_attr = torch.tensor(edge_xy_attributes, dtype=torch.float).unsqueeze(1)

        data1 = pyg.data.Data(
            x=x,
            edge_index=pyg.utils.to_undirected(torch.tensor(list(G.edges)).T),#阈值 PCC+先验
            edge_xy_index=pyg.utils.to_undirected(torch.tensor(xyg).T),#先验知识图
            edge_full_index=pyg.utils.to_undirected(torch.tensor(list(Fg.edges)).T),#全连接
            edge_xy_attr=edge_xy_attr,#全连接+先验知识 如果先验图存在边 边attr=0.5
            num_edges=G.number_of_edges(),
            label=torch.tensor(label),
            edge_attr=edge_attr, #PCC+先验0.5,
            y=y
        )
        data_list.append(data1)

    torch.save(data_list,f'./data/detection/detection_set_{index}.pth')
    print(index,len(intervals),cont)
    # train1.pth w=5 train1.pth w=10

def load_data():
    dataset = torch.load('./data/detection_set_all.pth')
    print(dataset)


if __name__ == '__main__':
    for i in range(1,17):
        data,labels_true=load(i)
        intervals,intervals_trues= divide_data(data)
        corr_matrix=corr_matr(intervals)
        # print(corr_matrix)
        create_graphs(intervals,corr_matrix,labels_true,intervals_trues,i)
    # load_data()

# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
#
# # Optional: Standardize the data
# scaler = StandardScaler()
# Data1_scaled = scaler.fit_transform(Data1)
#
# # Apply k-means clustering
# kmeans = KMeans(n_clusters=2, random_state=0)
# kmeans.fit(Data1_scaled)
#
# # The cluster assignments for each point are stored in 'labels_'
# clusters = kmeans.labels_
#
# # Add the cluster assignments to your original DataFrame
# Data['Cluster'] = clusters

# from sklearn.metrics import adjusted_rand_score
#
# # 计算调整兰德指数
# ari_score = adjusted_rand_score(labels_true, clusters)
#
# print(ari_score)
# If you want to see how many points are in each cluster
# print(Data['Cluster'].value_counts())

# If you want to take a quick look at the clustered dataset
# print(Data.head())
#
# Data.to_csv('./data/clu_columns.csv', index=False)
#
# from sklearn.metrics import accuracy_score
#
# # 直接比较聚类结果与真实标签的准确率
# accuracy_direct = accuracy_score(labels_true, clusters)
# #
# # # 反转聚类标签后比较的准确率（将0变为1，1变为0）
# clusters_inverted = 1 - clusters
# accuracy_inverted = accuracy_score(labels_true, clusters_inverted)
#
# # 取这两个准确率的最大值作为估计准确率
# estimated_accuracy = max(accuracy_direct, accuracy_inverted)
# print(estimated_accuracy)
