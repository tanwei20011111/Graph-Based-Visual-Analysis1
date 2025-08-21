import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric.nn import global_mean_pool,GCNConv,global_max_pool,GATConv
from torch_geometric.data import DataLoader, Batch  # 确保正确导入Batch
import torch_geometric.utils as pyg_utils
import json
import numpy as np
import dgl
from torch.autograd import Variable
from graph_lstm import *
import pickle
import matplotlib.pyplot as plot

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)


    def forward(self, x, edge_index,edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        z=self.conv2(x, edge_index,edge_attr)
        return z



class GNNDemo(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,out_chinnels):
        super(GNNDemo, self).__init__()
        self.evolve_gcn_o1 = EvolveGCNO(in_channels=in_channels)
        # self.evolve_gcn_o2 = EvolveGCNO(in_channels=in_channels)
        # self.evolve_gcn_o3 = EvolveGCNO(in_channels=in_channels)

        self.econv1 = GATConv(in_channels, 2 * hidden_channels)
        self.econv2 = GATConv(2 * hidden_channels, hidden_channels)

        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, in_channels)
        self.conv4 = GCNConv(16, in_channels)
        self.lin = nn.Linear(in_channels*2, out_chinnels)  # 假设池化后的特征维度与in_channels相同
        self.lin2 = nn.Linear(16, in_channels)

        self.encoder=Encoder(in_channels,hidden_channels)
    def forward(self, x, edge_index, edge_attr):
        # x = F.normalize(x, p=2, dim=-1)
        # z=self.econv1(x,edge_index)
        # z = self.econv2(z, edge_index)

        # edge_attr = torch.matmul(z, z.t())
        #
        # edge_attr = edge_attr[edge_index[0], edge_index[1]].unsqueeze(1)

        # x = F.normalize(x, p=2, dim=-1)
        # x1 = F.relu(self.evolve_gcn_o1(x,edge_index,edge_attr))
        x2 = F.relu(self.conv1(x, edge_index,edge_attr))
        x3 = F.relu(self.conv2(x2, edge_index,edge_attr))
        x4 = F.relu(self.conv3(x3, edge_index,edge_attr))

        x=torch.concat((x,x4),dim=1)

        # x = self.lin2(x)  # 应用一个线性层来进行分类
        x = self.lin(x)
        return x  # 对于二分类任务应用sigmoid函数


class GNNLSTMDemo(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,out_chinnels=62):
        super(GNNLSTMDemo, self).__init__()


        self.gnn = GNNDemo(in_channels, hidden_channels,out_chinnels=1) #输入62*in_channels（10） 输出62*out_chinnels（1）
        self.graphLstm = GraphLSTM_pyg(x_size=out_chinnels, h_size=hidden_channels, output_size=out_chinnels,
                                 max_node_num=in_channels)#输入in_channels（10）*x_size（62） 输出1*output_size（62）

        self.lin = nn.Linear(out_chinnels * 2, out_chinnels)

    def forward(self, g1_batch, g1_order, g1_edge_order_mask_list, g2_batch, g2_order, g2_edge_order_mask_list,
                   len_input,graph_gnn):

        x1 =self.graphLstm(g1_batch, g1_order, g1_edge_order_mask_list, g2_batch, g2_order, g2_edge_order_mask_list,
                   len_input)
        x2 = self.gnn(graph_gnn.x, graph_gnn.edge_index,None)


        x=torch.concat((x1,x2.transpose(0, 1)),dim=1)
        #
        #
        out = self.lin(x)

        # x=(x1 + x2) / 2

        return out  # 对于二分类任务应用sigmoid函数

def save_model(save_path, model):
    torch.save({'model_dict': model.state_dict()},save_path)
    print("model save success")

# def load_model(save_path, model):
#     model_data = torch.load(save_path)
#     model.load_state_dict(model_data['model_dict'])
#     print("model load success")

def construct_prediction_new(graph,model,graph_gnn):

    graph_len = len(graph)
    graphlist1 = []
    graphlist2 = []
    graphlist1_dgl = []
    graphlist2_dgl = []
    len_input = np.zeros((graph_len))
    gnum = 0

    # check the device that the model is running on
    device = next(model.parameters()).device
    accu_count = 0
    from torch_geometric.data import Data
    from torch_geometric.data import Batch
    for j in range(0,graph_len):
        len_node = graph[j].num_nodes
        len_input[gnum] = len_node
        nodenum =  graph[j].num_nodes
        g_x = Variable(graph[j].x)
        g1_edge_index = torch.from_numpy(graph[j].g1_edge_index).long()
        g1_edge_label = torch.from_numpy(graph[j].g1_edge_label).float()

        g2_edge_index = torch.from_numpy(graph[j].g2_edge_index).long()
        g2_edge_label = torch.from_numpy(graph[j].g2_edge_label).float()
        g1_data = Data(x=g_x,raw_edge_index=graph[j].edge_index, edge_index=g1_edge_index, edge_attr=g1_edge_label)  # .to(device)
        g2_data = Data(x=g_x, raw_edge_index=graph[j].edge_index,edge_index=g2_edge_index, edge_attr=g2_edge_label)  # .to(device)
        graphlist1_dgl.append(graph[j].g1)
        graphlist2_dgl.append(graph[j].g2)

        graphlist1.append(g1_data)
        graphlist2.append(g2_data)

        accu_count = accu_count + nodenum
        gnum = gnum + 1
    # Variable and cuda

    len_input = torch.from_numpy(len_input).long().to(device)

    ### Use the trained model to predict coordinates
    g1_batch = Batch.from_data_list(graphlist1)  # .to(device)
    g2_batch = Batch.from_data_list(graphlist2)  # .to(device)
    g1_dgl_batch = dgl.batch(graphlist1_dgl)
    g2_dgl_batch = dgl.batch(graphlist2_dgl)
    g1_order = dgl.topological_nodes_generator(g1_dgl_batch)
    g2_order = dgl.topological_nodes_generator(g2_dgl_batch)
    g1_order_mask = np.zeros((len(g1_order), accu_count))
    g2_order_mask = np.zeros((len(g2_order), accu_count))
    g1_edge_index = g1_batch.edge_index
    g2_edge_index = g2_batch.edge_index
    g1_edge_order_mask_list = []
    g2_edge_order_mask_list = []
    for i in range(len(g1_order)):
        order = g1_order[i]
        g1_order_mask[i, order] = 1
        mask_index = g1_order_mask[i, g1_edge_index[0]]
        mask_index = np.nonzero(mask_index)
        g1_edge_order_mask_list.append(mask_index[0])
    for i in range(len(g2_order)):
        order = g2_order[i]
        g2_order_mask[i, order] = 1
        mask_index = g2_order_mask[i, g2_edge_index[0]]
        mask_index = np.nonzero(mask_index)
        g2_edge_order_mask_list.append(mask_index[0])
    g1_order = [order.to(device) for order in g1_order]
    g2_order = [order.to(device) for order in g2_order]
    g1_edge_order_mask_list = [torch.from_numpy(edge_mask).long().to(device) for edge_mask in
                               g1_edge_order_mask_list]
    g2_edge_order_mask_list = [torch.from_numpy(edge_mask).long().to(device) for edge_mask in
                               g2_edge_order_mask_list]
    g1_batch = g1_batch.to(device)
    g2_batch = g2_batch.to(device)
    # y_pred = model(g1_batch, g1_order, g1_edge_order_mask_list, g2_batch, g2_order, g2_edge_order_mask_list,
    #                len_input)
    y_pred = model(g1_batch, g1_order, g1_edge_order_mask_list, g2_batch, g2_order, g2_edge_order_mask_list,
                   len_input,graph_gnn)



    result = {
        "y_pred": y_pred,
        "len_input": len_input
    }
    return result

if __name__ == '__main__':
    # dataset_lstm = torch.load('./data/train0626_lstm.pth')
    #
    # split_index = int(len(dataset_lstm)*0.8)
    # trainset_lstm=dataset_lstm[:split_index]
    # testset_lstm = dataset_lstm[split_index:]
    #
    # dataset = torch.load('./data/train0626.pth')
    # split_index = int(len(dataset)*0.8)
    # trainset=dataset[:split_index]
    # testset = dataset[split_index:]
    # # #
    # # # 模型初始化
    # # # model = GNNDemo(in_channels=10, hidden_channels=16,out_chinnels=1)  # 对于二分类，num_classes设置为1
    # # model = GraphLSTM_pyg(x_size=62, h_size=32, output_size=62,
    # #                              max_node_num=len(trainset[0].x))  # 16 128 2
    window_size=10
    model=GNNLSTMDemo(in_channels=window_size, hidden_channels=32)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Loss function
    loss_fn = torch.nn.MSELoss()

    # ab=torch.nn.L1Loss(reduction='sum')

    #
    # # # 损失函数
    # # loss_fn = nn.BCELoss()
    #
    # 假设trainset是已经准备好的训练数据集
    # train_loader_lstm = DataLoader(trainset_lstm, batch_size=1, shuffle=False)  # 使用DataLoader来批处理数据
    #
    # test_loader_lstm = DataLoader(testset_lstm, batch_size=1, shuffle=False)  # 使用DataLoader来批处理数据
    #
    #
    # train_loader = DataLoader(trainset, batch_size=1, shuffle=False)  # 使用DataLoader来批处理数据
    # test_loader = DataLoader(testset, batch_size=1, shuffle=False)  # 使用DataLoader来批处理数据
    #
    #
    # # # # 训练模型
    # for epoch in range(30):
    #     total_loss=0
    #     total_loss2 = 0
    #
    #     i=0
    #     model.train()
    #     for graph in train_loader_lstm:
    #         optimizer.zero_grad()  # 清除梯度
    #
    #         graph1 = []
    #         graph1.append(graph[0])
    #         result = construct_prediction_new(graph1, model,trainset[i])
    #         # y_pred = model(graph.x, graph.edge_index, graph.batch,None)  # 获取模型预测
    #
    #         # if epoch==3:
    #         #     y_pred = model(graph.x, graph.edge_full_index, graph.batch,graph.edge_xy_attr)
    #         # y=F.normalize(graph.y, p=2, dim=-1)
    #         loss = loss_fn(result['y_pred'], graph.y)  # 计算损失，确保graph.y为正确的形状和类型
    #
    #         # if loss.item()>0:
    #         #     print(y_pred,graph.label,loss)
    #         loss.backward()  # 反向传播
    #         optimizer.step()  # 更新权重
    #         total_loss += loss.item()
    #         i+=1
    #     # print(f"Epoch {epoch + 1}, Training Loss: {total_loss/len(train_loader):.4f}")
    #     model.eval()
    #     with torch.no_grad():
    #         i=0
    #         for graph in test_loader_lstm:
    #             graph1 = []
    #             graph1.append(graph[0])
    #             result = construct_prediction_new(graph1, model,testset[i])
    #             loss2 = loss_fn(result['y_pred'], graph.y)
    #             total_loss2 += loss2.item()
    #             i +=1
    #             # print( f"Loss: {loss.item():.4f}" )
    #         # print(f"Test Loss: {total_loss2/len(test_loader):.4f}")
    #     print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader_lstm):.4f} Test Loss: {total_loss2/len(test_loader_lstm):.4f}")
    #
    #
    # save_model("./data/model0626_lstm_gnn1.pth",model)
    # # #
    # success model0609_lstm_gnn2  train0606.pth train0606_lstm.pth detection0607_lstm/detection_set_1.pth detection0607/detection_set_1.pth
    model_data = torch.load("./data/model0609_lstm_gnn2.pth")
    model.load_state_dict(model_data['model_dict'])
    # #
    detection_set_lstm = torch.load('./data/detection0607_lstm/detection_set_16.pth')
    # detection_set2_lstm = torch.load('./data/detection0607_lstm/detection_set_9.pth')
    # detection_set3_lstm = torch.load('./data/detection0607_lstm/detection_set_12.pth')
    # print(len(detection_set_lstm),len(detection_set2_lstm),len(detection_set3_lstm))


    detection_set = torch.load('./data/detection0607/detection_set_16.pth')
    # detection_set2 = torch.load('./data/detection0607/detection_set_9.pth')
    # detection_set3 = torch.load('./data/detection0607/detection_set_12.pth')
    # d1=detection_set[336]
    # d2 = detection_set[337]
    # d3=detection_set[338]
    # d4 = detection_set[339]
    # d5 = detection_set[340]

    # edges = d1.edge_index.t().tolist()
    # g1=nx.Graph()
    # g1.add_edges_from(edges)
    # nx.draw(g1,pos=nx.circular_layout(g1),with_labels=True)
    # plot.show()
    #
    # edges2 = d2.edge_index.t().tolist()
    # g2 = nx.Graph()
    # g2.add_edges_from(edges2)
    # nx.draw(g2,pos=nx.circular_layout(g2),with_labels=True)
    # plot.show()
    #
    # edges3 = d3.edge_index.t().tolist()
    # g3 = nx.Graph()
    # g3.add_edges_from(edges3)
    # nx.draw(g3, pos=nx.circular_layout(g3), with_labels=True)
    # plot.show()
    #
    # edges4 = d4.edge_index.t().tolist()
    # g4 = nx.Graph()
    # g4.add_edges_from(edges4)
    # nx.draw(g4, pos=nx.circular_layout(g4), with_labels=True)
    # plot.show()
    #
    # edges5 = d5.edge_index.t().tolist()
    # g5 = nx.Graph()
    # g5.add_edges_from(edges5)
    # nx.draw(g5, pos=nx.circular_layout(g5), with_labels=True)
    # plot.show()

    # #
    detection_loader = DataLoader(detection_set_lstm[:380], batch_size=1, shuffle=False)  # 使用DataLoader来批处理数据
    # detection_loader2=DataLoader(detection_set2_lstm[:380], batch_size=1, shuffle=False)
    # detection_loader3 = DataLoader(detection_set3_lstm[:380], batch_size=1, shuffle=False)
    #
    with torch.no_grad():
        i=0
        totle_l1=0
        loss_list1=[]
        loc_list=[]
        for graph in detection_loader:
            graph1 = []
            graph1.append(graph[0])
            result = construct_prediction_new(graph1, model, detection_set[i])
            loss = loss_fn(result['y_pred'], graph.y)
            loc_list.append({'y_pred': result['y_pred'].numpy(), 'y': graph.y.numpy()})

            loss_list1.append({'loss':loss.item(),'label':graph.label.item()})
            print(f"index:{i},Loss: {loss.item():.4f},label:{graph.label.item()}")
            i+=1
        with open('./data/loc_local_d16.pkl', 'wb') as f:
            pickle.dump(loc_list, f)
        with open('./data/loss_all_d16.json', 'w') as file:
            json.dump(loss_list1, file)

    #     i = 0
    #     loss_list2 = []
    #     for graph in detection_loader2:
    #         graph1 = []
    #         graph1.append(graph[0])
    #         result = construct_prediction_new(graph1, model, detection_set2[i])
    #         loss = loss_fn(result['y_pred'], graph.y)
    #         print(f"index:{i},Loss: {loss.item():.4f},label:{graph.label.item()}")
    #         i += 1
    #     print(loss_list2)
    #
    #     i = 0
    #     loss_list3 = []
    #     for graph in detection_loader3:
    #         graph1 = []
    #         graph1.append(graph[0])
    #         result = construct_prediction_new(graph1, model, detection_set3[i])
    #         loss = loss_fn(result['y_pred'], graph.y)
    #         print(f"index:{i},Loss: {loss.item():.4f},label:{graph.label.item()}")
    #         i += 1
    #     my_list=[]
    #     my_list.append(loss_list1)
    #     my_list.append(loss_list2)
    #     my_list.append(loss_list3)
    #     with open('./data/loss_metrics_lstm.json', 'w') as file:
    #         json.dump(my_list, file)
