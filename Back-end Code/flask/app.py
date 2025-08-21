from flask import Flask,request,jsonify
from flask import render_template
from flask_cors import *
import networkx as nx
import torch
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
import random


app = Flask(__name__)
CORS(app, supports_credentials=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)





def convert_decimal_hours_to_hms(times):
    # Define a helper function to convert a single decimal hour to HH:MM:SS
    def decimal_to_hms(decimal_hour):
        hours = int(decimal_hour)
        minutes = int((decimal_hour - hours) * 60)
        seconds = int(((decimal_hour - hours) * 60 - minutes) * 60)
        return datetime(1, 1, 1, hours, minutes, seconds)

    # Convert the initial time
    initial_time = decimal_to_hms(times[0])
    formatted_times = [initial_time.strftime("%H:%M:%S")]

    # Add 54 seconds to each subsequent time
    for i in range(1, len(times)):
        initial_time += timedelta(seconds=54)
        formatted_times.append(initial_time.strftime("%H:%M:%S"))

    return formatted_times


# Example usage



@app.route('/getheatdata')
def getheatdata():
    data = pd.read_csv(f'../data/detection/detection_set1.csv')

    # x坐标轴
    x_l = list(data['Hour of the day'].values[10:401])
    # times = list(x_a)
    times = convert_decimal_hours_to_hms(x_l)

    data = data.drop(columns=['Hour of the year', 'Hour of the day', 'Label'])
    # metrics的name
    columns = list(data.columns)

    # 折线的数据
    with open('../data/loss_all_d1.json', 'r') as file:
        loss_all = json.load(file)

    color_l=[]
    x_data = []
    for item in loss_all:
        x_data.append(round(item['loss'], 2))
        if item['loss']<3.1167629913094683:
            color_l.append('#A6A6A6')
        else:
            color_l.append('#BF444C')


    with open('../data/loc_local_d1.pkl', 'rb') as file:
        # 加载文件中的数据
        loaded_list = pickle.load(file)

    diff_list = []
    save_list = []
    epsilon = 1e-8
    for item in loaded_list[:391]:
        # print(np.abs(item['y_pred'][0] - item['y'][0]))
        item['y_pred'][0][-1]=item['y_pred'][0][-1]*0.1
        item['y'][0][-1] = item['y'][0][-1] * 0.1

        item['y_pred'][0][0] = item['y_pred'][0][0] * 0.1
        item['y'][0][0] = item['y'][0][0] * 0.1

        # for s in range(21,34):
        #     item['y_pred'][0][s] = item['y_pred'][0][0] * 10
        #     item['y'][0][s] = item['y'][0][0] * 10
        # item['y_pred'][0][33] = item['y_pred'][0][0] * 12
        # item['y'][0][33] = item['y'][0][0] * 12

        diff_list.append(np.abs(item['y_pred'][0] - item['y'][0]))



    v_metrics_list = []  # 变化率大的metric
    for i in range(1, len(diff_list)):
        v_metrics = {}
        v_metrics['hours'] = [times[i]]

        # abs_diff=np.abs(diff_list[i])

        abs_diff = diff_list[i]
        # abs_diff = diff_list[i]-diff_list[i-1]
        # indices = np.argpartition(-abs_diff, 5)[:5]
        # 排序这5个索引，以得到从大到小的顺序

        # mean = np.mean(abs_diff)
        # std = np.std(abs_diff)
        # # # Z-score标准化
        # abs_diff = (abs_diff - mean) / std
        #
        # abs_diff = (abs_diff - abs_diff.min()) / (abs_diff.max() - abs_diff.min())



        largest_five_indices = list(np.argsort(abs_diff)[-6:][::-1])
        largest_five_values = abs_diff[largest_five_indices]
        if x_l[i]>=12.015:
            # print(i)
            largest_five_values = np.log(largest_five_values + 1)+0.5

            if  x_l[i]<=12.2:
                largest_five_values[-1]+=0.3
                largest_five_values[-2] += 0.2
                largest_five_values[0] += 0.2
        else:
            largest_five_values = np.log(largest_five_values + 1)

        for idx, value in zip(largest_five_indices, largest_five_values):
            abs_diff[idx] = value
        # abs_diff[33]+=1.5
        # for h in range(0, len(abs_diff)):
        #     if abs_diff[h] < 1:
        #         abs_diff[h] += 2
        diff_list[i] = abs_diff  # 更新 diff_list 中的条目

        # 保存更新后的 diff_list



        v_name = []
        for k in largest_five_indices:
            v_name.append(columns[k])

        v_metrics['days'] = v_name
        # Find the largest five values

        v_data = []
        for k in range(0, len(largest_five_values)):
            value = largest_five_values[k]
            # if v_name[k]=='power':
            #     value=math.sqrt(value)
            value = float(f"{value:.2f}")

            v_data.append([k, 0,value])


        v_metrics['data'] = v_data
        # print(v_metrics)
        v_metrics_list.append(v_metrics)
    #diff_list 针对g4 diff_list2针对g3 diff_list3针对界面
    # np.save('./static/diff_list3.npy', diff_list)


    heatdata={}
    heatdata['x']=times[1:]
    heatdata['loss'] = x_data[1:391]
    heatdata['heatd'] = v_metrics_list
    heatdata['l_c']=color_l[1:391]

    return json.dumps(heatdata, ensure_ascii=False, cls=NumpyEncoder)

@app.route('/getrawdata')
def getrawdata():
    raw={}

    data = pd.read_csv(f'../data/detection/detection_set1.csv')

    labels_true = data['Label'].values
    # ddetete the TIME column

    x_a = data['Hour of the day'].values
    # print(list(x_a))
    times = list(x_a)
    formatted_times = convert_decimal_hours_to_hms(times)
    # print(formatted_times)

    raw['x']=formatted_times

    data = data.drop(columns=['Hour of the year', 'Hour of the day', 'Label'])
    data['power'] = data['power'] * 10e-6

    lab = pd.read_csv(f'../data/detection/Column_Categories.csv')

    dict = {}
    for l in lab.values:
        dict[l[0]] = l[1]

    # dict2 = {}
    # for l in lab.values:
    #     dict2[l[1]] = []
    # for l in lab.values:
    #     dict2[l[1]].append(l[0])
    # print(dict2)

    # {
    #     name: 'Union Ads',
    #     data: [25, 10, 20, 2, 15, 20, 10],
    #     type: 'line'
    # }

    t1=[]
    t1_name=[]
    t2 = []
    t2_name = []
    t3 = []
    t3_name = []
    con=0
    con2 = 0
    for c in data:

        if dict[c]==1020:
            con2+=1
            if(con2>5):
                t2_name.append(c)
                item = {}
                item['name'] = c
                item['data'] =list(data[c].values)
                item['type']='line'
                t2.append(item)
        if c == 'power':
            t1_name.append(c)
            item = {}
            item['name'] = c
            item['data'] = list(data[c].values)
            item['type'] = 'line'
            t1.append(item)
        if c.split('_')[0] == 'PMV' and con<8:
            con+=1
            t3_name.append(c)
            item = {}
            item['name'] = c
            item['data'] = list(data[c].values)
            item['type'] = 'line'
            t3.append(item)
    raw['t1_name']=t1_name
    raw['t1'] = t1

    raw['t2_name'] = t2_name
    raw['t2'] = t2

    raw['t3_name'] = t3_name
    raw['t3'] = t3



    return json.dumps(raw, ensure_ascii=False, cls=NumpyEncoder)

@app.route('/getGraph1')
def getGraph1():  # put application's code here

    loaded_diff_list = np.load('./static/diff_list.npy')

    data = pd.read_csv(f'../data/detection/detection_set1.csv')
    columns = list(data.columns[2:])
    # x_l = list(data['Hour of the day'].values[10:401])
    # times = list(x_a)
    # times = convert_decimal_hours_to_hms(x_l)


    detection_set = torch.load('../data/detection0626/detection_set_1.pth')


    jianclist=[335,339,341,342,343,344]
    g_l={}
    for j in range(1,len(jianclist)):
        d0=detection_set[jianclist[j-1]]
        d1 = detection_set[jianclist[j]]

        G1 = d1.G
        G0 = d0.G
        # 找出 g2 中比 g1 多出来的边
        new_nodes = set(G1.nodes()) - set(G0.nodes())
        new_edges = set(G1.edges()) - set(G0.edges())

        degree_g1 = dict(G0.degree())
        degree_g2 = dict(G1.degree())

        # 找出所有的节点（包括 g1 和 g2 中的节点）
        all_nodes = set(degree_g1.keys()).union(degree_g2.keys())

        # 计算度数变化
        degree_changes = {}
        for node in all_nodes:
            degree_g1_value = degree_g1.get(node, 0)  # 如果节点不在 g1 中，则度数为 0
            degree_g2_value = degree_g2.get(node, 0)  # 如果节点不在 g2 中，则度数为 0
            degree_changes[node] = abs(degree_g2_value - degree_g1_value)

        # print("Degree changes for each node:", degree_changes)




        diff1=loaded_diff_list[jianclist[j-1]]

        # nodes_list=list(G1.nodes())
        # g01 = nx.Graph()
        # for edge in list(G0.edges):
        #     g01.add_edge(nodes_list.index(edge[0]),nodes_list.index(edge[1]))



        data={}
        nodes=[]
        edges=[]
        nodes_cunzai=[]

        for edge in G1.edges:
            edge_item={}
            edge_item['source']=str(edge[0])
            edge_item['target'] =str(edge[1])

            if edge in new_edges:
                edge_item['newadd']=1.5 #'#A6A6A6'
            else:
                edge_item['newadd'] = 0.8#'#D9D9D9'
            edges.append(edge_item)

        for i in list(G1.nodes()):
            # if i in nodes_cunzai:
                node_item={}
                node_item['id']=str(i)
                node_item['label']=columns[i]
                node_item['c']=diff1[i]
                if i in new_nodes:
                    node_item['bolder']='#7F7F7F'
                else:
                    node_item['bolder'] = '#BF444C'
                node_item['bold']=  degree_changes[i]
                nodes.append(node_item)
        data['nodes']=nodes
        data['edges']=edges
        g_l[f'g{j}']=data

    return json.dumps(g_l, ensure_ascii=False, cls=NumpyEncoder)

@app.route('/getGraph2')#得到大图数据
def getGraph2():  # put application's code here
    data={}

    loaded_diff_list = np.load('./static/diff_list.npy')

    detection_set = torch.load('../data/detection0626/detection_set_1.pth')

    data1 = pd.read_csv(f'../data/detection/detection_set1.csv')
    columnsl = list(data1.columns[2:])

    list_g = [339, 341, 342, 343, 344]
    G1 = nx.Graph()
    for d in list_g:
        g = detection_set[d].G
        for edge in g.edges:
            if edge not in G1.edges:
                G1.add_edge(edge[0], edge[1])

    edges=[]
    nodes_list = list(G1.nodes)
    for edge in G1.edges:
        edge_item = {}
        edge_item['source'] = str(edge[0])
        edge_item['target'] = str(edge[1])
        edges.append(edge_item)

    diff1 = loaded_diff_list[341]

    nodes=[]
    for id in nodes_list:
        # if i in nodes_cunzai:
        node_item = {}
        node_item['id'] = str(id)
        node_item['label'] = columnsl[id]
        node_item['c'] = diff1[id]
        # node_item['bold'] = degree_changes[nodes_list[i]]
        nodes.append(node_item)
    data['nodes'] = nodes
    data['edges'] = edges
    return json.dumps(data, ensure_ascii=False, cls=NumpyEncoder)

@app.route('/getGraph3')#力导向布局/环形 局部/全局 向前还输出坐标
def getGraph3():  # put application's code here




#nodes_cir_all.json 全局环形  #nodes_all.json 全局力导向  #nodes.json 异常局部力导向
    with open('static/nodes_cir_all.json', 'r', encoding='utf-8') as file:
        nodes_pos = json.load(file)
    nodesposj = {}
    # 打印读取到的数据
    # print(nodes_pos)
    nodesposj_list=[]
    for pos in nodes_pos:
        nodesposj[int(pos['id'])]=pos['position']
        nodesposj_list.append(int(pos['id']))



    loaded_diff_list = np.load('./static/diff_list3.npy')
    # loaded_diff_list2 = np.load('./static/diff_list2.npy')

    data = pd.read_csv(f'../data/detection/detection_set1.csv')
    columns = list(data.columns[2:])
    # x_l = list(data['Hour of the day'].values[10:401])
    # times = list(x_a)
    # times = convert_decimal_hours_to_hms(x_l)


    detection_set = torch.load('../data/detection0626/detection_set_1.pth')


    jianclist=[335,336,341,342,343,344]
    g_l={}
    for j in range(1,len(jianclist)):
        d0=detection_set[jianclist[j-1]]
        d1 = detection_set[jianclist[j]]

        G1 = d1.G
        G0 = d0.G
        # 找出 g2 中比 g1 多出来的边
        new_nodes = set(G1.nodes()) - set(G0.nodes())
        new_edges = set(G1.edges()) - set(G0.edges())

        degree_g1 = dict(G0.degree())
        degree_g2 = dict(G1.degree())

        # 找出所有的节点（包括 g1 和 g2 中的节点）
        all_nodes = set(degree_g1.keys()).union(degree_g2.keys())

        # 计算度数变化
        degree_changes = {}
        for node in all_nodes:
            degree_g1_value = degree_g1.get(node, 0)  # 如果节点不在 g1 中，则度数为 0
            degree_g2_value = degree_g2.get(node, 0)  # 如果节点不在 g2 中，则度数为 0
            degree_changes[node] = abs(degree_g2_value - degree_g1_value)

        # print("Degree changes for each node:", degree_changes)




        diff1=loaded_diff_list[jianclist[j-1]]
        # if j==3:
        #     diff1 = loaded_diff_list2[jianclist[j - 1]]

        # nodes_list=list(G1.nodes())
        # g01 = nx.Graph()
        # for edge in list(G0.edges):
        #     g01.add_edge(nodes_list.index(edge[0]),nodes_list.index(edge[1]))



        data={}
        nodes=[]
        edges=[]
        nodes_cunzai=[]

        for edge in G1.edges:
            edge_item={}
            edge_item['source']=str(edge[0])
            edge_item['target'] =str(edge[1])
            if j==2:
                new_edges_list = list(new_edges)

                # 计算要删除的元素数量
                # num_to_remove = len(new_edges_list) // 2
                # print(num_to_remove)

                # 随机选择要删除的元素索引
                indices_to_remove = random.sample(range(len(new_edges_list)), 55)

                # 删除选定的元素
                updated_edges_list = [edge for i, edge in enumerate(new_edges_list) if i not in indices_to_remove]

                # 将更新后的列表转换回 set
                add_edges = set(updated_edges_list)


            else:
                add_edges=new_edges

            if edge in add_edges:
                edge_item['newadd']='#A6A6A6'
            else:
                edge_item['newadd'] = '#D9D9D9'
            edges.append(edge_item)

        for i in list(G1.nodes):
            # if i in nodes_cunzai:
                node_item={}
                node_item['id']=str(i)
                node_item['label']=columns[i]
                node_item['c']=diff1[i]
                if i in new_nodes:
                    node_item['bolder']='#BF444C'
                else:
                    node_item['bolder'] = '#A6A6A6'
                node_item['bold']=  degree_changes[i]
                if j==2:
                    node_item['bold'] = degree_changes[i]/2
                    # print(node_item['bold'])
                # if i in nodesposj_list:
                node_item['fx']=nodesposj[i][0]
                node_item['fy'] = nodesposj[i][1]

                nodes.append(node_item)
        data['nodes']=nodes
        data['edges']=edges
        g_l[f'g{j}']=data

    return json.dumps(g_l, ensure_ascii=False, cls=NumpyEncoder)

@app.route('/getGraph4')#力导向布局/环形 局部/全局 小的布局
def getGraph4():  # put application's code here


    with open('static/small_nodes_cir_all.json', 'r', encoding='utf-8') as file:
        nodes_pos = json.load(file)
    nodesposj = {}
    # 打印读取到的数据
    # print(nodes_pos)
    nodesposj_list=[]
    for pos in nodes_pos:
        nodesposj[int(pos['id'])]=pos['position']
        nodesposj_list.append(int(pos['id']))



    loaded_diff_list = np.load('./static/diff_list3.npy')
    # loaded_diff_list2 = np.load('./static/diff_list2.npy')

    data = pd.read_csv(f'../data/detection/detection_set1.csv')
    columns = list(data.columns[2:])
    # x_l = list(data['Hour of the day'].values[10:401])
    # times = list(x_a)
    # times = convert_decimal_hours_to_hms(x_l)


    detection_set = torch.load('../data/detection0626/detection_set_1.pth')


    jianclist=[334,336,341,342,343,344]
    g_l={}
    for j in range(1,len(jianclist)):
        d0=detection_set[jianclist[j-1]]
        d1 = detection_set[jianclist[j]]

        G1 = d1.G
        G0 = d0.G
        # 找出 g2 中比 g1 多出来的边
        new_nodes = set(G1.nodes()) - set(G0.nodes())
        new_edges = set(G1.edges()) - set(G0.edges())

        degree_g1 = dict(G0.degree())
        degree_g2 = dict(G1.degree())

        # 找出所有的节点（包括 g1 和 g2 中的节点）
        all_nodes = set(degree_g1.keys()).union(degree_g2.keys())

        # 计算度数变化
        degree_changes = {}
        for node in all_nodes:
            degree_g1_value = degree_g1.get(node, 0)  # 如果节点不在 g1 中，则度数为 0
            degree_g2_value = degree_g2.get(node, 0)  # 如果节点不在 g2 中，则度数为 0
            degree_changes[node] = abs(degree_g2_value - degree_g1_value)

        # print("Degree changes for each node:", degree_changes)




        diff1=loaded_diff_list[jianclist[j-1]]
        # if j==3:
        #     diff1 = loaded_diff_list2[jianclist[j - 1]]

        # nodes_list=list(G1.nodes())
        # g01 = nx.Graph()
        # for edge in list(G0.edges):
        #     g01.add_edge(nodes_list.index(edge[0]),nodes_list.index(edge[1]))



        data={}
        nodes=[]
        edges=[]
        nodes_cunzai=[]

        for edge in G1.edges:
            edge_item={}
            edge_item['source']=str(edge[0])
            edge_item['target'] =str(edge[1])

            if edge in new_edges:
                edge_item['newadd']=1.5 #'#A6A6A6'
            else:
                edge_item['newadd'] = 0.8#'#D9D9D9'
            edges.append(edge_item)

        for i in list(G1.nodes):
            # if i in nodes_cunzai:
                node_item={}
                node_item['id']=str(i)
                node_item['label']=columns[i]
                node_item['c']=diff1[i]
                if i in new_nodes:
                    node_item['bolder']='#BF444C'
                else:
                    node_item['bolder'] = '#A6A6A6'
                node_item['bold']=  degree_changes[i]
                # if i in nodesposj_list:
                node_item['fx']=nodesposj[i][0]
                node_item['fy'] = nodesposj[i][1]

                nodes.append(node_item)
        data['nodes']=nodes
        data['edges']=edges
        g_l[f'g{j}']=data

    return json.dumps(g_l, ensure_ascii=False, cls=NumpyEncoder)

if __name__ == '__main__':
    app.run(debug=True)
