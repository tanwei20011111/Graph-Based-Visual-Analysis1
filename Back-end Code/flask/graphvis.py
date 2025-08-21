import networkx as nx
import pandas as pd
import matplotlib.pyplot as plot
import torch
import math

def com_layout(com_graph):

    center={'fx':0,'fy':0}
    nodes=[]
    poslist=[]


    radius=1
    deltaAngle = 2 * math.pi / ( len(com_graph.nodes));
    for i in range(0,len(com_graph)):
        x = center['fx']+ math.sin(deltaAngle * i)*radius
        y = center['fy'] -math.cos(deltaAngle * i) * radius
        pos={'fx':x,'fy':y}
        posl=[x,y]
        nodes.append(pos)
        poslist.append(posl)
    return nodes,poslist

if __name__ == '__main__':
    data = pd.read_csv(f'../data/detection/detection_set1.csv')
    columns = list(data.columns[2:])


    detection_set = torch.load('../data/detection0626/detection_set_1.pth')
    # detection_set2 = torch.load('./data/detection0607/detection_set_9.pth')
    # detection_set3 = torch.load('./data/detection0607/detection_set_12.pth')
    # d0 = detection_set[339]
    d01 = detection_set[340]
    # d1=detection_set[341]
    d2 = detection_set[342]
    d3=detection_set[343]
    d4 = detection_set[344]

    # G0 = d1.G
    # nodes_list=list(G0.nodes())
    # g01 = nx.Graph()
    # for edge in list(G0.edges):
    #     g01.add_edge(nodes_list.index(edge[0]),nodes_list.index(edge[1]))
    # # nx.draw(g01, pos=nx.circular_layout(g01), with_labels=True)
    # # plot.show()

    G2 = d01.G
    nodes_list2 = list(G2.nodes())
    g2 = nx.Graph()
    for edge in list(G2.edges):
        g2.add_edge(nodes_list2.index(edge[0]), nodes_list2.index(edge[1]))
    # nx.draw(g2, pos=nx.circular_layout(g2), with_labels=True)
    # plot.show()

    new_nodes = set(G2.nodes()) - set(d4.G.nodes())

    # 找出 g2 中比 g1 多出来的边
    new_edges = set(G2.edges()) - set(d4.G.edges())

    print("New nodes in g2:", new_nodes)
    print("New edges in g2:", len(new_edges))

    for i in range(0, len(nodes_list2)):
        print(i,columns[nodes_list2[i]])

    # edges_n=[]
    # for edge in G2.edges:
    #     if edge in new_edges:
    #         # print(edge)
    #         edges_n.append(edge)
    # print(len(edges_n))
    # print(len(G2.edges),len(G0.edges))

    degree_g1 = dict(G2.degree())
    degree_g2 = dict(d3.G.degree())

    # 找出所有的节点（包括 g1 和 g2 中的节点）
    all_nodes = set(degree_g1.keys()).union(degree_g2.keys())

    # 计算度数变化
    degree_changes = {}
    for node in all_nodes:
        degree_g1_value = degree_g1.get(node, 0)  # 如果节点不在 g1 中，则度数为 0
        degree_g2_value = degree_g2.get(node, 0)  # 如果节点不在 g2 中，则度数为 0
        degree_changes[node] = abs(degree_g2_value - degree_g1_value)

    print("Degree changes for each node:", degree_changes)

    # d5 = detection_set[347]
    #
    # d6 = detection_set[348]


    # d7 = detection_set[345]

    # edges0 = d0.edge_index.t().tolist()
    # g0 = nx.Graph()
    # for i in range(0,61):
    #     g0.add_node(i)
    # g0.add_edges_from(edges0)
    # nx.draw(g0, pos=nx.circular_layout(g0), with_labels=True)
    # plot.show()

    edges01 = d01.edge_index.t().tolist()
    g01 = nx.Graph()
    for i in range(0,61):
        g01.add_node(i)
    g01.add_edges_from(edges01)
    nx.draw(g01, pos=nx.circular_layout(g01), with_labels=True)
    plot.show()


    edges = d1.edge_index.t().tolist()
    g1=nx.Graph()
    for i in range(0,61):
        g1.add_node(i)

    g1.add_edges_from(edges)
    nodes,posl=com_layout(g1)


    nx.draw(g1,pos=nx.circular_layout(g1),with_labels=True)
    plot.show()
    #
    #
    #
    # edges2 = d2.edge_index.t().tolist()
    # g2 = nx.Graph()
    # for i in range(0,61):
    #     g2.add_node(i)
    # g2.add_edges_from(edges2)
    # nx.draw(g2,pos=nx.circular_layout(g2),with_labels=True)
    # plot.show()
    #
    # edges3 = d3.edge_index.t().tolist()
    # g3 = nx.Graph()
    # for i in range(0,61):
    #     g3.add_node(i)
    # g3.add_edges_from(edges3)
    # nx.draw(g3, pos=nx.circular_layout(g3), with_labels=True)
    # plot.show()
    #
    # edges4 = d4.edge_index.t().tolist()
    # g4 = nx.Graph()
    # for i in range(0,61):
    #     g4.add_node(i)
    # g4.add_edges_from(edges4)
    # nx.draw(g4, pos=nx.circular_layout(g4), with_labels=True)
    # plot.show()
    #
    # edges5 = d5.edge_index.t().tolist()
    # g5 = nx.Graph()
    # for i in range(0,61):
    #     g5.add_node(i)
    # g5.add_edges_from(edges5)
    # nx.draw(g5, pos=nx.circular_layout(g5), with_labels=True)
    # plot.show()
    #
    #
    # edges6 = d6.edge_index.t().tolist()
    # g6 = nx.Graph()
    # for i in range(0,61):
    #     g6.add_node(i)
    # g6.add_edges_from(edges6)
    # nx.draw(g6, pos=nx.circular_layout(g6), with_labels=True)
    # plot.show()
    #
    # # edges7 = d7.edge_index.t().tolist()
    # # g7 = nx.Graph()
    # # g7.add_edges_from(edges7)
    # # nx.draw(g7, pos=nx.circular_layout(g7), with_labels=True)
    # # plot.show()