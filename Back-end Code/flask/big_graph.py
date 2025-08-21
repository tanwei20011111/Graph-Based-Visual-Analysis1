import networkx
import networkx as nx
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plot

if __name__ == '__main__':
    detection_set = torch.load('../data/detection0626/detection_set_1.pth')
    list_g=[ 339, 341, 342, 343, 344]
    G1=nx.Graph()
    for d in list_g:
        g=detection_set[d].G
        for edge in g.edges:
            if edge not in G1.edges:
                G1.add_edge(edge[0],edge[1])
    print(len(G1.nodes))
    nx.draw(G1)
    plot.show()

