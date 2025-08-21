import torch
import pandas as pd



if __name__ == '__main__':
    data = pd.read_csv(f'../data/detection/detection_set1.csv')

    labels_true = data['Label'].values
    # ddetete the TIME column

    x_a = data['Hour of the day'].values
    print(list(x_a))


    data = data.drop(columns=['Hour of the year', 'Hour of the day', 'Label'])
    data['power'] = data['power'] * 10e-6



    lab=pd.read_csv(f'../data/detection/Column_Categories.csv')

    dict={}
    for l in lab.values:
         dict[l[0]]=l[1]

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
    line_y=[]
    c_list=[]
    for c in data:

        # if dict[c]==1:
        #     c_list.append(c)
        #     item = {}
        #     item['name'] = c
        #     item['data'] =list(data[c].values)
        #     item['type']='line'
        #     line_y.append(item)
        if c.split('_')[0]=='PMV':
            c_list.append(c)
            item = {}
            item['name'] = c
            item['data'] =list(data[c].values)
            item['type']='line'
            line_y.append(item)
    print(line_y)
    print(c_list)


