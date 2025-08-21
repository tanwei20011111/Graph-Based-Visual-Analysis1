import json
import pickle
import pandas as pd
import numpy as np
import math
import torch

print(torch.__version__)

if __name__ == '__main__':


    data = pd.read_csv(f'../data/detection/detection_set1.csv')

    #x坐标轴
    x_l = list(data['Hour of the day'].values[10:401])

    data = data.drop(columns=['Hour of the year', 'Hour of the day', 'Label'])
    #metrics的name
    columns=list(data.columns)

    #折线的数据
    with open('../data/loss_all_d1.json', 'r') as file:
        loss_all = json.load(file)
    x_data=[]
    for item in loss_all:
        x_data.append(item['loss'])

    #w+2开始
    print(x_l[1:],x_data[1:391])

    with open('../data/loc_local_d1.pkl', 'rb') as file:
        # 加载文件中的数据
        loaded_list = pickle.load(file)

    diff_list=[]
    save_list=[]
    epsilon=1e-8

    j=0
    for item in loaded_list[:391]:
        # print(np.abs(item['y_pred'][0] - item['y'][0]))
        item['y_pred'][0][-1]=item['y_pred'][0][-1]*0.1
        item['y'][0][-1] = item['y'][0][-1] * 0.1

        item['y_pred'][0][0] = item['y_pred'][0][0] * 0.1
        item['y'][0][0] = item['y'][0][0] * 0.1
        diff_list.append(np.abs(item['y_pred'][0] - item['y'][0]))

        if j >= 341:
            save_list.append(np.log(np.abs(item['y_pred'][0] - item['y'][0]) + 1)+1)
        else:
            save_list.append(np.log(np.abs(item['y_pred'][0] - item['y'][0]) + 1))
        j+=1


    # np.save('./static/diff_list.npy', save_list)






    v_metrics_list=[] #变化率大的metric
    for i in range(1,len(diff_list)):
        v_metrics={}
        v_metrics['hours']=[x_l[i]]


        # abs_diff=np.abs(diff_list[i])

        abs_diff = diff_list[i]
        # abs_diff = diff_list[i]-diff_list[i-1]
        indices = np.argpartition(-abs_diff, 5)[:5]
        # 排序这5个索引，以得到从大到小的顺序
        largest_five_indices = list(np.argsort(abs_diff)[-5:][::-1])
        largest_five_values = abs_diff[largest_five_indices]

        largest_five_values=np.log(largest_five_values + 1)



        # mean = np.mean(largest_five_values)
        # std = np.std(largest_five_values)
        # # Z-score标准化
        # largest_five_values = (largest_five_values - mean) / std

        # min_val = np.min(largest_five_values)
        # max_val = np.max(largest_five_values)
        # largest_five_values = (largest_five_values - min_val) / (max_val - min_val)

        v_name=[]
        for k in largest_five_indices:
            v_name.append(columns[k])

        v_metrics['days']=v_name
        # Find the largest five values

        v_data=[]
        for k in range(0,len(largest_five_values)):
            value=largest_five_values[k]
            # if v_name[k]=='power':
            #     value=math.sqrt(value)
            v_data.append([k,0,value])
        v_metrics['data'] = v_data

        v_metrics_list.append(v_metrics)

    print(v_metrics_list)
    print(len(x_l[1:]))
    print(len(x_data[1:391]))
    print(len(v_metrics_list))




