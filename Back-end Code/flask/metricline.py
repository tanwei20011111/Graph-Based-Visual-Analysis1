import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

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

if __name__ == '__main__':
    data = pd.read_csv(f'../data/detection/detection_set1.csv')

    labels_true = data['Label'].values
    # ddetete the TIME column

    x_a = data['Hour of the day'].values[1:]
    # print(list(x_a))


    data = data.drop(columns=['Hour of the year', 'Hour of the day', 'Label'])
    data['power'] = data['power'] * 10e-6


    #
    # lab=pd.read_csv(f'../data/detection/Column_Categories.csv')
    #
    # dict={}
    # for l in lab.values:
    #      dict[l[0]]=l[1]

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
    # m1=data['Setpoints_Chiller_T_chiller'].values

    m1 = data['Sensor_AHU_T_woB'].values
    x_a=convert_decimal_hours_to_hms(x_a)

    for i in range(0,len(m1)):

        # if dict[c]==1:
        #     c_list.append(c)
        #     item = {}
        #     item['name'] = c
        #     item['data'] =list(data[c].values)
        #     item['type']='line'
        #     line_y.append(item)
        # if data[i]=='Sensor_Chiller_T_chiller':
        #     c_list.append(c)
        #     item = {}
        #     item['name'] = c
        #     item['data'] =list(data[c].values)
        #     item['type']='line'
        #     line_y.append(item)

        if i>=349 and i<354:
            c_list.append(m1[i])
            line_y.append(x_a[i])
    # time=convert_decimal_hours_to_hms(line_y)
    print(line_y)
    print(c_list)

    with open('../data/loc_local_d1.pkl', 'rb') as file:
        # 加载文件中的数据
        loaded_list = pickle.load(file)

    p_list=[]
    list1=loaded_list[339:344]
    for item in list1:
        p_list.append(item['y_pred'][0][17])



    print(p_list)

