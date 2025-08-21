import pandas as pd

# Load the dataset
# df = pd.read_excel('./data/HVAC system dataset - Log 1 - Final.xlsx')
#
# # Select rows starting from the third row (Python uses 0-based indexing)
# selected_rows = df.iloc[2:]
#
# # # Define the new column names you want to use
# new_column_names = [
#     'Hour of the year', 'Hour of the day', 'Sensor_Tamb', 'Sensor_A_Tz1', 'Sensor_A_Tz2', 'Sensor_A_Tz3', 'Sensor_A_Tz4',
#     'Sensor_B_Tz1', 'Sensor_B_Tz2', 'Sensor_B_Tz3', 'Sensor_B_Tz4', 'Sensor_C_Tz1', 'Sensor_C_Tz2', 'Sensor_C_Tz3', 'Sensor_C_Tz4',
#     'Sensor_AHU_T_aoA', 'Sensor_AHU_T_aoB','Sensor_AHU_T_aoC', 'Sensor_AHU_T_woA', 'Sensor_AHU_T_woB', 'Sensor_AHU_T_woC',
#     'Sensor_Chiller_T_t', 'Sensor_Chiller_T_chiller',
#     'Signal_A_Uz1', 'Signal_A_Uz2', 'Signal_A_Uz3', 'Signal_A_Uz4',
#     'Signal_B_Uz1', 'Signal_B_Uz2', 'Signal_B_Uz3', 'Signal_B_Uz4', 'Signal_C_Uz1', 'Signal_C_Uz2', 'Signal_C_Uz3', 'Signal_C_Uz4',
#     'Signal_Cwt_U_t', 'Setpoints_A_Tz1', 'Setpoints_A_Tz2', 'Setpoints_A_Tz3', 'Setpoints_A_Tz4',
#     'Setpoints_B_Tz1', 'Setpoints_B_Tz2', 'Setpoints_B_Tz3', 'Setpoints_B_Tz4',
#     'Setpoints_C_Tz1', 'Setpoints_C_Tz2', 'Setpoints_C_Tz3', 'Setpoints_C_Tz4',
#     'Setpoints_AHU_Tao', 'Setpoints_Cwt_T_t', 'Setpoints_Chiller_T_chiller', 'Label'
# ]
# # print(len(new_column_names))
# # print(len(selected_rows.columns))
# # print(selected_rows.columns)
# #
# # # Check if the length of new column names matches the number of columns in selected_rows
# if len(new_column_names) == len(selected_rows.columns):
#     # Update the column names
#     selected_rows.columns = new_column_names
#     # Save the updated DataFrame to a CSV file, without the index
#     selected_rows.to_csv('./data/log1_updated_columns.csv', index=False)
# else:
#     print("The number of new column names does not match the number of columns in the DataFrame.")


# df = pd.read_excel('./data/HVAC system dataset - Log 3 - Final.xlsx')
#
# # Select rows starting from the third row (Python uses 0-based indexing)
# selected_rows = df.iloc[2:]
# print(selected_rows)
#
# # # # Define the new column names you want to use
# new_column_names = [
#     'Hour of the year', 'Hour of the day', 'Sensor_Tamb', 'Sensor_A_Tz1', 'Sensor_A_Tz2', 'Sensor_A_Tz3', 'Sensor_A_Tz4',
#     'Sensor_B_Tz1', 'Sensor_B_Tz2', 'Sensor_B_Tz3', 'Sensor_B_Tz4', 'Sensor_C_Tz1', 'Sensor_C_Tz2', 'Sensor_C_Tz3', 'Sensor_C_Tz4',
#     'Sensor_AHU_T_aoA', 'Sensor_AHU_T_aoB','Sensor_AHU_T_aoC', 'Sensor_AHU_T_woA', 'Sensor_AHU_T_woB', 'Sensor_AHU_T_woC',
#     'Sensor_Chiller_T_t', 'Sensor_Chiller_T_chiller',
#     'Signal_A_Uz1', 'Signal_A_Uz2', 'Signal_A_Uz3', 'Signal_A_Uz4',
#     'Signal_B_Uz1', 'Signal_B_Uz2', 'Signal_B_Uz3', 'Signal_B_Uz4', 'Signal_C_Uz1', 'Signal_C_Uz2', 'Signal_C_Uz3', 'Signal_C_Uz4',
#     'Signal_Cwt_U_t', 'Setpoints_A_Tz1', 'Setpoints_A_Tz2', 'Setpoints_A_Tz3', 'Setpoints_A_Tz4',
#     'Setpoints_B_Tz1', 'Setpoints_B_Tz2', 'Setpoints_B_Tz3', 'Setpoints_B_Tz4',
#     'Setpoints_C_Tz1', 'Setpoints_C_Tz2', 'Setpoints_C_Tz3', 'Setpoints_C_Tz4',
#     'Setpoints_AHU_Tao', 'Setpoints_Cwt_T_t', 'Setpoints_Chiller_T_chiller',
#     'PMV_A_z1','PMV_A_z2','PMV_A_z3','PMV_A_z4','PMV_B_z1','PMV_B_z2','PMV_B_z3','PMV_B_z4','PMV_C_z1','PMV_C_z2','PMV_C_z3','PMV_C_z4','power','Label'
# ]
# # print(len(new_column_names))
# # print(len(selected_rows.columns))
# # print(selected_rows.columns)
# #
# # # Check if the length of new column names matches the number of columns in selected_rows
# if len(new_column_names) == len(selected_rows.columns):
#     # Update the column names
#     selected_rows.columns = new_column_names
#     # Save the updated DataFrame to a CSV file, without the index
#     selected_rows.to_csv('./data/log3_updated_columns.csv', index=False)
# else:
#     print("The number of new column names does not match the number of columns in the DataFrame.")

# df = pd.read_excel('./data/HVAC system dataset - Log 2 - Final.xlsx')
#
# # Select rows starting from the third row (Python uses 0-based indexing)
# selected_rows = df.iloc[2:]
# print(selected_rows)
#
# # # # Define the new column names you want to use
# new_column_names = [
#     'Hour of the year', 'Hour of the day', 'Sensor_Tamb', 'Sensor_A_Tz1', 'Sensor_A_Tz2', 'Sensor_A_Tz3', 'Sensor_A_Tz4',
#     'Sensor_B_Tz1', 'Sensor_B_Tz2', 'Sensor_B_Tz3', 'Sensor_B_Tz4', 'Sensor_C_Tz1', 'Sensor_C_Tz2', 'Sensor_C_Tz3', 'Sensor_C_Tz4',
#     'Sensor_AHU_T_aoA', 'Sensor_AHU_T_aoB','Sensor_AHU_T_aoC', 'Sensor_AHU_T_woA', 'Sensor_AHU_T_woB', 'Sensor_AHU_T_woC',
#     'Sensor_Chiller_T_t', 'Sensor_Chiller_T_chiller',
#     'Signal_A_Uz1', 'Signal_A_Uz2', 'Signal_A_Uz3', 'Signal_A_Uz4',
#     'Signal_B_Uz1', 'Signal_B_Uz2', 'Signal_B_Uz3', 'Signal_B_Uz4', 'Signal_C_Uz1', 'Signal_C_Uz2', 'Signal_C_Uz3', 'Signal_C_Uz4',
#     'Signal_Cwt_U_t', 'Setpoints_A_Tz1', 'Setpoints_A_Tz2', 'Setpoints_A_Tz3', 'Setpoints_A_Tz4',
#     'Setpoints_B_Tz1', 'Setpoints_B_Tz2', 'Setpoints_B_Tz3', 'Setpoints_B_Tz4',
#     'Setpoints_C_Tz1', 'Setpoints_C_Tz2', 'Setpoints_C_Tz3', 'Setpoints_C_Tz4',
#     'Setpoints_AHU_Tao', 'Setpoints_Cwt_T_t', 'Setpoints_Chiller_T_chiller',
#     'PMV_A_z1','PMV_A_z2','PMV_A_z3','PMV_A_z4','PMV_B_z1','PMV_B_z2','PMV_B_z3','PMV_B_z4','PMV_C_z1','PMV_C_z2','PMV_C_z3','PMV_C_z4','power','Label'
# ]
# # print(len(new_column_names))
# # print(len(selected_rows.columns))
# # print(selected_rows.columns)
# #
# # # Check if the length of new column names matches the number of columns in selected_rows
# if len(new_column_names) == len(selected_rows.columns):
#     # Update the column names
#     selected_rows.columns = new_column_names
#     # Save the updated DataFrame to a CSV file, without the index
#     selected_rows.to_csv('./data/train.csv', index=False)
# else:
#     print("The number of new column names does not match the number of columns in the DataFrame.")


# 加载 Excel 文件
xls = pd.ExcelFile('data/HVAC system dataset - Log 3 - Final.xlsx')

# 获取所有工作表(sheet)的名字
sheet_names = xls.sheet_names

# 为每个工作表创建一个 DataFrame，并存储到字典中
dfs = {sheet_name: pd.read_excel(xls, sheet_name) for sheet_name in sheet_names}

# 初始化一个空的列表，用来存储每个工作表处理后的数据
dataframes_to_concat = []

# 遍历每个工作表名称，并处理相应的 DataFrame
for i in range(1, 17):
    df_sheet = dfs[f'Attack ({i})']
    # 选择每个工作表的第三行及之后的所有行
    selected_rows = df_sheet.iloc[2:]
    # 将处理后的 DataFrame 添加到列表中
    dataframes_to_concat.append(selected_rows)

# 使用 pd.concat 来沿着行方向拼接所有的 DataFrame
# final_dataframe = pd.concat(dataframes_to_concat, axis=0)
#
# # final_dataframe 现在包含了所有工作表的第三行及之后的所有行的数据
#
# print(final_dataframe)


# df = pd.read_excel('./data/HVAC system dataset - Log 3 - Final.xlsx')
#
# # Select rows starting from the third row (Python uses 0-based indexing)
# selected_rows = df.iloc[2:]
# print(selected_rows)
#
# # # Define the new column names you want to use
new_column_names = [
    'Hour of the year', 'Hour of the day', 'Sensor_Tamb', 'Sensor_A_Tz1', 'Sensor_A_Tz2', 'Sensor_A_Tz3', 'Sensor_A_Tz4',
    'Sensor_B_Tz1', 'Sensor_B_Tz2', 'Sensor_B_Tz3', 'Sensor_B_Tz4', 'Sensor_C_Tz1', 'Sensor_C_Tz2', 'Sensor_C_Tz3', 'Sensor_C_Tz4',
    'Sensor_AHU_T_aoA', 'Sensor_AHU_T_aoB','Sensor_AHU_T_aoC', 'Sensor_AHU_T_woA', 'Sensor_AHU_T_woB', 'Sensor_AHU_T_woC',
    'Sensor_Chiller_T_t', 'Sensor_Chiller_T_chiller',
    'Signal_A_Uz1', 'Signal_A_Uz2', 'Signal_A_Uz3', 'Signal_A_Uz4',
    'Signal_B_Uz1', 'Signal_B_Uz2', 'Signal_B_Uz3', 'Signal_B_Uz4', 'Signal_C_Uz1', 'Signal_C_Uz2', 'Signal_C_Uz3', 'Signal_C_Uz4',
    'Signal_Cwt_U_t', 'Setpoints_A_Tz1', 'Setpoints_A_Tz2', 'Setpoints_A_Tz3', 'Setpoints_A_Tz4',
    'Setpoints_B_Tz1', 'Setpoints_B_Tz2', 'Setpoints_B_Tz3', 'Setpoints_B_Tz4',
    'Setpoints_C_Tz1', 'Setpoints_C_Tz2', 'Setpoints_C_Tz3', 'Setpoints_C_Tz4',
    'Setpoints_AHU_Tao', 'Setpoints_Cwt_T_t', 'Setpoints_Chiller_T_chiller',
    'PMV_A_z1','PMV_A_z2','PMV_A_z3','PMV_A_z4','PMV_B_z1','PMV_B_z2','PMV_B_z3','PMV_B_z4','PMV_C_z1','PMV_C_z2','PMV_C_z3','PMV_C_z4','power','Label'
]
print(new_column_names.index('Setpoints_AHU_Tao'))
# print(len(new_column_names))
# print(len(selected_rows.columns))
# print(selected_rows.columns)
#
# # Check if the length of new column names matches the number of columns in selected_rows

i=1
for data in dataframes_to_concat:

    if len(new_column_names) == len(data.columns):
        # Update the column names
        data.columns = new_column_names
        # Save the updated DataFrame to a CSV file, without the index
        data.to_csv(f'./data/detection/detection_set{i}.csv', index=False)
    else:
        print("The number of new column names does not match the number of columns in the DataFrame.")
    i+=1
