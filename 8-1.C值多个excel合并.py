#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/18 21:21
# @Author  : zifeng
# @File    : 石角2022.5.23
import pandas as pd
import os
def get_manyExcel_to_one(source_path, res_path):
    '''
    description: 将多张excel表的数据纵向合成一张表
    :param source_path: 多张excel表所在的文件夹路径
    :param res_path: 结果表存放的路径
    :return: None
    '''

    # 获取所有需要合并的excel表的表名（带xlsx后缀名）例如: aaa.xlsx
    fileName_list = os.listdir(source_path)

    # 创建一个空的Dataframe，用来存放最后需要写入的结果
    file_res = pd.DataFrame()

    # 循环遍历每一张表
    for file in fileName_list:
        df = pd.read_excel(source_path + '\\' + file)
        file_res = file_res.append(df)

    # 写入最终结果表
    file_res.to_excel(res_path, index=False)
    print(file_res.to_string())


if __name__ == "__main__":
    get_manyExcel_to_one('C:\\Users\\Administrator\\Desktop\\小论文\\C值测试（3.21参数更改）\\net4（60-150）\\',
                         'C:\\Users\\Administrator\\Desktop\\小论文\\C值测试（3.21参数更改）\\net4（60-150）\\net4（60-150）.xlsx')

