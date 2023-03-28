#   _*_utf-8_*_
#   @time:2022/9/19
import pandas as pd
import numpy as np
'''//
目的：创建输入到waterdesk用水模式的excel模板
    1.导入外部excel的远传表或者区域监控表用户编号
    2.创建一个新的DataFarme
    3.将列数据转换成行作为第一行索引，将excel表的名字添加到列表中使索引和excel表名相符时：
        历遍yuanchuanTB文件夹中的每一个文件：
            搜索文件名与行索引相同的那个excel
            将特定excel表的‘value’列添加到指定索引行的位置
    打印新生成的excel表
'''
# source_path = 'C:\\Users\\Administrator\\Desktop\\城南输入文件\\输出文件4\\处理后数据\\远传用户表\\城南远传户表.xlsx'
# c = pd.read_excel(source_path)
# df = c['ID'].tolist()
# print(df)
shuru_path = 'C:\\Users\\Administrator\\Desktop\\小论文\\C值测试（3.21参数更改）\\net4（60-150）\\net4（60-150）.xlsx'
d = pd.read_excel(shuru_path)
df1 = d[['ID','VALUE']]
print(df1)
file_res = pd.DataFrame()
cols = []
values = []
for name,value in df1.groupby('ID'):
    cols.append(name)
    values.append(value['VALUE'])
    print(name)
    print(value['VALUE'])
values = np.array(values).reshape(34,30)   #这是一个矩阵 一天24小时48个数据，看多少行除以48就是前面的数字
values = np.transpose(values)
print(values)
df = pd.DataFrame(data=values,columns=cols)
out_put = 'C:\\Users\\Administrator\\Desktop\\小论文\\C值测试（3.21参数更改）\\net4（60-150）\\net4（60-150）处理后.xlsx'
df.to_excel(out_put)
print(df)











