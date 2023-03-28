import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
from matplotlib import rcParams

path = 'C:\\Users\\Administrator\\Desktop\\小论文\\C值测试（3.21参数更改）\\'
df = pd.read_excel(path + '总表.xlsx',sheet_name='压力误差分析（60-140）')
# print(df)

x = np.arange(1,26)
y_ture = df['120-监测值']
y_net4 = df['120-net4']
y_suiji = df['120-随机优化']
cha1 = y_net4 - y_ture
cha1_jue = abs(cha1)
cha2 = y_suiji - y_ture
cha2_jue = abs(cha2)
index = df['time'].tolist()
print('net4:',sum(cha1_jue))
print('suiji:',sum(cha2_jue))

def drawHistogram():
    plt.rc("font", family='MicroSoft YaHei')
    list1 = cha1   # 柱状图第一组数据
    list2 = cha2   # 柱状图第二组数据
    length = len(list1)
    x = np.arange(length)   # 横坐标范围
    listDate = index

    config = {
        "font.family": 'serif',
        "font.size": 12,  # 相当于小四大小
        "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
        "font.serif": ['SimSun'],  # 宋体
        'axes.unicode_minus': False  # 处理负号，即-号
    }
    rcParams.update(config)

    plt.figure(figsize=(15, 6))
    total_width, n = 0.8, 2   # 柱状图总宽度，有几组数据
    width = total_width / n   # 单个柱状图的宽度
    x1 = x - width / 2   # 第一组数据柱状图横坐标起始位置
    x2 = x1 + width   # 第二组数据柱状图横坐标起始位置

    # plt.title("一周每天吃悠哈软糖颗数柱状图")   # 柱状图标题
    # plt.xlabel("星期")   # 横坐标label 此处可以不添加
    # plt.xticks(rotation=20)
    plt.ylabel("压力误差（单位：米）")   # 纵坐标label
    plt.xlabel("数据采集时间点（单位：小时）")
    plt.bar(x1, list1, width=width, label="基于压力估计的120节点压力误差")
    plt.bar(x2, list2, width=width, label="传统优化算法的120节点压力误差")
    plt.xticks(x, listDate,fontproperties='Times New Roman', size=12)   # 用星期几替换横坐标x的值
    plt.yticks( fontproperties='Times New Roman', size=12)
    plt.legend()   # 给出图例
    # for a, b in zip(index, list1):
    #     plt.text(a, b,
    #              b,
    #              ha='center',
    #              va='bottom',
    #              )
    # for a, b in zip(index, list2):
    #     plt.text(a, b,
    #              b,
    #              ha='center',
    #              va='bottom',
    #              )
    plt.savefig("pres_120.png", dpi=500, bbox_inches='tight')  # 解决图片不清晰，不完整的问题
    plt.show()

if __name__ == '__main__':
    drawHistogram()


# # 折线图
# line1, = plt.plot(np.arange(1,26), y1, color='purple', lw=0.5, ls='-', marker='o', ms=4)
#
# # 折线图置信区间
# # plt.fill_between(np.arange(1,26), y1 - 0.5, y1 + 0.5, color=(229/256, 204/256, 249/256), alpha=0.9)
#
# # 散点图
# y1_1 = df['120-net4']
# y1_2 = df['120-随机优化']
# y1_1 = plt.scatter(np.arange(1,26),y1_1, color=(0/256, 114/256, 189/256))  # 蓝色
# y1_2 = plt.scatter(np.arange(1,26),y1_2, color=(217/256, 83/256, 25/256))  # 红色
#
# # x,y轴坐标文字设置
# plt.xticks(rotation = 20)
# plt.xticks(x,index,horizontalalignment='right')
# plt.ylabel('压力值(单位:m)')
#
# # 图例设置
# mpl.rcParams["font.sans-serif"] = ["SimHei"]    # 设置显示中文字体
# plt.legend([line1,y1_1,y1_2], ['120节点实测值置信区间','基于压力估计的120节点压力预测值','传统优化方法120节点压力预测值'], loc = 9)
#
# # 保存数据并画图
# plt.savefig("pres_120.png",dpi=500,bbox_inches = 'tight')#解决图片不清晰，不完整的问题
# plt.show()