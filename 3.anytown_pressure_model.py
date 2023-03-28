#   _*_coding_*_:utf-8


import pandas as pd
'''导入数据'''
data = pd.read_excel('D:\\pycharmProject\\pytorch\\ANN\\shuju\\new_file.xlsx',
                     header=None, index_col=None)  # 一共4853组数据，每组数据100个特征，325个特征对应一个输出
x = data.loc[:, 1:100]  # 将特征数据存储在x中，表格前13列为特征,
y = data.loc[:, 101:425]  # 将标签数据存储在y中，表格最后一列为标签
print(x)
'''对每列（特征）归一化'''
# from sklearn.preprocessing import MinMaxScaler  # 导入归一化模块
#
# # feature_range控制压缩数据范围，默认[0,1]
# scaler = MinMaxScaler(feature_range=[0, 1])  # 实例化，调整0,1的数值可以改变归一化范围
#
# X = scaler.fit_transform(x)  # 将标签归一化到0,1之间
# Y = scaler.fit_transform(y)  # 将特征归于化到0,1之间

# x = scaler.inverse_transform(X) # 将数据恢复至归一化之前

'''对每列数据执行标准化'''

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # 实例化
X = scaler.fit_transform(x)  # 标准化特征

Y_scaler = StandardScaler()  # 实例化
Y = Y_scaler.fit_transform(y)  # 标准化标签


# x = scaler.inverse_transform(X) # 这行代码可以将数据恢复至标准化之前

import torch

X = torch.tensor(X, dtype=torch.float32)  # 将数据集转换成torch能识别的格式
Y = torch.tensor(Y, dtype=torch.float32)
torch_dataset = torch.utils.data.TensorDataset(X, Y)  # 组成torch专门的数据库
batch_size = 43  # 设置批次大小

# 划分训练集测试集与验证集
torch.manual_seed(seed=2021)  # 设置随机种子分关键，不然每次划分的数据集都不一样，不利于结果复现
train_validaion, test = torch.utils.data.random_split(
    torch_dataset,
    [4300, 554],
)  # 先将数据集拆分为训练集+验证集（共4300组），测试集（554组）
train, validation = torch.utils.data.random_split(
    train_validaion, [3800, 500])  # 再将训练集+验证集拆分为训练集3800，测试集500

# 再将训练集划分批次，每batch_size个数据一批（测试集与验证集不划分批次）
train_data = torch.utils.data.DataLoader(train,
                                         batch_size=batch_size,
                                         shuffle=True)

'''训练部分'''
import torch.optim as optim

feature_number = 100  # 设置特征数目
out_prediction = 325  # 设置输出数目
learning_rate = 0.01  # 设置学习率
epochs = 500  # 设置训练代数


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output, n_neuron1, n_neuron2,
                 n_layer):  # n_feature为特征数目，这个数字不能随便取,n_output为特征对应的输出数目，也不能随便取
        self.n_feature = n_feature
        self.n_output = n_output
        self.n_neuron1 = n_neuron1
        self.n_neuron2 = n_neuron2
        self.n_layer = n_layer
        super(Net, self).__init__()
        self.input_layer = torch.nn.Linear(self.n_feature, self.n_neuron1)  # 输入层
        self.hidden1 = torch.nn.Linear(self.n_neuron1, self.n_neuron2)  # 1类隐藏层
        self.hidden2 = torch.nn.Linear(self.n_neuron2, self.n_neuron2)  # 2类隐藏
        self.predict = torch.nn.Linear(self.n_neuron2, self.n_output)  # 输出层

    def forward(self, x):
        '''定义前向传递过程'''
        out = self.input_layer(x)
        out = torch.relu(out)  # 使用relu函数非线性激活
        out = self.hidden1(out)
        out = torch.relu(out)
        for i in range(self.n_layer):
            out = self.hidden2(out)
            out = torch.relu(out)
        out = self.predict(  # 回归问题最后一层不需要激活函数
            out
        )  # 除去feature_number与out_prediction不能随便取，隐藏层数与其他神经元数目均可以适当调整以得到最佳预测效果
        return out


net = Net(n_feature=feature_number,
          n_output=out_prediction,
          n_layer=1,
          n_neuron1=20,
          n_neuron2=20)  # 这里直接确定了隐藏层数目以及神经元数目，实际操作中需要遍历
optimizer = optim.Adam(net.parameters(), learning_rate)  # 使用Adam算法更新参数
criteon = torch.nn.MSELoss()  # 误差计算公式，回归问题采用均方误差
# print(net.state_dict().keys())

loss_list = []
for epoch in range(epochs):  # 整个数据集迭代次数
    net.train()  # 启动训练模式
    for batch_idx, (data, target) in enumerate(train_data):
        logits = net.forward(data)  # 前向计算结果（预测结果）
        loss = criteon(logits, target)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 后向传递过程
        optimizer.step()  # 优化权重与偏差矩阵

    logit = []  # 这个是验证集，可以根据验证集的结果进行调参，这里根据验证集的结果选取最优的神经网络层数与神经元数目
    target = []
    net.eval()  # 启动测试模式
    for data, targets in validation:  # 输出验证集的平均误差
        logits = net.forward(data).detach().numpy()
        targets = targets.detach().numpy()
        target.append(targets[0])
        logit.append(logits[0])
    average_loss = criteon(torch.tensor(logit), torch.tensor(target))
    loss_list.append(average_loss)
    print('\nTrain Epoch:{} for the MSE of VAL'.format(average_loss))

torch.save(net, 'net4.pkl')

#   loss曲线
import matplotlib.pyplot as plt

x1 = range(0, epochs)
y1 = loss_list
plt.subplot()
plt.plot(x1,y1)
plt.title('Val MSE vs. epoches')
plt.ylabel('Val MSE')
plt.xlabel('epoches')
plt.show()


import numpy as np

prediction = []
test_y = []
net.eval()  # 启动测试模式
for test_x, test_ys in test:
    predictions = net(test_x)
    predictions = predictions.detach().numpy()
    prediction.append(predictions)
    test_ys.detach().numpy()
    test_y.append(test_ys)
print('--------------------------归一化预测----------------------')
print(prediction[1],type(prediction))
prediction = Y_scaler.inverse_transform(np.array(prediction).reshape(
    -1, 325))  # 将数据恢复至归一化之前
print('------------------------反归一化之后-----------------------------')
print(prediction[1],type(prediction))
test_y = torch.tensor([item.detach().numpy() for item in test_y])
test_y = Y_scaler.inverse_transform(np.array(test_y).reshape(-1, 325))
# print(test_y,len(test_y))

#   均方误差计算
test_loss = criteon(torch.tensor(prediction, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
print('测试集均方误差：', test_loss.detach().numpy())

#   测试集预测压力与epanet
# print(prediction[1])
# print(test_y[1])
# x1 = range(0,325)
# y1 = test_loss_each
# plt.subplot()
# plt.plot(x1,y1)
# plt.title('Test accuracy vs. such_sence')
# plt.ylabel('Test accuracy')
# plt.xlabel('such_sence')
# plt.show()
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
acc = prediction[1]  # 实际值数据

pre = test_y[1]  # 预测值数据

plt.plot(acc, color="r", label="acc")  # 颜色表示

plt.plot(pre, color=(0, 0, 0), label="pre")

plt.xlabel("输出神经元")  # x轴命名表示

plt.ylabel("压力值")  # y轴命名表示

plt.axis([0, 325, 25, 90])  # 设定x轴 y轴的范围

plt.title("实际值与预测值折线图")

plt.legend()  # 增加图例

plt.show()  # 显示图片

# v = np.concatenate(np.square(abs(np.subtract(acc, pre)))).sum()
# print(v)