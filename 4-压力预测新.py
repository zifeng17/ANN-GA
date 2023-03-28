#   _*_coding_*_:utf-8

import pandas as pd

#----------------------------------数据导入---------------------------------------------
data_ture = pd.read_excel('D:\\pycharmProject\\pytorch\\ANN\\shuju\\测试.xlsx',
                     header=None, index_col=None)
data_ture_1 = pd.read_excel('D:\\pycharmProject\\pytorch\\ANN\\shuju\\1.xlsx',
                     header=None, index_col=None)
print(data_ture)
x = data_ture.loc[:, 1:100]  # 将特征数据存储在x中，表格前13列为特征,
Y_1 = data_ture_1.loc[1:2,101:425]
print(x)
print(Y_1)

#------------------------------特征化标签---------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # 实例化
X = scaler.fit_transform(x)  # 标准化特征
Y_scaler = StandardScaler()  # 实例化
Y_1 = Y_scaler.fit_transform(Y_1)
Y_1 = Y_scaler.inverse_transform(Y_1)
#-----------------------------模型加载--------------------------------
import torch
import numpy as np
X = torch.tensor(X, dtype=torch.float32)  # 将数据集转换成torch能识别的格式

torch_dataset = torch.utils.data.TensorDataset(X)  # 组成torch专门的数据库

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

#-------------------------------获取预测值----------------------------------------
prediction = []

net = torch.load('net4.pkl')
net.eval()  # 启动测试模式
for test_x in torch_dataset:
    test_x = torch.tensor([item.detach().numpy() for item in test_x])
    predictions = net(test_x)
    predictions = predictions.detach().numpy()
    prediction.append(predictions)

#--------------------------反归一化预测值----------------------------------------
# print(prediction, type(prediction[0]))
prediction = prediction[0]
prediction = Y_scaler.inverse_transform(np.array(prediction).reshape(-1,325))  # 将数据恢复至归一化之前

# print('-------------------------------------------------')
prediction = prediction.reshape(25,13)
Y_1 = np.array(Y_1).reshape(25,26)
print(prediction,prediction.shape)
# df = pd.DataFrame(data = prediction)
# df1 = pd.DataFrame(data = Y_1)
# print(df.to_string())
# print(df1.to_string())
# mse = df - df1
# print(mse.to_string())

# df.to_excel( '预测压力(net4).xlsx')