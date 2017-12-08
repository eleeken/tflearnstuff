#加激活函数的方法1：mode.add(Activation(''))
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x_data=np.linspace(-0.5,0.5,200)
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

#构建一个顺序模型
model=Sequential()

#在模型中添加一个全连接层
#units是输出维度,input_dim是输入维度(shift+两次tab查看函数参数)
#输入1个神经元,隐藏层10个神经元,输出层1个神经元
model.add(Dense(units=10,input_dim=1))
model.add(Activation('tanh'))   #增加非线性激活函数
model.add(Dense(units=1))   #默认连接上一层input_dim=10
model.add(Activation('tanh'))

#定义优化算法(修改学习率)
defsgd=SGD(lr=0.3)

#编译模型
model.compile(optimizer=defsgd,loss='mse')   #optimizer参数设置优化器,loss设置目标函数

#训练模型
for step in range(3001):
    #每次训练一个批次
    cost=model.train_on_batch(x_data,y_data)
    #每500个batch打印一个cost值
    if step%500==0:
        print('cost:',cost)

#打印权值和偏置值
W,b=model.layers[0].get_weights()   #layers[0]只有一个网络层
print('W:',W,'b:',b)

#x_data输入网络中，得到预测值y_pred
y_pred=model.predict(x_data)

plt.scatter(x_data,y_data)

plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()