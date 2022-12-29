import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 支持中文
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['font.sans-serif']=['Songti SC'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


data = pd.read_csv('/Users/mac/Asiainfo/性能测试/RabbitMQ/rabbitmq-600-1.csv')
print (data.head())
print (data.tail())

# show the column names!
print([column for column in data])
# print(data)
# print(data.max)

# datanp = np.array(data)

aves = data["ave"]
maxs = data['max']
sessions = data['sessions']

aves2 = []
maxs2 = []
sessions2 = []
for i in range(len(sessions)):
    print( "No. ",i)
    print (sessions[i])
    if i > 0 and sessions[i] == sessions[i-1]:
        print (sessions[i],sessions[i-1])
        aves2[len(aves2)-1] = ((aves[i]+aves2[len(aves2)-1])/2)
        maxs2[len(maxs2)-1] = ((maxs[i]+maxs2[len(maxs2)-1])/2)
        print (len(aves2))
    else:
        sessions2.append(sessions[i])
        aves2.append(aves[i])
        maxs2.append(maxs[i])

print(sessions2)
print(aves2)
print(maxs2)
# 散点
# plt.scatter(sessions, aves)
# plt.scatter(sessions, maxs)

# 折线
plt.plot(sessions2, aves2, label='平均速率(/s)')
plt.plot(sessions2, maxs2, label='最大速率(/s)')
# plt.bar ( np.array(len(sessions)), sessions)
# plt.xticks(np.arrange(len(sessions)), ave)

plt.legend()  # 显示右上角的那个label,即上面的label = 'sinx'
plt.legend(loc='upper right', frameon=True)
plt.xlabel('并发数')  # 设置x轴的label，pyplot模块提供了很直接的方法，内部也是调用的上面当然讲述的面向对象的方式来设置；
plt.ylabel('消息生产速率(/s)')  # 设置y轴的label;

plt.title('RabbitMQ Benchmark')
plt.grid(True)
plt.show()

# x = data.sessions
# y = data.max
# plt.scatter(x, y)
# plt.scatter(data.sessions, data.max)
# plt.plot(y, x)

# date,endpoint,nodes,p1,p2,sessions,msgbytes,msgpersession,msgtotal,err,during, ave, max,vmspec
# 2021-11-29,ha2, node4+5+6+7+8+9, , ,100,71,600, 60.00(k),0,5(s),11632,14430,2cpu4g
#             日期 连接节点              实际负载 Unnamed: 3  ...       耗时   平均速率   最大速率    vm规格
# 0   2021-11-29  ha2   node4+5+6+7+8+9             ...     5(s)  11632  14430  2cpu4g
# 1   2021-11-29  ha2   node4+5+6+7+8+9             ...     5(s)  11545  15197  2cpu4g




