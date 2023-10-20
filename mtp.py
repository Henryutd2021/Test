from matplotlib import pyplot as plt
import random

#plt.figure(figsize=(20, 12), dpi=80)
x = range(1, 121)
y = [random.randint(20, 35) for _ in range(120)]
#y = [15, 13, 14.5, 17, 20, 25, 26, 26, 24, 22, 18, 15]
plt.plot(x, y)
# plt.xticks(range(2, 26, 1))# 调整X轴的刻度,可迭代参数,可以加入标签旋转参数，让刻度标签竖着表示
# plt.yticks(y)# 调整X轴的刻度
plt.xticks(x, )
plt.xlabel
plt.title
plt.legend#图例
#plt.savefig("./t1.png")
plt.show()