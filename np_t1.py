import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib

a = np.array([1, 2, 3])
# np.zeros(3, 4)
b = np.array([[1, 2, 3, 4, 5],[2, 3, 4, 5]])
c = np.array([(1, 2, 3, 4, 5),(2, 3, 4, 5)])
print(np.eye(4))
print(b)
print(type(b[1]))
print(type(c[1]))
print(type(c))
dt = np.dtype(np.int32)
print(dt)
a=np.arange(0,12)
a.shape=(3,4)
print(a)
print(a[0:2,1:3])
print (a[...,1:])

arr = np.arange(24).reshape((2, 3, 4))
print(arr)
#print(arr[1:3, 1:3])

print(type(arr[0:1,0:1,1]) )
x=np.arange(48).reshape((3, 4, 4))
print(x)
print(x[0:2,1:3, 1:2])
y=np.arange(48).reshape((3,2, 8))
print(y)
#print(y[[0,2,5,3], [0,2,3,7]])
# print(x[[4, 2, 1, 7]])
print(y[[1,2,0,1],[0,1,1,1],[3,5,0,7]])
print(y[y>10])
b = np.array([1, 2, 3, 4])
c = np.array([2, 3, 4, 5, 6, 7], dtype=int)
# #print(b*c)
arr = np.arange(24).reshape((6, 4))
arr1 = np.arange(24).reshape((4, 6))
print(arr1)
print(c)
# print(b+arr)
# print(c+arr1)
for n, m in np.nditer([c, arr1]):
    print("%d:%d" %(m,n), end=",")
arr = np.arange(24).reshape((6, 4))
for x in np.nditer(arr.T):
    print(x, end=",")
a = np.arange(0, 60, 5).reshape(3, 4)
#print(a)
b = a.T
# print(b)
# c = b.copy(order='C')
# print(c)
# for n in np.nditer(c):
#     print(n, end=",")
d = b.copy(order='F')
#print(d)
# for m in np.nditer(d, op_flags=["readwrite"]):
#     m[...] = 2*m
#     print(m, end=",")
# print("\n")
print(a)
for n in np.nditer(a, flags=["external_loop"], order="F"):
    print(n, end=",")
print(np.matlib.empty((2,2)))
print (np.matlib.eye(n =  3, M =  4, k =  0, dtype =  float))
print (np.matlib.identity(5, dtype =  float))
# 计算正弦和余弦曲线上的点的 x 和 y 坐标
x = np.arange(0,  3  * np.pi,  0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)
# 建立 subplot 网格，高为 2，宽为 1
# 激活第一个 subplot
plt.subplot(2,  1,  1)
# 绘制第一个图像
plt.plot(x, y_sin,".m")
plt.plot(x, y_cos,":c")

plt.title('Sin/Cos')
# 将第二个 subplot 激活，并绘制第二个图像
plt.subplot(2,  1,  2)
plt.plot(x, y_tan,".y")
# plt.plot(x, y_cos)
plt.title('Tan')
# # 展示图像
plt.show()
# 录入了四位同学的成绩，按照总分排序，总分相同时语文高的优先
math    = (10, 20, 50, 10)
chinese = (30, 50, 40, 60)
total   = (40, 70, 90, 70)
# 将优先级高的项放在后面
ind = np.lexsort((math, chinese, total))

for i in ind:
    print(total[i],chinese[i],math[i])