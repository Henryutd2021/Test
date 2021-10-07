import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure(2)                                #新开一个窗口
ax1 = fig.add_subplot(1,2,1,polar=True)                  #启动一个极坐标子图
theta=np.arange(0,2*np.pi,0.02)              #角度数列值
ax1.plot(theta,2*np.ones_like(theta),lw=2)   #画图，参数：角度，半径，lw线宽
ax1.plot(theta,theta/6,linestyle='--',lw=2)           #画图，参数：角度，半径，linestyle样式，lw线宽

ax2 = fig.add_subplot(1,2,2,polar=True)                  #启动一个极坐标子图
ax2.plot(theta,np.cos(5*theta),linestyle='--',lw=2)
ax2.plot(theta,2*np.cos(4*theta),lw=2)

ax2.set_rgrids(np.arange(0.2,2,0.2),angle=45)   #距离网格轴，轴线刻度和显示位置
ax2.set_thetagrids([0,45,90])                   #角度网格轴，范围0-360度

plt.show()