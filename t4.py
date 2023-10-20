import random
import math

icount = 0
iter_num = 10000000
# x_axis = []
# y_axis = []
for _ in range(iter_num):
    x = random.random()
    y = random.random()

    if math.sqrt(x ** 2 + y ** 2) <= 1:
        icount += 1
print('Pi = ', float(4 * icount) / float(iter_num))
