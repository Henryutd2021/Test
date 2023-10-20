import math


class Square:  # 正方形

    def __init__(self, l):
        self.length = l  # 边长

    def __setattr__(self, key, value):
        s = f"调用__setattr__, key={key}, value={value}"
        print(s)

        if key == "length" and value > 0:
            self.__dict__["length"] = value
            self.__dict__["perimeter"] = value * 4
            self.__dict__["area"] = value ** 2

        if key == "perimeter" and value > 0:
            self.__dict__["length"] = value / 4
            self.__dict__["perimeter"] = value
            self.__dict__["area"] = (value / 4) ** 2

        if key == "area" and value > 0:
            self.__dict__["length"] = math.sqrt(value)
            self.__dict__["perimeter"] = math.sqrt(value) * 4
            self.__dict__["area"] = value


sq = Square(10)
print("length =", sq.length)
print("perimeter =", sq.perimeter)
print("area =", sq.area)
print("-------")

# sq.perimeter = 12
# print("length =", sq.length)
# print("perimeter =", sq.perimeter)
# print("area =", sq.area)
# print("########")
#
# sq.area = 25
# print("length =", sq.length)
# print("perimeter =", sq.perimeter)
# print("area =", sq.area)

# # 执行结果
# 调用__setattr__, key=length, value=10
# length = 10
# perimeter = 40
# area = 100
# -------
# 调用__setattr__, key=perimeter, value=12
# length = 3.0
# perimeter = 12
# area = 9.0
# ########
# 调用__setattr__, key=area, value=25
# length = 5.0
# perimeter = 20.0
# area = 25