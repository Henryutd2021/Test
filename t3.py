from itertools import product
# l = [1, 2, 3]
# for i in list(product(l, repeat=3)):
#     print(i)


# for x,y,z in product(['a','b','c'],['d','e','f'],['m','n']):
#     print(x,y,z)

for i, j in product((0, 2), (1, 3)):
    print(i, j)

