l = [list(range(1, 5)), list(range(2, 6))]
w = []


def sqr(a):
    for i in a:
        w.append(i*i)
    print(w)


for j in l:
    sqr(j)
    w = []

