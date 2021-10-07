l = list(range(1, 4))
a =[]
n=1
# a = []
# for i in l:
#     for j in l:
#         for k in l:
#             print([i, j, k])
#print(l[::])
#while n >0:
for i in l:
    #while n > 0:
    a.append(i)

    for j in l:

        a.append(j)


        for k in l:
            a.append(k)
            
            a.pop()


        a.pop()

    a.pop()
    #print(a)
    #     a=[]
    # n -= 1



    #print(a)



# def itr(n):
#     while n > 0:
#         l[::]
#
#
#         print([i, i, i])
# print(list(range(10))[::2])