l = [list(range(1, 5)), list(range(2, 6))]
w = []


def sqr(a):
    for i in a:
        w.append(i*i)
    print(w)


for j in l:
    sqr(j)
    w = []

def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)
    
person('Adam', 45, gender='M', job='Engineer')

extra = {'city': 'Beijing', 'job': 'Engineer'}

person('Jack', 24, city=extra['city'], job=extra['job'])

path = 'C:/Users/hxl210015/REopt_Lite_API-hes_nuclear/hes-nuclear/results/base_case.json'