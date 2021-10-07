import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import xlrd

#
# s = Series([1, 4, 'w', 't'])
# # print(s)
# # print(s.index)
# # print(s.values)
# data = {"name":['google','baidu','yahoo', 'ali'],"marks":[100,200,300, 400],"price":[1,2,3, 4]}
# f1 = DataFrame(data)
# print(f1)
# data = pd.read_excel("pdt.xlsx")
# score = pd.read_csv("Scores.csv")
# #print(data.head(3))
# #print(data.describe())
# # print(score.head(2))
# # print(score.describe())
# d = pd.read_excel("HW2.xlsx")
# print(d.head(5))

df = pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006],
 "date":pd.date_range('20130102', periods=6),
 "city":['Beijing ', 'SH', ' guangzhou ', 'Shenzhen', 'shanghai', 'BEIJING '],
 "age":[23,44,54,32,34,32],
 "category":['100-A','100-B','110-A','110-C','210-A','130-F'],
 "price":[1200,np.nan,2133,5433,np.nan,4432]},
 columns =['id','date','city','category','age','price'])
df1=pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006,1007,1008],
"gender":['male','female','male','female','male','female','male','female'],
"pay":['Y','N','Y','Y','N','Y','N','Y',],
"m-point":[10,12,20,40,40,40,30,20]})
df_inner=pd.merge(df,df1,how='inner')
# print(df)
# print(df.shape)
df_inner.to_csv('excel_to_python.csv')
df_inner.sample(n=6, replace=False)
