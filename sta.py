# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:03:16 2018

@author: DELL
"""
#统计打分情况的一致性

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

path = r'D:\NLP\口述题\口述题汇总.xlsx'
plt.style.use('seaborn')#使用ggplot样式
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False

df_data = pd.read_excel(path)
df_data.dropna(inplace=True)


def draw_pie(df_data,feature_name, i):
    df_data['same3'] = 0
    same3 = df_data[(df_data['评价人1'] == df_data['评价人2']) & (df_data['评价人2'] == df_data['评价人3'])]
    df_data['same3'].loc[same3.index] = 1

    X = [df_data[df_data['same3']== 1].shape[0], df_data[df_data['same3']== 0].shape[0]]
    labels = ['三者一致','三者不一致']
    plt.subplot(1,3,i)
    plt.pie(X, labels=labels, autopct='%1.2f%%')
    plt.title(feature_name)
    #plt.show()

def draw_pie2(df_data,feature_name, i):
    rule1 = (df_data['评价人1'] == df_data['评价人2']) | (df_data['评价人2'] == df_data['评价人3']) | (df_data['评价人1'] == df_data['评价人3'])
    same2 = df_data[rule1]
    df_data['same2'] = 0
    df_data['same2'].loc[same2.index] = 1

    X = [df_data[df_data['same2']== 1].shape[0], df_data[df_data['same2']== 0].shape[0]]
    print(X)
    labels = ['两者一致','任两者不一致']
    plt.subplot(1,3,i)
    plt.pie(X, labels=labels, autopct='%1.2f%%')
    plt.title(feature_name)
    #plt.show()

#数字化标签
#df_data[['评价人1','评价人2','评价人3']]=df_data[['评价人1','评价人2','评价人3']].apply(LabelEncoder().fit_transform)

#绘制
def draw_bar(df_data):
    t1 = pd.DataFrame(df_data['评价人1'].value_counts(),columns=['评价人1'])
    t2 = pd.DataFrame(df_data['评价人2'].value_counts(),columns=['评价人2'])
    t3 = pd.DataFrame(df_data['评价人3'].value_counts(),columns=['评价人3'])
    pd.concat([t1,t2,t3],axis=1)
    t = pd.concat([t1,t2,t3],axis=1)
    t.plot.bar()
    
''' 分布绘制三个特征的分布图
i = 1
for idx,group in df_data.groupby(by='问题',axis=0):
    draw_pie(group,idx,i)
    i += 1
'''
