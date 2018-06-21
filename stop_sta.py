# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:08:22 2018

@author: DELL
"""
#二分类模型预测流利度

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

path = r'D:\口述题\重新标注50.xlsx'
path2 = r'D:\口述题\有效时长500_result.xls'
path3 = r'D:\口述题\3medium+bad.xlsx'
path4 = r'D:\口述题\3good.xlsx'

plt.style.use('seaborn')#使用ggplot样式
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False


df_data_text = pd.read_excel(path2)
df_data_bad = pd.read_excel(path3)
df_data_good = pd.read_excel(path4)

'''
#统计精准50条样本中停用词分布情况
df_data = pd.read_excel(path)
df_data.rename(columns={'来源':'文件名'},inplace=True)
temp = pd.merge(df_data, df_data_text)
df_data = temp[['文件名', '问题', '打分人1', '打分人2', '打分人3', '识别文本']]
rule = ((df_data['打分人1']=='好') & (df_data['打分人2'] =='好')) | ((df_data['打分人2'] == '好') & (df_data['打分人3']=='好')) | ((df_data['打分人1'] == '好') & (df_data['打分人3']=='好'))
data_excellent = df_data[rule]

df_data['label'] = 0
df_data['label'].loc[data_excellent.index] = 1

import thulac
thu = thulac.thulac(seg_only=True)
df_data['words'] = df_data['识别文本'].map(lambda x: thu.cut(x, text=True))

stopwords = []
f = open('./stopwords_chinese.txt',mode='r',encoding='utf-8')
for line in f.readlines():
    stopwords.append(line.strip())
stopwords = set(stopwords)
df_data['stopword_count'] = df_data['words'].map(lambda x: len(set(x) & stopwords))
'''
    
df_data_bad['label'] = 0
df_data_good['label'] = 1
df_data = pd.concat([df_data_good,df_data_bad],axis=0)
df_data.rename(columns={'来源':'文件名'},inplace=True)
temp = pd.merge(df_data, df_data_text)
df_data = temp[['文件名','语音语速(字/分钟)', '文件语速(字/分钟)', '停顿次数(<5)', '停顿次数(>5)', '识别文本', 'label']]
df_data['words'] = df_data['识别文本'].map(lambda x: thu.cut(x, text=True))
df_data['stopword_count'] = df_data['words'].map(lambda x: len(set(x) & stopwords))

def gen_con_repeat(s,N):
    #只计算重复的连续值
    repeat_words = []
    repeat_num = 0
    two_gram_dict = []
    for i in range(len(s)):
        two_gram_dict.append([s[i:i+N]])
    for index,value in enumerate(two_gram_dict):
        if index < len(two_gram_dict)-1:
            if two_gram_dict[index] == two_gram_dict[index + 1]:
                if not str.isdigit(str(two_gram_dict[index])): #去除数字    
                    repeat_num += 1
                    repeat_words.append(two_gram_dict[index])
    return repeat_num

df_data["con_repeat_num_1"] = df_data['识别文本'].map(lambda x: gen_con_repeat(x,1))
df_data["con_repeat_num_2"] = df_data['识别文本'].map(lambda x: gen_con_repeat(x,2))

X = df_data[['文件名','语音语速(字/分钟)', '文件语速(字/分钟)', '停顿次数(<5)', '停顿次数(>5)', 'stopword_count','con_repeat_num_1','con_repeat_num_2','label']]
from sklearn.utils import shuffle
X = shuffle(X)
y = X['label']
X = X.drop(['label', '文件名'],axis=1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score

sca = StandardScaler()
X_std =  sca.fit_transform(X)
X_train,X_val,y_train,y_val = train_test_split( X_std, y, test_size=0.25, random_state=2018)
#model = LogisticRegression()
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print('准确率: ', accuracy_score(y_val, y_pred)) #72% -80%
print('AUC score: ', roc_auc_score(y_val, y_pred)) 

def draw_fea_importance(model,features_list):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    fi_threshold = 1   
    important_idx = np.where(feature_importance > fi_threshold)[0]
    important_features = features_list[important_idx]
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    #get the figure about important features
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.title('Feature Importance')
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]],color='b',align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.draw()
    plt.show()