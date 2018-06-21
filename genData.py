# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:46:14 2018

@author: DELL
"""
import os
import re
import pandas as pd
import chardet
import codecs

'''
例如 367929_49642_aly-zby-rec-qingqing03-031_Part1.wav
fileID表示文件编号 dialogID表示031 partID表示1 
label: 0表示老师 1表示学生
'''

list_data = []

def genDialog(file_path, label, fileID):
    content = []
    chatest = chardet.detect(open(file_path,'rb').read())
    if chatest['encoding'] == 'utf-8':
        f = codecs.open(file_path, 'r', encoding = 'utf-8')
    elif chatest['encoding'] == 'UTF-8-SIG':
        f = codecs.open(file_path, 'r', encoding = 'UTF-8-SIG')
    elif chatest['encoding'] == 'GB2312':
        f = codecs.open(file_path, 'r', encoding = 'GB2312', errors='ignore')
    #f = codecs.open(file_path,'r', encoding='utf-8')
    for line in f.readlines():
        content.append(line)
    f.close()
    content = content[:-1]
    pattern = re.compile(r'\d+') #用于匹配至少一个++++++++++++++++数字
    temp = []
    for idx in range(len(content)  ):
        if idx%2 == 0: #偶数 说明是文本中的文件名行
            s = pattern.findall(content[idx].split('-')[-1]) #找到的第一个数字为第几轮对话 第二个数字为Part值
            try:
                temp = [fileID, s[0], s[1], label] #fileID dialogID partID label
            except:
                print('出错行:'+ str(idx+1) + '所属文件：' + file_path)
        else: #内容行
            temp.append(content[idx])
            list_data.append(temp)
            

path = r'D:/NLP/data/录音数据/all'
fileID = 0
for file_dir in os.listdir(path):
    if file_dir!= 'others':
        tmp_dir = os.path.join(path + '\\' + file_dir)
        for file in os.listdir(tmp_dir):
            tmp_file = tmp_dir + '\\' + file
            if os.path.isfile(tmp_file):
                if re.search('student',file): #学生讲话内容
                    genDialog(tmp_file, label=1, fileID=fileID)
                else:
                    genDialog(tmp_file, label=0, fileID=fileID)
    fileID += 1
 

            
df_data = pd.DataFrame(data = list_data, columns=['fileID','dialogID','partID','label','text'])

'''
#师生对话次数
dialog_num = []          
import matplotlib.pyplot as plt
plt.style.use('ggplot')#使用ggplot样式
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False

for idx,group in df_data.groupby(by=['fileID','dialogID']):
    dialog_num.append([idx[0],idx[1]])
dialog_num = pd.DataFrame(dialog_num,columns=['fileID','dialogID'])
dialog_num.groupby(by=['fileID']).count().plot.bar(title='有效对话次数统计')         
            
talk_num = []
for idx,group in df_data.groupby(by=['fileID']):
    talk_num.append([idx,group[group['label']==0].shape[0],group[group['label']==1].shape[0]])
talk_num = pd.DataFrame(talk_num,columns=['fileID','teacher','student'])
talk_num[['teacher','student']].plot.bar(title='师生对话次数对比')
talk_num['talk_rate'] = talk_num['student'] / talk_num['teacher']
talk_num['talk_rate'].plot.bar(title='师生对话次数比率')

talk_length = []
sigle_length = []
df_data['talk_length'] = df_data['text'].map(lambda x: len(x.replace(' ','').replace('\n','').replace('\r','')))
for idx,group in df_data.groupby(by=['fileID']):
    talk_length.append([group[group['label']==0]['talk_length'].mean(),group[group['label']==1]['talk_length'].mean()])
    sigle_length.append([group[(group['talk_length']==1) & (group['label']==0)].shape[0],group[(group['talk_length']==1) & (group['label']==1)].shape[0]])
talk_length = pd.DataFrame(talk_length,columns=['teacher_length','student_length'])
talk_length.plot.bar(title='师生对话平均长度对比')
sigle_length = pd.DataFrame(sigle_length,columns=['teacher_sig_length','student_sig_length'])
sigle_length.plot.bar(title='师生单字对话次数对比')

from snownlp import SnowNLP
def getSentiment(s):
    s = SnowNLP(s)
    return s.sentiments
df_data['sentiment'] = df_data['text'].map(lambda x:getSentiment(x) ) 
for idx,group in df_data.groupby(by=['fileID']):
    group['sentiment'].plot.hist(title='课堂'+ str(idx))
    plt.subplot(4,11,idx+1)
'''