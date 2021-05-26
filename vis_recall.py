#!/usr/bin/env python

#load libraries
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
subjectpath = '/home/rami/Documents/sync_me/EmoSal/subjects'
subjects = os.listdir(subjectpath)
print(subjects)
subjects.remove('Subs.mat')
subjects.remove('AVL-002')
print(subjects)
# Grab recall files
results = []
for s in subjects:
    if(s.startswith('A')):
        recallpath = os.path.join('/home/rami/Documents/sync_me/EmoSal/subjects',s,'recall')
        files=os.listdir(recallpath)
        for f in files:
            if f.endswith('.csv'):
                results.append(f)
print(results)
# Create an empty dataframe
data = pd.DataFrame(columns=['Stimulus','Response','RT','ID'])
recode = {'very_unlikely':1,'somewhat_unlikely':2,'somewhat_likely':3,'very_likely':4}
# Load recall files and append results
recall = dict(zip(subjects,results))
for r in recall:
    m = pd.read_csv(os.path.join('/home/rami/Documents/sync_me/EmoSal/subjects',r,'recall',recall[r]))
    n = pd.DataFrame.replace(m,to_replace=recode)
    n['ID'] = recall[r][0:7]
    if int(r[4:7]) < 100:
        n['Status'] = 'control'
    elif int(r[4:7]) > 100:
        n['Status'] = 'FDR'
    data = data.append(n, ignore_index=True,sort=False)
# Unpivot table
o = pd.melt(data, id_vars =['ID','Stimulus'], value_vars =['Response'], var_name='Recall',value_name='Value')
# Define stim orders
subs = ['AVL-001','AVL-003', 'AVL-004', 'AVL-005', 'AVL-006', 'AVL-007', 'AVL-009', 'AVL-010', 'AVL-011', 'AVL-101', 'AVL-102', 'AVL-103', 'AVL-105']
firsts = ['A','A','A','A','B','C','E','E','B','B','C','D','D']
seconds = ['B','B','C','C','A','A','B','C','D','A','B','A','B']
thirds = ['C','E','B','E','D','D','D','A','A','E','D','B','C']
fourths = ['D','D','D','B','C','E','A','B','C','C','A','C','A']
stims = ['CSm','CSp']
# Grab stim order and table it
recall_1 = []
for i, s in enumerate(subs): #count, item
    for st in stims:
        val = o.loc[(o['ID']==s) & (o['Stimulus']==fourths[i]+st), 'Value'].to_list()
        recall_1.append([s, st, val[0]])
        recall_1.append([s, st, val[1]])
recall_run1 = pd.DataFrame(recall_1, columns=['ID','Stim','Value'])
# Plot stim presentations
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
# agg_csp.iloc[0:13].plot(x='event', y='csp')
# agg_csm.iloc[0:13].plot(x='event', y='csm')
# agg_contrast.iloc[0:13].plot(x='event', y='contrast')
fig, ax = plt.subplots(figsize=(6,6))
sns.set_style("ticks")
ax.tick_params(width=4, length=10, labelsize=28, bottom=False, pad=15)
sns.barplot(data=recall_run1, y='Value', x='Stim', ci=95, errwidth=4, capsize=0.2, edgecolor='black', linewidth=3, ax=ax)
ax.set_xlabel('Stimuli', fontsize=30)
ax.set_ylabel('Likert Scale (1-4)', fontsize=30)
ax.set_ylim(bottom=1, top=4)
ax.spines['left'].set_linewidth(4)
fig.suptitle('Recall (N=13)',fontsize=34, fontweight='bold')
sns.despine(bottom=True)
plt.tight_layout(pad=4)
plt.savefig('/home/rami/Downloads/run4_avg_recall.jpg', dpi=300)

# Separate implicit and explicit measures
length_implicit = len(o.loc[o['Recall']=='RT'])
length_explicit = len(o.loc[o['Recall']=='Response'])
implicit = o[length_implicit:length_implicit+length_explicit]
explicit = o[0:length_explicit]

# Plot implicit recall
sns.set_style('dark')
sns.set_palette('Paired')
alldata = (sns.barplot(x='Stimulus', y='Value', data=implicit, order=['ACSm','ACSp','BCSm','BCSp','CCSm','CCSp','DCSm','DCSp','ECSm','ECSp'], linewidth=2.5, capsize=0.2, edgecolor='black').set_title('Face Discrimination - Implicit Recall', fontsize=20)).get_figure()
plt.xlabel('Stimulus')
plt.xticks(rotation='vertical')
plt.ylabel('Reaction Time (s)')
plt.rcParams.update({'font.size': 20})
alldata.savefig(os.path.join('/home/lauri/Documents/temp/','implicit_recall.jpg'), dpi=300,  bbox_inches='tight')

# Plot explicit recall
sns.set_style('dark')
sns.set_palette('Paired')
alldata = (sns.barplot(x='Stimulus', y='Value', data=explicit, order=['ACSm','ACSp','BCSm','BCSp','CCSm','CCSp','DCSm','DCSp','ECSm','ECSp'], linewidth=2.5, capsize=0.2, edgecolor='black').set_title('Face Discrimination - Explicit Recall', fontsize=20)).get_figure()
plt.xlabel('Stimulus')
plt.xticks(rotation='vertical')
plt.ylabel('Likert Scale (1-4)')
plt.ylim(1, 4)
plt.rcParams.update({'font.size': 20})
alldata.savefig(os.path.join('/home/lauri/Documents/temp/','explicit_recall.jpg'), dpi=300,  bbox_inches='tight')
