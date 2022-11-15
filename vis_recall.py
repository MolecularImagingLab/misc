# Load libraries
import os, sys, glob
import pandas as pd
import numpy as np

def grab_subs(path):
    subs_paths = os.listdir(path)
    subs = sorted([p for p in subs_paths if 'AVL' in p])
    return subs

def grab_parord(subs):
    par_order = {}
    for s in subs:
        grab_par = glob.glob(os.path.join(path, s,'*csv'))
        filt_rec = [p for p in grab_par if 'RECALL' not in p]
        oresults = sorted (filt_rec, key = lambda filt_rec: int(filt_rec[-12:-4]))
        par_order[s] = [o[-25:-24] for o in oresults]
    return par_order

def collect_recalls(path, subs, par_order):
    recalls = []
    recode = {'very_unlikely':1,'somewhat_unlikely':2,'somewhat_likely':3,'very_likely':4}
    # Load recall files and append results
    for s in subs:
        grab_par = glob.glob(os.path.join(path, s,'*csv'))
        # Filter for recall files
        filt_rec = [p for p in grab_par if 'RECALL' in p]
        # Sort recall by time
        recall_ord = sorted (filt_rec, key = lambda filt_rec: int(filt_rec[-12:-4]))
        # Add data to master list
        for recall in recall_ord:
            m = pd.read_csv(recall)
            n = pd.DataFrame.replace(m,to_replace=recode)
            # Get ID, session == index
            n['ID'] = s
            n['Session'] = str(recall_ord.index(recall) +1)
            # Get status
            if int(s[-3:]) < 100 or 199 < int(s[-3:]) < 299:
                n['Status'] = 'control'
            else:
                n['Status'] = 'FDR'
            # Get run == index
            for order in par_order[s]:
                x = str(par_order[s].index(order) +1)
                n.loc[n['Stimulus'].str.startswith(order), 'Run'] = x
            recalls.append(n)
    # Concat all subjects
    recall_vals = pd.concat([rec for rec in recalls], ignore_index=True)
    recall_vals.fillna('Sham', inplace=True)
    # Sort by letter
    for letter in ['A','B','C','D','E']:
        recall_vals.loc[recall_vals['Stimulus'].str.startswith(letter), 'Run Type'] = letter
    # Aggregate csp and csm events
    recall_vals.loc[recall_vals['Stimulus'].str.endswith('CSm'), 'Stimulus'] = 'CSm'
    recall_vals.loc[recall_vals['Stimulus'].str.endswith('CSp'), 'Stimulus'] = 'CSp'
    return recall_vals

def learning_idxs(df):
    agg = df.groupby(['ID','Stimulus','Run', 'Run Type'], as_index=False).mean()
    diffs = []
    for s in df['ID'].unique():
        for r in ['1', '2', '3', '4', 'Sham']:
            try:
                csp = float(agg.loc[(agg['ID'] == s) & (agg['Stimulus'] == 'CSp') & (agg['Run'] == r), 'Response'].values)
                csm = float(agg.loc[(agg['ID'] == s) & (agg['Stimulus'] == 'CSm') & (agg['Run'] == r), 'Response'].values)
                diff = csp - csm
                di = (csp - csm)/(csp + csm)
                runtype = (agg.loc[(agg['ID'] == s) & (agg['Run'] == r), 'Run Type'].values)[0]
                diffs.append([s, r, diff, di, runtype])
            except:
                pass
    learning = pd.DataFrame(diffs, columns=['ID', 'Run', 'Contrast', 'DI', 'Run Type'])
    return learning

path = '/Users/ramihamati/Documents/PhD_Work/AVL/PAR_FILES'
subs = grab_subs(path)
par_order = grab_parord(subs)
recall_vals = collect_recalls(path, subs, par_order)
# Filter data for healthy controls
hc = recall_vals.loc[(recall_vals.Status == 'control') & (recall_vals.Session == '1')]
# Get learning indices
learning = learning_idxs(hc)
learning.to_csv('/Users/ramihamati/Downloads/learning.csv')
