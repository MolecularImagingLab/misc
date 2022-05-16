# Load libraries
import os, sys, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns

# Grab subjects
path = '/Users/ramihamati/Documents/PhD_Work/AVL/PAR_FILES'
subs_paths = glob.glob(os.path.join(path, 'AVL*'))
subs = [s[-7:] if len(s) == 58 else s[-10:] for s in subs_paths]

# Grab run order
par_order = {}
for s in subs:
    grab_par = glob.glob(os.path.join(path, s,'*csv'))
    filt_rec = [p for p in grab_par if 'RECALL' not in p]
    oresults = sorted (filt_rec, key = lambda filt_rec: int(filt_rec[-12:-4]))
    par_order[s] = [o[-25:-24] for o in oresults]

# Create an empty dataframe
data = pd.DataFrame(columns=['Stimulus','Response','RT','ID','Session','Run'])
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
        # Group by Stimulus type, according to ID and session
        #n = n.groupby(['ID','Session','Stimulus'], as_index=False).mean()
        # Get status
        if int(s[-3:]) < 100 or 199 < int(s[-3:]) < 299:
            n['Status'] = 'control'
        else:
            n['Status'] = 'FDR'
        # Get run == index
        for order in par_order[s]:
            x = str(par_order[s].index(order) +1)
            n.loc[n['Stimulus'].str.startswith(order), 'Run'] = x
        data = data.append(n, ignore_index=True,sort=False)
data.fillna('Sham', inplace=True)

# Aggregate csp and csm events
data.loc[data['Stimulus'].str.endswith('m'), 'Stimulus'] = 'CSm'
data.loc[data['Stimulus'].str.endswith('p'), 'Stimulus'] = 'CSp'
# Filter data for healthy controls
hc = data.loc[(data['Status'] == 'control') & (data['Session'] == '1')]

# Get learning indices
agg = hc.groupby(['ID','Stimulus','Run'], as_index=False).mean()
diffs = []
for s in agg['ID'].unique():
    for r in ['1', '2', '3', '4', 'Sham']:
        csp = float(agg.loc[(agg['ID'] == s) & (agg['Stimulus'] == 'CSp') & (agg['Run'] == r), 'Response'].values)
        csm = float(agg.loc[(agg['ID'] == s) & (agg['Stimulus'] == 'CSm') & (agg['Run'] == r), 'Response'].values)
        diff = csp - csm
        di = (csp - csm)/(csp + csm)
        diffs.append([s, r, diff, di])
learning = pd.DataFrame(diffs, columns=['ID', 'Run', 'Contrast', 'DI'])
