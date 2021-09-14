#!/usr/bin/env python
import sys
import pandas as pd
# Define input
in_file = sys.argv[1]
# Read in csv
df = pd.read_csv(in_file, names=['onset','value','duration','weight'], sep=' ')
# Rearrange
df = df[['onset','duration','value','weight']]
# Add header, add tabs
df.to_csv(in_file, header=True, sep="\t", index=False)
