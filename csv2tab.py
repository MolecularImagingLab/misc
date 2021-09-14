#!/usr/bin/env python
import sys
import pandas as pd
# Define input
in_file = sys.argv[1]
# Read in csv
df = pd.read_csv(in_file, header=None)
# Add header, add tabs
df.to_csv(in_file, header=["onset","value","duration","weight"], sep="\t", index=False)
