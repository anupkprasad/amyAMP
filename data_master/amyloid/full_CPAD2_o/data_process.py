#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:11:30 2023

@author: anupkumar
"""
# Python program to read
# json file

import pandas as pd 
import numpy as np
path =  "/media/anupkumar/Backup Plus/project/ML_project/data_master/amyloid/full_CPAD2_o/"
df = pd.read_excel("aggregating peptides.xlsx") 

df1 = df.loc[:, ['Entry', 'Peptide', 'Classification']] 

df1 = np.array(df1)

seqences = {}
for i in range(len(df1)):
    if df1[i,2] == "Amyloid" :
        sid = df1[i,0]
        seq = df1[i,1]
        seqences[sid] = seq

## seqences contains unique seq
values  = list(seqences.values())
values = set(values)
### use write_fasta def of utils.py
# write_fasta(CARs_seqs, "/media/anupkumar/Backup Plus/project/ML_project/data_master/amyloid/all_json_waltz_output_disprot/first.fasta")


print("Done")
