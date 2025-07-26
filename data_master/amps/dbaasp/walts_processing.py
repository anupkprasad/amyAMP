#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:11:30 2023

@author: anupkumar
"""
from amp_gan.utils import ProcessSeqs, write_fasta
import csv

#%%variables

path_cur = "/media/anupkumar/Backup Plus/project/ML_project/data_master/amps/dbaasp/"

path_result = "/media/anupkumar/Backup Plus/project/ML_project/pytorch_AMP_BiGAN/amp_gan/results/"

#%%   
f = open(path_cur + 'WaltzJob_1698567046.txt')
data = []
d = ""
for x in f:
    if x[0] == ">":
        data.append(d.split())
        d = ""
    d = d+x[:-1]
f.close()

data = data[1:]

### data containg APR has length > 7

APR_data = []
id_list = []
for d in data:
    if len(d)>7:
        APR_data.append(d)
        id_list.append(d[0][1:])

dbassp_file = [path_cur + "dbaasp.fasta"]
dbassp = ProcessSeqs(*dbassp_file).get_seqs()

APR_dbaasp = {}

for key, value in dbassp.items():
    if key in id_list:
        APR_dbaasp[key] = value

write_fasta(APR_dbaasp,path_cur + "dbaasp_APR.fasta" )

processed_dbassp_APR = ProcessSeqs(*[path_cur + "dbaasp_APR.fasta"]).get_unique_seqs()
write_fasta(processed_dbassp_APR,path_cur + "dbaasp_APR_processed.fasta" )



