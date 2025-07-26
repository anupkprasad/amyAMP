#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:11:30 2023

@author: anupkumar
"""
# Python program to read
# json file

import json

path =  "/media/anupkumar/Backup Plus/project/ML_project/data_master/amyloid/all_json_waltz_output_disprot/"
# Opening JSON file
f = open(path + '73-5_waltz_output_disprot.json')
f1 = open(path + '80_waltz_output_disprot.json')
f2 = open(path + '85_waltz_output_disprot.json')
# the JSON file is returning a list object of dictionary, load them in datasets
datasets = []
for file in [f,f1,f2]:
    for i in json.load(file):
        datasets.append(i)
# Closing file
f.close()
f1.close()
f2.close()

#######create dict of disprot_id and its CARs
CARs_datasets = {}
for l in datasets:
    uid = l["DisProt_ID"]
    seq = l["Seq_pep"]
    try:
        value = CARs_datasets[uid]
        CARs_datasets[uid].append(seq)
    except:
        CARs_datasets[uid] = [seq]

### combine all seqs together in list
seqences = []
for key, values in CARs_datasets.items():
    for value in values:
        seqences.append(value)
        
## remove duplicates using set and then make dict so can create fasta files
seqences = set(seqences)
seqences = list(seqences)

CARs_seqs = {}
for i in range(len(seqences)):
    CARs_seqs["car_"+str(i+1)] = seqences[i]

### use write_fasta def of utils.py
# write_fasta(CARs_seqs, "/media/anupkumar/Backup Plus/project/ML_project/data_master/amyloid/all_json_waltz_output_disprot/first.fasta")


print("Done")
