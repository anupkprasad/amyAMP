#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:11:30 2023

@author: anupkumar
"""
# Python program to read
# json file

import json

path =  "/media/anupkumar/Backup Plus/project/ML_project/data_master/amyloid/waltzdb/"
# Opening JSON file
f = open(path + 'waltzdb_export.json')

datasets = json.load(f)
# Closing file
f.close()

#######create dict of disprot_id and its CARs
seqs = []

for i, d in enumerate(datasets):
    if d["Classification"] == "amyloid":
        seq = d["Sequence"]
        seqs.append(seq)
seqs = set(seqs)

waltz_datasets = {}
for i, s in enumerate(seqs):
    sid = "waltz_" + str(i)
    waltz_datasets[sid] = s


### use write_fasta def of utils.py
# write_fasta(CARs_seqs, path

print("Done")
