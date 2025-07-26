#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:11:30 2023

@author: anupkumar
"""
f = open("./dbaasp_APR_processed.fasta", "r")
data = f.readlines()
f.close()

seqs_removedJZ = []
for i, seq in enumerate(data):
    if i % 2 != 0:
        s = seq[-2]
        if s in ["Z", "J"]:
            seqs_removedJZ.append(data[i-1])
            seqs_removedJZ.append(seq[:-2] + "\n")
        else:
            print(f"No J/Z record in seq{int((i-1)/2)}")
    
out_fasta = open("dbaasp_APR_processed_removedJZ.fasta", "w")

for l in seqs_removedJZ:
    out_fasta.write(l)

out_fasta.close()