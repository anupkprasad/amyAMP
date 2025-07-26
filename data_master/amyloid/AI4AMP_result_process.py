#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:57:33 2023
@author: anupkumar
"""
from amp_gan.utils import ProcessSeqs, write_fasta, processAI4AMP_results

path  = "/media/anupkumar/Backup Plus/project/ML_project/data_master/amyloid/"
file_result = path + "amys_uniqueAI4AMP.out"
seqs_file = path + "amys_all_unique.fasta"
  
conditions = {'Score':0.4} 
results = processAI4AMP_results(file_result, seqs_file, conditions)
seqs = {}
for key, value in results.items():
    seqs[key] = value[0]


write_fasta(seqs,path + "amys_uniqueAI4AMP.fasta" )
## read again to process  
processed_amys_AIAMP4 = ProcessSeqs(*[path + "amys_uniqueAI4AMP.fasta"]).get_unique_seqs()
write_fasta(processed_amys_AIAMP4,path + "amys_uniqueAI4AMP_processedtotrain.fasta" )

