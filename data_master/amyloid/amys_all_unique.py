#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:11:30 2023

@author: anupkumar
"""
from amp_gan.utils import ProcessSeqs, write_fasta
import csv

#%%variables

path_cur = "/media/anupkumar/Backup Plus/project/ML_project/data_master/amyloid/"

files =[path_cur+f for f in ["all_json_waltz_output_disprot/CARs_all.fasta","full_CPAD2_o/CPAD_all.fasta", "waltzdb/waltz_all.fasta" ]]

#%%   
amys = ProcessSeqs(*files).get_seqs()
amys_removed_amide = ProcessSeqs().remove_amideCter(amys)
amys_filtered = ProcessSeqs().filter_seqs(amys_removed_amide)
amys_unique = ProcessSeqs().get_unique_seqs(amys_filtered)
write_fasta(amys_unique,path_cur + "amys_all_unique.fasta" )

