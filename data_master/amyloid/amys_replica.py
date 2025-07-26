#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:11:30 2023

@author: anupkumar
"""
from amp_gan.utils import ProcessSeqs, write_fasta
import csv

#%%variables

path_cur = "/media/anupkumar/Backup Plus/project/ML_project/data_master/amyloid/amys_uniqueAI4AMP_processedtotrain _for_replica.fasta"


#%%   
amys = ProcessSeqs(path_cur).get_seqs()

write_fasta(amys,path_cur + "amys_uniqueAI4AMP_processedtotrain_5replica.fasta" )

