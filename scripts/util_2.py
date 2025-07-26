#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 10:12:53 2024
@author: anupkumar
This script contains methods only for data anlysis and plots of 
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.nn.functional import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from Bio import SeqIO
import peptides as ptd
import seaborn as sns
from sklearn.manifold import TSNE



#%% analsis and plot
print('violin plots of phy_chem properties')

l_fasta = [path_result+"seqs_generated"+str(batch_generate)+".fasta",
           path_result+"seqs_realAMPs"+str(batch_generate)+".fasta",
           path_result+"seqs_realAMYs"+str(batch_generate)+".fasta",
           path_data+ "random_pep_uni.fasta"]

prop = ["charge", "hydrophobicity", "hydrophobic_moment", "boman", "instability_index", "isoelectric_point"]
data = []
for p in prop:
    data_cur = []
    for f in l_fasta:
        phychem_prop = GetPhychem(f)
        phychem_prop1 = phychem_prop.get_phychem_prop(l_prop = prop)
        data_cur.append(phychem_prop1[p])
    data.append(data_cur)

#plot_multiviolin(data, path_result+p + ".png", title=p.capitalize())
plot_multiviolin(data, path_result+"multiviolin.png", title=prop)





