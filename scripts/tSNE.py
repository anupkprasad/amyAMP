#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It imports models to do basic stats for model evaluations
    generate peptide sequences
    wirte shuffeled seqs of AMY and AMP datasets
    plot training loss
    plot violin_plots of physiochemical properties
    plot frequency
    plot tSNE
"""
import sys
sys.path.append("/home/anup/workspace/amyAMP")
import torch
from __main__ import *  ### import namespace of caller script
from scripts import util
from models_nn import model as model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filter_max = 100  ## filters in Conv and linear (encoded dimention) NN
run_num = 10  ## number of training run
batch_generate = 1000 ###number of peptides want to generate

path_data = "/home/anup/workspace/amyAMP/data_master/"
fasta_AMPs = [path_data+"amps/dbaasp/dbaasp_APR_processed.fasta"]
fasta_AMYs =[path_data+"amyloid/amys_uniqueAI4AMP_processedtotrain.fasta"]

path_root = os.path.dirname(os.path.abspath(__file__))
path_model = path_root + "/model_saved_231206/"   ### model folder which is gooing to analyse

path_result = path_model + "/results_" + str(run_num)+  "/"  ### analysis output dir
if not os.path.exists(path_result):
    os.makedirs(path_result)


path_data = path_data
fasta_AMPs = fasta_AMPs
fasta_AMYs =fasta_AMYs

path_model = path_model
path_root = path_root

path_result = path_result
if not os.path.exists(path_result):
    os.makedirs(path_result)

filter_max = filter_max  ## maximum filters applied in NN
run_num = run_num  ## number of training run
batch_generate = batch_generate ###number of peptides want to generate


table = util.get_conversion_table(path_data+"physical_chemical_6.txt")

### write AMPs and AMYs shuffled seqs
seq_object = util.get_shuffled_sample(*fasta_AMPs,num_seqs = batch_generate)
util.write_fasta(seq_object, path_result + "seqs_realAMPs"+str(batch_generate)+".fasta")

seq_object = util.get_shuffled_sample(*fasta_AMYs,num_seqs = batch_generate)
util.write_fasta(seq_object, path_result + "seqs_realAMYs"+ str(batch_generate) + ".fasta")


#####        analsis and plot
###violin plot of phy_chem properties
l_fasta = [path_result+"seqs_generated"+str(batch_generate)+".fasta",
           path_result+"seqs_realAMPs"+str(batch_generate)+".fasta",
           path_result+"seqs_realAMYs"+str(batch_generate)+".fasta",
           path_data+ "random_pep_uni.fasta"]


prop = ["charge", "hydrophobicity", "hydrophobic_moment", "boman", "instability_index", "isoelectric_point"]
for p in prop:
    data = []
    for f in l_fasta:
        phychem_prop = util.GetPhychem(f)
        phychem_prop1 = phychem_prop.get_phychem_prop(l_prop = prop)
        data.append(phychem_prop1[p])
    util.plot_violin(data, path_result+p + ".png", title=p.capitalize())

############    plot loss and identity
f = path_model + 'collectedseqs_loss_epochinfo_r'+ str(run_num)+'.json'
# returns JSON object as a dictionary
loss_n_identity = torch.load(f)
loss_all = loss_n_identity["loss_all"]
util.plot_loss(  loss_all[0], loss_all[1],loss_all[2], loss_all[3], file = path_result+"loss.png")

### frequncy 
from scripts import util
### shuffled AMPs with amidation tag for freq analysis
seq_object = util.get_shuffled_sample(*fasta_AMPs,num_seqs = batch_generate, amidion_tag= True)
util.write_fasta(seq_object, path_result + "seqs_realAMPs_amidation_tag"+str(batch_generate)+".fasta")
l_fasta[1] = path_result+"seqs_realAMPs_amidation_tag"+str(batch_generate)+".fasta" 
label = ["BiGAN-peps", "AMPs", "AMYs", "Random-peps"]
fr_list = []
for i, f in enumerate(l_fasta):
    if i == 0:
        fr = util.GetPhychem(f, amidated=True).aa_freq()
        test1 = util.GetPhychem(f, amidated=True).seqs
        fr_list.append(list(fr.keys()))
        fr_list.append(list(fr.values()))
    else:
        fr = util.GetPhychem(f, amidated=True).aa_freq()
        fr_list.append(list(fr.values()))



import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(fr_list[1:])
df.columns = fr_list[0]
df = df.transpose()
fig, ax = plt.subplots(2,2, figsize= (7,5), constrained_layout = True)
x = -1
for r in range(2):
    for c in range(2):
        x +=1 
        df[x].plot(kind='bar', ax = ax[r,c], rot=0, xlabel='Amino acid', ylabel='Frequency',
             width=0.6, color=['black'], alpha = 0.7)
        ax[r,c].legend(label[x:], fontsize = 8)
        ax[r,c].set_ylim([0,0.14])
fig.savefig(path_result + "aa_freq_multiplots.png", bbox_inches='tight', dpi = 600)



#### tSNE plot
tsne_transformed_data =  util.tSNE2D_df(l_fasta, table, path_result+"tSNE2D_df.png")
util.tSNE3DI(l_fasta, table, path_result+"tSNE3DI.html")

