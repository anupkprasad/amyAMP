#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
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
    

#from scripts import analysis
## for analysis_generated_seqs
fasta_random = [path_data+ "random_pep_uni.fasta"]
fasta_uperin = [path_data+ "uperin.fasta"]
fasta_generated = [path_result+"seqs_generated" + str(batch_generate) +".fasta"]
fasta_all = fasta_generated + fasta_AMPs + fasta_AMYs + fasta_random + fasta_uperin
conditions_AMPs = {"AMP Score": 0.95,"Hemolysis /Y/N": "No","ACP Score": 0.90, "ACVP Score": 0.90, "AFP Score": 0.90,"AVP Score": 0.90}
conditions_AMPs = {"AMP Score": 0.95,"Hemolysis /Y/N": "No"}
score_AMYs = 0.95

import sys
sys.path.append("/home/anup/workspace/amyAMP")
from scripts import analysis_generated_seqs
ai4amp_waltz = analysis_generated_seqs.ai4amp_waltz
