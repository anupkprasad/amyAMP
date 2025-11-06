from datetime import datetime
import torch
import os
import sys
if os.path.dirname("~/workspace/amyAMP/") not in sys.path:
    sys.path.append(os.path.dirname("~/workspace/amyAMP/"))   
from scripts import util
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# varriables for training
istrain = False
batch_size =128
epoch = 7000
run_num = 2
filter_max = 100

path_root = os.path.dirname("~/workspace/amyAMP/")

path_model = path_root + "/model_saved/" ## models are saved in this directory which will also be used for pep generation 
if not os.path.exists(path_model):
    os.makedirs(path_model)
path_result =  f"{path_root}results" ### analysis output dir
if not os.path.exists(path_result):
    os.makedirs(path_result)
    
## training data path
fasta_AMPs = [f"{path_root}/data_master/amps/dbaasp/dbaasp_APR_processed.fasta"]   
fasta_AMYs =[f"{path_root}/data_master/amyloid/amys_uniqueAI4AMP_processedtotrain.fasta"]

#other varriables for analysis
## for analysis.py
batch_generate =1000
## for analysis_generated_seqs.py
fasta_random = [f"{path_root}/data_master/random_pep_uni.fasta"]
fasta_uperin = [f"{path_root}/data_master/uperin.fasta"]
fasta_generated = [path_result+"seqs_generated" + str(batch_generate) +".fasta"]
## fasta_generated should be first in list in following list
fasta_all = fasta_generated + fasta_AMPs + fasta_AMYs + fasta_random + fasta_uperin
conditions_AMPs = {"AMP Score": 0.95}
score_AMYs = 0.95


path_PC6 = f"{path_root}/data_master/physical_chemical_6.txt"
table = util.get_conversion_table(path_PC6)
seqs ={}
for f in fasta_AMPs + fasta_AMYs:
    cur_seq = util.read_fasta(f)
    print("seqs_num= {}".format(len(cur_seq)))
    seqs.update(cur_seq) ### merging two dict
print("Total seqs = {}".format(len(seqs)))

#### PC6 encoding
encoded = util.get_encoded_seqs(seqs, table)
dataset = torch.utils.data.TensorDataset(torch.Tensor(encoded))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#Training
"""
# Training
"""
from models_nn import train
if istrain == True:
    start_time = datetime.now().strftime("%H:%M:%S")
            
    G, loss_all, collected_seqs = train.training(path_model, epoch = epoch, filter_max= filter_max, train_data=dataloader, run_num = run_num)
    
    finish_time = datetime.now().strftime("%H:%M:%S")
    print("Current Time = {} \n Finish Time = {}".format(start_time, finish_time))

#model summary and get trained models for peptide generation
from models_nn import train, load_model
train.models_summary(filter_max, path_model)
E,G,D,optimizer_EG,optimizer_D,dataloader, loss_all, epoch_i = load_model(path_model, E, G, D, optimizer_EG, optimizer_D, run_num)
z = torch.randn(batch_size, filter_max, 1, 1)
z = z.to(device)
Gz = G(z)

#analysis

from scripts import analysis_generated_seqs





