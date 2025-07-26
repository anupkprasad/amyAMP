#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:11:30 2023

@author: anupkumar
"""
# Python program to read
# json file

import json
import re
from Bio import SeqIO

class ProcessSeqs:
   
    def __init__(self, fasta_files_or_seqs_dict = None):        
        self.fasta_files_or_seqs = fasta_files_or_seqs_dict
        self.us_aa ={"A","C", "D", "E","F","G", "H","I","K","L","M","N",
                     "P","Q","R", "S","T","V","W","Y"}
        
    def get_seqs(self) -> dict:
        """Return all seqs: read from list of fasta files or single fasta path or dict od seqs
        """
        fasta_files_or_seqs = self.fasta_files_or_seqs
        seqs = {}
        if type(fasta_files_or_seqs)!= dict: ## for fasta_files
            if type(fasta_files_or_seqs) == str: ## if single file put in list
                fasta_files_or_seqs = [fasta_files_or_seqs]
            for file in fasta_files_or_seqs:
                print(file)
                for record in SeqIO.parse(file, "fasta"):
                    if record.id in list(seqs.keys()):
                        for d in range(10):
                            key = str("dup_")+str(d)+str(record.id)
                            if key not in list(seqs.keys()):
                                break                           
                        seqs[key] = str(record.seq)
                    else:
                        seqs[str(record.id)] = str(record.seq)
        else:  ## for seqs_dict
            seqs = fasta_files_or_seqs
        return seqs
    
    def remove_dupSeqs(self, seqs = None):
        seqs = seqs or self.get_seqs()
        rev_dict = {} 
        for key, value in seqs.items():
            rev_dict[value] = key
        unique_seqs = {} 
        for key, value in rev_dict.items():
            unique_seqs[value] = key
        
        return unique_seqs
        
    def get_seqs_amideCter_maxlen(self, seqs = None, max_length = 30, amidion_tag = True):
        seqs = seqs or self.remove_dupSeqs()
        selected_seqs = {}
        for id_, seq in seqs.items():
            if len(seq) < max_length:
                if amidion_tag == True:
                    if seq[-1] in ["J", "Z"]:  ### J == amidated, Z == non-amidated
                        selected_seqs[id_] = seq
                    else:
                        selected_seqs[id_] = seq+"Z"
                else:
                    if seq[-1] in ["J", "Z"]:
                        selected_seqs[id_] = seq[:-1]
                    else:
                        selected_seqs[id_] = seq
        return selected_seqs
    
    def remove_amideCter(self, seqs = None, remove_amide=True):
        seqs = self.remove_dupSeqs() or seqs
        selected_seqs = {}
        for id_, seq in seqs.items():
            if remove_amide == True:
                if seq[-1] in ["J", "Z"]:
                    selected_seqs[id_] = seq[:-1]
                else:
                    selected_seqs[id_] = seq
        return selected_seqs
    
    def get_usualSeqs(self, seqs = None):
        seqs = seqs or self.remove_amideCter()
        us_aa = self.us_aa
        selected_seqs = {}
        unusual = 0
        for id_, seq in seqs.items():
            if set(seq).issubset(us_aa) == True:
                    selected_seqs[id_] = seq
            else:  ## id aa not in us_aa
                unusual += 1
        print("\n unusual seqs = " + str(unusual))
        return selected_seqs
    
    def write_fasta(self, seqs = None, path = None):
        seqs = seqs or self.get_usualSeqs()
        with open(path, "w") as output:
            for name, seq in seqs.items():
                output.write(">{}\n{}\n".format(name, seq))




# Opening JSON file
f = open('/media/anup/BackupPlus/project/ML_project/peptides-complete.json')

# returns JSON object as a dictionary
datasets = json.load(f)
# Closing file
f.close()
data_list = datasets['peptides']
aa = ["A","C", "D", "E","F","G", "H","I","K", "L","M","N", "P","Q","R", "S","T","V", "W","Y"]
features = ['dbaaspId', 'sequence', "cTerminus", "physicoChemicalProperties", 'targetActivities', 'synthesisType']
extracted_data = []
for data in data_list:
    avail_data = {}
    data_keys = data.keys()

    try:
        if len(data["sequence"])>3: #len(data["physicoChemicalProperties"])>6:
            for f in features:
                if f in data_keys:
                    if f == 'targetActivities':
                        mic = []
                        species_mic = []
                        activities = data[f]
                        for A in activities:
                            if "Escherichia" in A["targetSpecies"]["name"]:
                                numbers = re.findall(r'\d+\.?\d*', A["concentration"]) #### extract only numbers from concetrtion and leaves like >," ectc
                                mic.append(max(numbers))   ## if there are two digits then just take bigger number ex: ['2.9±1.6', 'Escherichia coli ATCC 25922'] so 2.9 only
                                species_mic.append([A["concentration"], A["targetSpecies"]["name"]])
                        avail_data[f] = species_mic
                        avail_data["Mic"] = min(mic)
                            
                    elif f == 'synthesisType':
                        avail_data[f] = data[f]["name"]
                    else:
                        avail_data[f] = data[f]
    except:avail_data
    pass
    if len(avail_data) > 0:
        extracted_data.append(avail_data)
    
            

### remove unusall seq, duplicate and do amidation check
temp = []
peptides_NH2 = []
c = 0
for i, data in enumerate(extracted_data):
    pc = data["sequence"]
    pc = pc.upper()
    if pc not in temp:
        temp.append(pc)
        for a in pc:
            if a not in aa:
                break    
        try:
            amide = data["cTerminus"]["name"]
            if amide == "AMD":
                pc = pc +"J"
                c = c+1
        except:
            pc=pc + "Z"
        #peptides_NH2["seq" + str(i)] = pc
        data["sequence"] = pc
        peptides_NH2.append(data)
        
print("Total amidated peptides: " + str(c))




def read_fasta(*fasta_files) -> dict:
    seqs = ProcessSeqs(*fasta_files).get_seqs()
    return seqs

path = "/media/anup/BackupPlus/project/ML_project/data_master/amps/dbaasp/"  ## this file was used in model training

fasta_training = read_fasta(path + "dbaasp_APR_processed.fasta")
train_seqs = fasta_training.values()
### now sliced data from peptides_NH2 which was used for training

train_data_with_mic = []
for pep in peptides_NH2:
    if pep["sequence"] in train_seqs:
        if "cTerminus" in pep.keys():
            del pep["cTerminus"]
        if "physicoChemicalProperties" in pep.keys():
            del pep["physicoChemicalProperties"]
        
        seqid = [key for key, value in fasta_training.items() if value == pep["sequence"]]
        pep["seqIDtraindata"] = seqid
        train_data_with_mic.append(pep)
    



import pandas as pd
# Convert to DataFrame
df = pd.DataFrame(train_data_with_mic)

# Save to Excel
df.to_excel(path + 'dbaasp_APR_processed_mic.xlsx', index=False)

