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

"""
Class
"""

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



class GetPhychem:

    def __init__(self, fasta_files_or_seqs = None, amidated = False):
        self.fasta_files_or_seqs = fasta_files_or_seqs
        self.amidated = amidated
        if amidated == True:
            self.seqs = ProcessSeqs(fasta_files_or_seqs).get_seqs_amideCter_maxlen()
        else:
            self.seqs = ProcessSeqs(fasta_files_or_seqs).get_usualSeqs()

    def get_phychem_prop(self, l_prop:list=None):
        """
        l_prop : list of properties that are in peptides module (installed on system)
        Returns : dic(fasta:dict(l_prop:list))
        """
        seqs = self.seqs
        prop = {}
        for p in l_prop:
            p_value_of_peps = []
            for seq in seqs.values():
                peptide = ptd.Peptide(seq) ## Peptide is a class
                p_value = getattr(peptide, p)()
                p_value_of_peps.append(p_value)
            prop[p] = p_value_of_peps
        return prop

    def aa_freq(self):
        seqs = self.seqs
        amidated = self.amidated
        l_seqs = []
        num_elements = 0
        for key, values in seqs.items():
            l_seqs.append(values)
            num_elements += len(values)        
        seqs_array = np.array(l_seqs)
        aa = ["A","C", "D", "E","F","G", "H","I","K", "L","M","N",
              "P","Q","R", "S","T","V", "W","Y", "J", "Z"]
        if amidated == False:
            aa = aa[:-2]
        aa_count = {}
        for a in aa:
            c = np.char.count(seqs_array, a)
            a_freq = np.sum(c)/num_elements
            aa_count[a] = a_freq
        return aa_count





"""
Def
"""
def read_fasta(*fasta_files) -> dict:
    seqs = ProcessSeqs(*fasta_files).get_seqs()
    return seqs   


### Table of physiochemical properties, need for PC6 conversion 
def get_conversion_table(path = "/media/anup/BackupPlus/project/ML_project/data_master/physical_chemical_6.txt", norm=True): 
    table = pd.read_csv(path, sep=" ", index_col=0)
    index = list(table.index)
    if norm:
        scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(table)
        table = {}
        for index, aa in enumerate(index):
            table[aa] = np.array(scaled[index])[0:6]
    table["X"] = np.array([0] * 6)
    
    
    return table

def write_fasta(seqs, path, mode = "w"):
    with open(path, mode) as output:
        for name, seq in seqs.items():
            output.write(">{}\n{}\n".format(name, seq))

def append_fasta(seqs, path, mode = "a"):
    with open(path, mode) as output:
        for name, seq in seqs.items():
            output.write(">{}\n{}\n".format(name, seq))

"""
Protein Encode-decode
"""

def padding_seqs(seqs, length=None, pad_value=None):
    length = length or 30
    pad_value = pad_value or "X"
    data = {}
    for key, seq in seqs.items():
        if len(seq) < length:
            data[key] = seq + pad_value * (length - len(seq))
        else:
            raise Exception("Length exceeds {}".format(length))
    return data


def unpadding_seqs(seqs: list, pad_value="X"):

    data = {}
    for key, seq in enumerate(seqs):
        cur_seq = [aa for aa in seq if aa != "X"] ## unpadding
        data[key] = "".join(cur_seq)
    return data

def encode(fasta, table):
    encoded_seqs = {}
    for key, seq in fasta.items():
        encoded_seqs[key] = [table[aa] for aa in list(seq)]
    return encoded_seqs


def get_encoded_seqs(seqs, table):
    seqs = padding_seqs(seqs)
    encoded_seqs = encode(seqs, table)
    encoded_seqs = np.array(list(encoded_seqs.values())).astype(float)
    encoded_seqs = np.expand_dims(encoded_seqs, axis=1)
    return encoded_seqs


def decode(seqs, table):
    decode_seqs = []
    for seq in seqs:
        seq = seq.reshape(30, -1)
        decode_seq = ""
        for index in range(seq.shape[0]):
            generated_vector = seq[index]
            similarity = {}
            for key, aa_vector in table.items():
                aa_vector = torch.FloatTensor(aa_vector)
                score = cosine_similarity(generated_vector, aa_vector, dim=-1, eps=1e-8)
                similarity[key] = score
            key_max = max(similarity.keys(), key=(lambda k: similarity[k]))
            decode_seq += key_max
        decode_seqs.append(decode_seq)
    return decode_seqs


def decode_elucidian(seqs, table):
    from math import dist
    decode_seqs = []
    for seq in seqs:
        seq = seq.reshape(30, -1)
        decode_seq = ""
        for index in range(seq.shape[0]):
            generated_vector = seq[index]
            similarity = {}
            for key, aa_vector in table.items():
                aa_vector = torch.FloatTensor(aa_vector)
                score = dist(generated_vector, aa_vector)
                similarity[key] = score
            key_min = min(similarity.keys(), key=(lambda k: similarity[k]))
            decode_seq += key_min
        decode_seqs.append(decode_seq)
    return decode_seqs


def generate_seqs(net, table, noise, *epoch):
    generated = net(noise)
    decoded_seqs = decode_elucidian(generated, table)
    seqs = {}
    for i, decoded_seq in enumerate(decoded_seqs):
        seq = [aa for aa in decoded_seq if aa != "X"] ## unpadding
        seqs["seq_epochs_{epoch}_num_{num}".format(epoch = epoch, num = i + 1)] = "".join(seq)
    return seqs





"""
Analysis
"""
def processAI4AMP_results (result_file, conditions:dict):
    df = pd.read_csv(result_file)
    data_met = df
    for key, values in conditions.items():
        if key == "Hemolysis /Y/N":
            data_met = data_met[(data_met[key] == conditions[key])]
        else:
            data_met = data_met[(data_met[key] >= conditions[key])]
    data_met = data_met.loc[:, ["Peptide"] + list(conditions.keys())] ## column selection of pept id and keys
    if 'AMP Score' in conditions.keys():
        data_met= data_met.sort_values(by=['AMP Score'], ascending=False)
    id_score = np.array(data_met)
    
    selected_seqs = {}
    for i in range(len(id_score)):
        key = id_score[i][0]
        score = list(id_score[i][1:])
        selected_seqs[key] = score
        
    return selected_seqs

def processAI4AMP_results_old (result_file, seqs_file, conditions:dict):
    df = pd.read_csv(result_file)
    realseqs = ProcessSeqs(seqs_file).get_seqs()
    df.columns = df.columns.str.replace(' ', '') 
    if len(list(conditions.values())) ==2:
        data_met = df[(df['Score'] >= conditions['Score']) & (df['Hemolysis'] == conditions['Hemolysis'])]
        data_met = data_met.loc[:,['Peptide','Score']]
        data_met= data_met.sort_values(by=['Score'], ascending=False)
        id_score = np.array(data_met)
    elif len(list(conditions.values())) ==1:
        data_met = df[(df['Score'] >= conditions['Score'])]
        data_met = data_met.loc[:,['Peptide','Score']]
        data_met= data_met.sort_values(by=['Score'], ascending=False)
        id_score = np.array(data_met)
    
    selected_seqs = {}
    for i in range(len(id_score)):
        key = id_score[i][0]
        score = id_score[i][1]
        try:
            selected_seqs[key] = [realseqs[key], score]
        except: ### if key contains extra "_"
            key_cur = key.split("_")
            key_cur1 = key_cur[0]
            try:
                selected_seqs[key_cur1] = [realseqs[key_cur1],score]
            except: ### if key is needed just one "_"
                key_cur1 = key_cur[0] +"_" + key_cur[1]
                selected_seqs[key_cur1] = [realseqs[key_cur1],score]
        else:
            selected_seqs[key] = [realseqs[key],score]
        
    return selected_seqs


def processWaltz_result (result_file, seqs_file, score_cutoff = 75):
    ## find only first amyloidgenic part
    f = open(result_file)
    data = []
    d = ""
    for x in f:
        if x[0] == ">":
            data.append(d.split())
            d = ""
        d = d+x[:-1]
    f.close()
    data = data[1:]
    ### data containg APR has length > 7
    APR_data = []
    id_list = []
    for d in data:
        if len(d)>7:
            APR_data.append(d)
            id_list.append(d[0][1:])
    
    realseqs = ProcessSeqs(seqs_file).get_seqs()
    selected_seqs = {}
    
    for i in range(len(id_list)):
        key = id_list[i]
        APR_rng = APR_data[i][6]
        score = APR_data[i][8]
        if float(score.split("-")[0]) > score_cutoff:
            selected_seqs[key] = [realseqs[key], score, APR_rng]
    return selected_seqs


from Bio import pairwise2
# def get_pairwiseAlignment(seq1, seq2):
#     alignment_score = pairwise2.align.globalxx(seq1, seq2,score_only = True)
#     min_lenth = min(len(seq1), len(seq2))
#     identity = alignment_score / min_lenth
#     return identity*100

def get_alignmentScore(seq1, seq2):
    if len(seq1)==0 or len(seq2) == 0:
        return 0
    else:
        alignment_score = pairwise2.align.globalxx(seq1, seq2,score_only = True)
        avg_lenth = (len(seq1) + len(seq2))/2
        identity = alignment_score / avg_lenth
    return identity*100

def get_batchAlignment(seqs1:dict, seqs2:dict, topmatch_num = 5):
    score_list = []
    identical_seqsnum = 0
    for k1, seq1 in seqs1.items():
        if seq1 in list(seqs2.values()):
            identical_seqsnum += 1
        else:
            identity = []
            for k2, seq2 in seqs2.items():
                identity.append([k1, k2, get_alignmentScore(seq1, seq2)])
            identity =  np.array(identity)
            maxid = np.argmax(identity[:,2].astype(float), axis = 0)
            score_list.append(identity[maxid])
    score_list = np.array(score_list)
    if len(score_list) == 0:
        print("all sequences are identical in batch alignment")
        return 
    else:
        indtop5 = np.argpartition(score_list[:,2].astype(float), -topmatch_num)[-topmatch_num:]
        ltop5 = [[seqs1[score_list[i][0]], seqs2[score_list[i][1]], float(score_list[i][2])] for i in indtop5]
        align_result = {"score":score_list, "identicalSeqs_num":identical_seqsnum,"topmatched": ltop5 }
    return align_result


def plot_hist(data, save_file, bins = 70):
    fig, axs = plt.subplots(1,1, figsize=(3.5, 2), constrained_layout =True, dpi = 300)
    label = ["Uperin", "AMYs", "Random-peptides"]
    data_l = [data]  ## list of list
    for i, data_l in enumerate(data_l):
        mu = np.mean(data_l)
        var = np.var(data_l)
        axs.set_ylabel("Peptide number")
        axs.set_xlabel("Identity (%)")
        axs.hist(data_l, color = 'blue', edgecolor = 'red', bins = bins)
        axs.text(0.57,0.7, label[i] +"\n" +  r"$\bar{x}$ = "+str(int(mu)) +"\n" +  r"$\sigma^2$ = "+ str(int(var)), transform=axs.transAxes, fontsize = 8)
    plt.savefig(save_file)


def plot_multihist(data_list, save_file, bins = 70):
    fig, axs = plt.subplots(3,1, figsize=(3.5, 6), constrained_layout =True, dpi = 300)
    mini, maxi = min([min(l) for l in data_list]), max([max(l) for l in data_list])
    label = ["AMPs", "AMYs", "Random-peptides"]
    for i, data_l in enumerate(data_list):
        mu = np.mean(data_l)
        var = np.var(data_l)
        axs[i].set_ylabel("Peptide number", fontsize = 8)
        axs[i].set_xlabel("Identity (%)", fontsize = 8)
        axs[i].hist(data_l, color = 'blue', edgecolor = 'red', bins = bins)
        axs[i].set_xlim(mini, maxi)
        axs[i].text(0.57,0.7, label[i] +"\n" +  r"$\bar{x}$ = "+str(int(mu)) +"\n" +  r"$\sigma^2$ = "+ str(int(var)), transform=axs[i].transAxes, fontsize = 8)
    plt.savefig(save_file)


def plot_identity(values, step, path):
    labels = [i * step for i in range(1, 1 + len(values))]
    fig, axs = plt.subplots(figsize=(15, 5))
    axs.set_ylabel("Generated sequence identity to natural ones")
    axs.set_xlabel("Epoch")
    axs.boxplot(np.transpose(values), labels=labels)
    plt.savefig(path)


def plot_loss(EG_loss, D_loss, E_loss = None, G_loss = None, file = str):
    plt.figure(figsize=(5, 3.5), constrained_layout = True)
    plt.title("Loss in training")
    x = list(range(len(EG_loss)))
    plt.plot([10*i for i in x], EG_loss, label="Encoder_Generator")
    plt.plot([10*i for i in x], D_loss, label="Discriminator")
    if E_loss != None:
        plt.plot([10*i for i in x], E_loss, label="Encoder")
    if G_loss != None:
        plt.plot([10*i for i in x], G_loss, label="Generator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(fontsize = 8)
    plt.savefig(file, dpi = 600)

### try with high to hydrophobic
def plot_fr(data_d,save_name):
    plt.figure(figsize=(5, 3.5), constrained_layout = True)
    names = list(data_d.keys())
    values = list(data_d.values())
    plt.bar(range(len(data_d)), values, tick_label=names)
    plt.ylim(0, 0.3)
    plt.ylabel("Frequency")
    plt.xlabel("Amino acids")
    plt.savefig(save_name)

def tSNE2D_df(fastafiles, table, filename):
    
    fig = plt.figure(figsize =  (3.5,3.5), dpi = 600, constrained_layout = True)
    ax = plt.axes()
    phio_prop = ["H1", "V", "P1", "Pl", "PKa", "NCl"]
    phio_prop = ["BiGAN-peps", "AMPs", "AMYs", "Random-peps", "PKa", "NCl"]
    c= ['r', 'b', 'c', 'm', 'y', 'k', 'lightblue', 'orange', 'gray']
    
    i = 0
    for fastafile in fastafiles:
        print(fastafile)
        i = i+1
        fasta_seqs = read_fasta(fastafile)
        
        selected_seqs = {}
        for id_, seq in fasta_seqs.items():
            if len(seq) < 30:
                selected_seqs[id_] = seq
        print("\n\n\n " + str(len(selected_seqs)) + "\n\n\n ")
        encoded = get_encoded_seqs(selected_seqs, table)
        encoded = encoded.reshape(-1,30,6)
        encoded6f = encoded.reshape(-1,6)
        tsne = TSNE(n_components=2, random_state=0, learning_rate=70, max_iter=2000, n_iter_without_progress=400, verbose=1)
        #tsne = TSNE(n_components=2)
        encoded3f = tsne.fit_transform(encoded6f)
        print("\n\n\n " + str(len(encoded3f)) + "\n\n\n ")
        ax.scatter(encoded3f[:,0], encoded3f[:,1], c = c[i-1], marker = 'o', label=phio_prop[i-1])
    ax.legend(fontsize = 7)
    plt.savefig(filename)
    return encoded3f


import plotly.express as px
def tSNE3DI(fastafiles, table, filename, data_size = 250):
    fig = plt.figure(figsize =  (6,5), dpi = 600, constrained_layout = True)
    ax = plt.axes(projection ='3d')
    pep_type = ["BiGAN-peps", "AMPs", "AMYs", "Random-peps", "PKa", "NCl"]
    c= ['r', 'b', 'g','c', 'y', 'k', 'lightblue', 'orange', 'gray']
    alpha = [0.3,0.3,0.8,0.1]
    marker = ["o", "s", "*", "D"]
    i = 0
    all_data = np.empty([1,4])
    for fastafile in fastafiles:
        print(fastafile)
        i = i+1
        seqs = read_fasta(fastafile)
        encoded = get_encoded_seqs(seqs, table)
        if len(encoded) > data_size:
            encoded = encoded[:data_size,:,:,:]
        
        encoded = encoded.reshape(-1,30,6)
        encoded6f = encoded.reshape(-1,6)
        tsne = TSNE(n_components=3, random_state=0, learning_rate=70, max_iter=2000, n_iter_without_progress=400, verbose=1)
        encodedf = tsne.fit_transform(encoded6f)
        print(len(encodedf))
        lbl = []
        for j in range(len(encodedf)):
            lbl.append([pep_type[i-1]])
        lbl = np.array(lbl)
        encodedf = np.append(encodedf,lbl, axis = 1)
        all_data = np.append(all_data,encodedf, axis = 0)
    all_data = all_data[1:]
    all_data= pd.DataFrame(all_data)
    all_data.columns = ["f1","f2","f3","l"]
    for col in ["f1","f2","f3"]:
        all_data[col] = all_data[col].astype(float)
    fig = px.scatter_3d(all_data, x='f1', y='f2', z='f3',
                    color='l')
    fig.write_html(filename)
    return all_data, lbl



def plot_violin(data:list, save_file, title:None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize = (5,3), constrained_layout = True)
    ax = sns.violinplot(data = data)
    #sns.swarmplot(data = data, color="k", alpha=0.8)
    ax.set_xticklabels(['BiGAN-pep','AMPs','AMYs','Random-pep'])
    ax.set_title(title, fontweight= "bold")
    ax.grid(axis='y', alpha=0.8, linestyle=':')
    #ax.set_ylabel("Gapminder Life Expectancy")
    #ax.set_xlabel("Nations")
    fig.savefig(save_file, dpi = 600)

def plot_multiviolin(data:list, save_file, title:None):
    import matplotlib.pyplot as plt
    row, column = int(len(data)/3), 3
    fig, ax = plt.subplots(row, column, figsize = (7,row*2), constrained_layout = True)
    x = -1
    for r in range(row):
        for c in range(column):
            x +=1
            print(title[x])
            sns.violinplot(data = data[x], ax = ax[r,c])
            #sns.swarmplot(data = data, color="k", alpha=0.8)
            ax[r,c].set_xticklabels(['BiGAN-peps','AMPs','AMYs','Random-peps'],rotation = 20, fontsize =6,  fontweight= "bold")
            ax[r,c].set_ylabel(title[x], fontweight= "bold", fontsize = 6)
            ax[r,c].grid(axis='y', alpha=0.8, linestyle=':')
            #ax.set_ylabel("Gapminder Life Expectancy")
            #ax.set_xlabel("Nations")
    fig.savefig(save_file, dpi = 600)


def get_shuffled_sample (*fasta_files, num_seqs = 128, amidion_tag = False)-> dict:
    if amidion_tag == True:
        seqs = ProcessSeqs(*fasta_files).get_seqs()
    else:
        seqs = ProcessSeqs(*fasta_files).get_usualSeqs()
    #data = custom_data(seqs)
    #table = get_conversion_table()
    #encoded = get_encoded_seqs(seqs, table)
    dataloader = torch.utils.data.DataLoader(
        list(zip(seqs.keys(), seqs.values())), batch_size=num_seqs, shuffle=True, drop_last=True)
    sample = next(iter(dataloader))
    shuffeled_seq = {}
    for key, value in zip(sample[0], sample[1]):
        shuffeled_seq[key] = value
    # sample = decode_elucidian(sample, table)
    # sample = unpadding_seqs(sample)
    return shuffeled_seq


def get_shuffled_sample_alternative (*fasta_files, num_seqs = 128)-> dict:
    seqs = ProcessSeqs(*fasta_files).get_usualSeqs()
    table = get_conversion_table()
    encoded = get_encoded_seqs(seqs, table)
    dataset = torch.utils.data.TensorDataset(torch.Tensor(encoded))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=num_seqs, shuffle=True, drop_last=True)
    sample = iter(dataloader)
    sample = next(sample)[0]
    sample = decode_elucidian(sample, table)
    sample = unpadding_seqs(sample)
    return sample

"""s
Garbage
"""

# def get_batchAlignment(seqs1, seqs2):
#     seqs1 = list(seqs1.values())  ### enquiring seqs
#     seqs2 = list(seqs2.values())    ### referenced seqs
#     score_list = []
#     identical_seqsnum = 0
#     for seq1 in seqs1:
#         identity_list = []
#         if seq1 in seqs2:
#             identical_seqsnum += 1
#         else:
#             for seq2 in seqs2:
#                 identity_list.append(pairwise2.align.globalxx(seq1, seq2,score_only = True))
#             score_list.append(max(identity_list))
#     return score_list, identical_seqsnum








# def get_phychem_propfdd(self, l_prop:list):
#     """
#     l_prop : list of properties that are in peptides module (installed on system)
#     Returns : dic(fasta:dict(l_prop:list))
#     """
#     l_fasta = [f for f in self.fasta_files]
#     phychem = {}
#     for file in l_fasta:
#         prop = {}
#         for p in l_prop:
#             p_value_of_peps = []
#             for record in SeqIO.parse(file, "fasta"): ### iterrator in fasta peptides (record.id, record.seq)
#                 peptide = str(record.seq) ## peptide seq in str ""
#                 peptide = ptd.Peptide(peptide) ## Peptide is a class
#                 p_value_of_peps.append(getattr(peptide, p)())
#             prop[p] = p_value_of_peps
#         phychem[file] = prop
#     return phychem

# def read_fasta(*fasta_files) -> dict:
#     fasta = {}
#     for file in fasta_files:
#         for record in SeqIO.parse(file, "fasta"):
#             if record.id in list(fasta.keys()):
#                 fasta[str(record.id) + str(1)] = str(record.seq)
#             else:
#                 fasta[str(record.id)] = str(record.seq)
#     return fasta   


def tSNE2D_need_to_work(real_seqs, gen_seqs, filename):
    real_seqs = real_seqs.cpu().detach().numpy()
    gen_seqs = gen_seqs.cpu().detach().numpy()
    real_seqs = real_seqs.reshape(128,30,6)
    gen_seqs = gen_seqs.reshape(128,30,6)
    fig = plt.figure(figsize =  (5,5), dpi = 600, constrained_layout = True)
    ax = plt.axes()
    phio_prop = ["H1", "V", "P1", "Pl", "PKa", "NCl"]
    c= ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'lightblue', 'orange', 'gray']
    for i in range(6):
        real = real_seqs[:,:,i]
        gen = gen_seqs[:,:,i]
        ax.scatter(real[:,0], real[:,1], c = c[i], marker = 'o', label=phio_prop[i])
        ax.legend()
    plt.savefig(filename)

def tSNE(fastafiles, table, filename, data_size = 128):
    
    fig = plt.figure(figsize =  (6,5), dpi = 600, constrained_layout = True)
    ax = plt.axes(projection ='3d')
    phio_prop = ["H1", "V", "P1", "Pl", "PKa", "NCl"]
    phio_prop = ["BiGAN-pep", "AMPs", "Amys", "random_pep", "Pl", "PKa", "NCl"]
    c= ['r', 'b', 'g','c', 'y', 'k', 'lightblue', 'orange', 'gray']
    alpha = [0.3,0.3,0.8,0.1]
    marker = ["o", "s", "*", "D"]
    i = 0
    for fastafile in fastafiles:
        print(fastafile)
        i = i+1
        seqs = read_fasta(fastafile)
        encoded = get_encoded_seqs(seqs, table)
        if len(encoded) > data_size:
            encoded = encoded[:data_size,:,:,:]
        
        encoded = encoded.reshape(-1,30,6)
        encoded6f = encoded.reshape(-1,6)
        tsne = TSNE(n_components=3, random_state=0, learning_rate=70, max_iter=2000, n_iter_without_progress=400, verbose=1)
        encoded3f = tsne.fit_transform(encoded6f)
        ax.scatter(encoded3f[:,0], encoded3f[:,1], encoded3f[:,2], c = c[i-1], marker = marker[i-1], label=phio_prop[i-1], alpha = alpha[i-1])
    ax.legend()
    plt.savefig(filename)
    
   


def tSNE2D(fastafiles, table, filename, data_size = 128):
    fig = plt.figure(figsize =  (3.5,2.5), dpi = 600, constrained_layout = True)
    ax = plt.axes()
    phio_prop = ["H1", "V", "P1", "Pl", "PKa", "NCl"]
    phio_prop = ["BiGAN-pep", "AMPs", "Amys", "random_pep", "Pl", "PKa", "NCl"]
    c= ['r', 'b', 'c', 'm', 'y', 'k', 'lightblue', 'orange', 'gray']
    alpha = [0.4,0.4,0.8,0.2]
    marker = ["o", "s", "*", "D"]
    i = 0
    for fastafile in fastafiles:
        print(fastafile)
        i = i+1
        seqs = read_fasta(fastafile)
        encoded = get_encoded_seqs(seqs, table)
        if len(encoded) > data_size:
            encoded = encoded[:data_size,:,:,:]
        
        encoded = encoded.reshape(-1,30,6)
        encoded6f = encoded.reshape(-1,6)
        tsne = TSNE(n_components=2, random_state=0, learning_rate=70, max_iter=2000, n_iter_without_progress=400, verbose=1)
        encoded2f = tsne.fit_transform(encoded6f)
        ax.scatter(encoded2f[:,0], encoded2f[:,1], c = c[i-1], marker = marker[i-1], label=phio_prop[i-1], alpha = alpha[i-1])
    ax.legend()
    plt.savefig(filename)

def aa_freq(file_fasta):
    seqs_d = read_fasta(file_fasta)
    seqs_l = []
    num_elements = 0
    for key, values in seqs_d.items():
        seqs_l.append(values)
        num_elements += len(values) 
        
    seqs_array = np.array(seqs_l)
    aa = ["A","C", "D", "E","F","G", "H","I","K", "L","M","N",
          "P","Q","R", "S","T","V", "W","Y", "J", "Z"]
    aa_count = {}
    for a in aa:
        c = np.char.count(seqs_array, a)
        a_freq = np.sum(c)/num_elements
        aa_count[a] = a_freq
    return aa_count
