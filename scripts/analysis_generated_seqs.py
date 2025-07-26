#%%import libraries
from __main__ import run_num, fasta_all,batch_generate, conditions_AMPs, score_AMYs  ### import namespace of caller script
from __main__ import path_root,path_model,path_result, path_data

from scripts.util import ProcessSeqs, GetPhychem, get_shuffled_sample, processAI4AMP_results, processWaltz_result, get_batchAlignment
from scripts.util import plot_multihist, plot_hist,plot_violin, plot_multiviolin
import pandas as pd
import peptides as ptd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%analysis antimicrobial and amyloidgenic properties

#postprocessing of generated seqs
gen_seqs = ProcessSeqs(fasta_all[0]).get_usualSeqs()
ProcessSeqs().write_fasta(gen_seqs, path_result + "seqs_generated_postprocessed.fasta")


## AI4AMP reslut
ai4amp = processAI4AMP_results(path_result + "AI4AMP_seqs_generated_postprocess.csv", conditions = conditions_AMPs )
waltz = processWaltz_result(path_result+"waltz_seqs_generated_postprocess.txt"
                            ,fasta_all[0], score_cutoff = score_AMYs )
gen_seqs1 = ProcessSeqs(fasta_all[0]).get_seqs()

ai4amp_waltz = {}
for k in ai4amp.keys():
    if k in list(waltz.keys()):
        ai4amp_waltz[k] = []
        ai4amp_waltz[k].append(gen_seqs1[k])
        for i in range(len(conditions_AMPs)):
            ai4amp_waltz[k].append(ai4amp[k][i])
        ai4amp_waltz[k].append(waltz[k][1])
        ai4amp_waltz[k].append(waltz[k][2])
        
ai4amp_waltz = pd.DataFrame(ai4amp_waltz).transpose()
col_num = len(ai4amp_waltz.columns)
ai4amp_waltz.rename(columns = {0:'Peptide', col_num-2:'Amy_score', col_num-1:'Amy_seg'}, inplace = True)
for x, k in enumerate(list(conditions_AMPs.keys())):
    ai4amp_waltz.rename(columns = {x+1: k}, inplace = True)

prop = ["charge", "hydrophobicity", "hydrophobic_moment"]
seqs = list(ai4amp_waltz["Peptide"])
for p in prop:
    p_values = []
    for seq in seqs:
        peptide = ptd.Peptide(seq) ## Peptide is a class
        p_value = getattr(peptide, p)()
        p_values.append(p_value)
    p_values = pd.DataFrame(p_values)
    p_values.rename(columns = {0: p}, inplace = True)
    p_values.index = (list(ai4amp_waltz.index))
    ai4amp_waltz = pd.concat([ai4amp_waltz, p_values], axis = 1)
    ai4amp_waltz.style.background_gradient()

ai4amp_waltz.to_excel(path_result+ "ai4amp_waltz"+ str(conditions_AMPs["AMP Score"])+"_"+ str(score_AMYs) + ".xlsx")
#%% analsis and plot
print('violin plots of phy_chem properties')

l_fasta = [path_result+"seqs_generated"+str(batch_generate)+".fasta",
           path_result+"seqs_realAMPs"+str(batch_generate)+".fasta",
           path_result+"seqs_realAMYs"+str(batch_generate)+".fasta",
           path_data+ "random_pep_uni.fasta"]


prop = ["charge", "hydrophobicity", "hydrophobic_moment", "boman", "instability_index", "isoelectric_point"]
y_lables = ["charge", "hydrophobicity", "hydrophobic moment", "boman", "instability index", "isoelectric point"]
data = []
for p in prop:
    data_cur = []
    for f in l_fasta:
        phychem_prop = GetPhychem(f)
        phychem_prop1 = phychem_prop.get_phychem_prop(l_prop = prop)
        data_cur.append(phychem_prop1[p])
    data.append(data_cur)

#plot_multiviolin(data, path_result+p + ".png", title=p.capitalize())
plot_multiviolin(data, path_result+"multiviolin.png", title=y_lables)

#%% Identity plot
seqs1=ProcessSeqs(fasta_all[0]).get_seqs()
identity_l = []
for i in fasta_all[1:]:
    identity = get_batchAlignment(seqs1=seqs1, seqs2=ProcessSeqs(i).get_seqs())
    identity_l.append(identity)
    
import numpy as np  
data_list = [identity_l[i]["score"][:,2] for i in range(4)]
data_list = np.array(data_list).astype(float)
plot_multihist(data_list[0:3], path_result+"identity_multihist.png")

### similarity with uperin
plot_hist(data_list[3], path_result+"hist_uperin.png")
### top similarity with uperins
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
top_matched = identity_l[3]['topmatched']
for t in top_matched:
    alignments= pairwise2.align.globalxx(t[0], t[1])
    print(format_alignment(*alignments[0]))
                                                                                                                                                                                                                                                        
#%% similarity of timeseries of generated seqs
epochi = list(range(99,1000,100))
epochf = list(range(1199,40000,2000))
label = ["AMPs", "AMYs", "Random", "Uperin"]
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
epoch_l = epochi+epochf
epoch_axis = list(range(1,10)) + list(range(10,410,20)) + [400]
epoch_axis = [0] + epoch_axis[0:-1:2] + [400]
collected_seqs = {}
for r in range(run_num):
    collection =torch.load(path_model + 'collectedseqs_loss_epochinfo_r'+ str(r+1)+'.json')
    collected_seqs.update(collection["collected_seqs"])

x = -1
for f in fasta_all[1:]:
    x = x +1
    identity_tl = []
    seqs2=ProcessSeqs(f).get_seqs()
    for i in epoch_l:
        seqs1 =collected_seqs[i]
        identity = get_batchAlignment(seqs1=seqs1, seqs2=seqs2)
        identity_tl.append(identity["score"][:,2])
    identity_tl = np.array(identity_tl).astype(float)
    identity_tl = identity_tl.T
    fig, axs = plt.subplots(1,1, figsize=(3.5, 2), constrained_layout =True, dpi = 300)
    # Creating plot
    axs.boxplot(identity_tl,medianprops = dict(color = "red", linewidth = 1.5), flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white'})
    axs.set_xlim([0,31])
    #axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.xaxis.set_major_locator(MultipleLocator(2))
    axs.set_xticklabels(epoch_axis)
    axs.set_xlabel(r"Epochs ($\times$100)", fontsize = 8)
    axs.set_ylabel("Identity (%)", fontsize = 8)
    fig.savefig(path_result + "timeseries_boxplot_"+ label[x] + ".png")

###### sequence alignment of generated with uperin
seqs_generated =ProcessSeqs(fasta_all[0]).get_seqs()
seqs_uperin =ProcessSeqs(fasta_all[-1]).get_seqs()




