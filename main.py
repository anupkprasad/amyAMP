from datetime import datetime
import torch
import os
import sys
from models_nn import train
from models_nn import model as model
from scripts import util  #, analysis_generated_seqs

# Add the project root directory to the system path
if os.path.dirname("~/workspace/amyAMP/") not in sys.path:
    sys.path.append(os.path.dirname("~/workspace/amyAMP/"))

# Set device
device = torch.device("cpu")  # Force CPU execution

# Paths and constants
path_root = os.path.expanduser(f"~/workspace/amyAMP/")
path_model = path_root + "/model_saved/"
path_result = f"{path_root}results"

# Ensure directories exist
os.makedirs(path_model, exist_ok=True)
os.makedirs(path_result, exist_ok=True)

# Training data paths
fasta_AMPs = [os.path.expanduser(f"~/workspace/amyAMP/data_master/amps/dbaasp/dbaasp_APR_processed.fasta")]
fasta_AMYs = [os.path.expanduser(f"~/workspace/amyAMP/data_master/amyloid/amys_uniqueAI4AMP_processedtotrain.fasta")]

# Analysis data paths
batch_generate = 1000
fasta_random = [os.path.expanduser(f"~/workspace/amyAMP/data_master/random_pep_uni.fasta")]
fasta_uperin = [os.path.expanduser(f"~/workspace/amyAMP/data_master/uperin.fasta")]
fasta_generated = [path_result + "seqs_generated" + str(batch_generate) + ".fasta"]
fasta_all = fasta_generated + fasta_AMPs + fasta_AMYs + fasta_random + fasta_uperin

# Other constants
conditions_AMPs = {"AMP Score": 0.95}
score_AMYs = 0.95
path_PC6 = f"{path_root}/data_master/physical_chemical_6.txt"

# Variables for training
batch_size = 128
epoch = 7000
run_num = 2
filter_max = 100


# Function to prepare the dataset
def prepare_dataset():
    table = util.get_conversion_table(path_PC6)
    seqs = {}
    for f in fasta_AMPs + fasta_AMYs:
        cur_seq = util.read_fasta(f)
        print("seqs_num= {}".format(len(cur_seq)))
        seqs.update(cur_seq)  # Merge dictionaries
    print("Total seqs = {}".format(len(seqs)))

    # PC6 encoding
    encoded = util.get_encoded_seqs(seqs, table)
    dataset = torch.utils.data.TensorDataset(torch.Tensor(encoded))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


# Function to train the model
def train_model(dataloader):
    start_time = datetime.now().strftime("%H:%M:%S")
    print(f"Training started at {start_time}")

    G, loss_all, collected_seqs = train.training(
        path_model, epoch=epoch, filter_max=filter_max, train_data=dataloader, run_num=run_num
    )

    finish_time = datetime.now().strftime("%H:%M:%S")
    print(f"Training finished at {finish_time}")
    return G, loss_all, collected_seqs


# Function to get model summary and generate peptides
def generate_peptides(path_model):
    #train.models_summary(filter_max, path_model)
        ### varriables
    table = util.get_conversion_table(path_PC6)
    G, E, D = model.get_model_and_optimizer() 
    E = E.to(device)
    G = G.to(device)
    D = D.to(device)
    E.apply(model.weights_init)
    G.apply(model.weights_init)
    D.apply(model.init_weights)
    optimizer_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), 
                                    lr=2e-5, betas=(0.5, 0.999), weight_decay=1e-5)
    optimizer_D = torch.optim.Adam(D.parameters(), 
                                lr=2e-5, betas=(0.5, 0.999), weight_decay=1e-5)
    run_num = 2

    all_parms = train.load_model(path_model, E, G, D, optimizer_EG, optimizer_D, run_num)
    E,G,D,optimizer_EG,optimizer_D,dataloader, loss_all, epoch_i = all_parms
    z = torch.randn(batch_size, filter_max, 1, 1).to(device)
    generated_seqs = util.generate_seqs(G, table, z)
    util.write_fasta(generated_seqs, os.path.join(path_model, "final_generated_seq.fasta"))
    print("Peptides generated successfully.")
    return


# Function to analyze generated data
def analyze_generated_data():
    print("Starting analysis of generated data...")
    analysis_generated_seqs.run_analysis(fasta_all, path_result)
    print("Analysis completed.")


# Main function to execute tasks
def main(task):
    dataloader = prepare_dataset()

    if task == "train":
        train_model(dataloader)
    elif task == "generate":
        generate_peptides(path_model=path_model)
    elif task == "analyze":
        analyze_generated_data()
    else:
        print("Invalid task. Please choose from 'train', 'generate', or 'analyze'.")


# Entry point
if __name__ == "__main__":
    # Set the task you want to perform: "train", "generate", or "analyze"
    task = "generate"  # Change this to "generate" or "analyze" as needed
    main(task)





