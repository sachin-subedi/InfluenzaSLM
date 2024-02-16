# InfluenzaSLM

Here are the steps I did:

1. Started an interactive session

```bash
interact --mem=10g
```

2. Load Python/3.10.4-GCCcore-11.3.0 with

```bash
ml Python/3.10.4-GCCcore-11.3.0
```

3. Create a virtual python environment with

```bash
python -m venv ~/env/genslm
```

4. Activate this env

```bash
.   ~/env/genslm/bin/activate
```

5. Install GenSLM in this env with
```bash
pip install git+https://github.com/ramanathanlab/genslm
```

6. Then, I modified embeddings.py python file forour dataset.

embedding.py is the file I used

```bash

import torch
import numpy as np
from torch.utils.data import DataLoader
from genslm import GenSLM, SequenceDataset
from Bio import SeqIO

# Initialize GenSLM model with a valid model_id that matches your model's architecture

model_path = '/scratch/ss11645/GenSLM/MLProject/models/patric_25m_epoch01-val_loss_0.57_bias_removed.pt'
model = GenSLM('genslm_25M_patric')  # This sets up the architecture

custom_model_state = torch.load(model_path, map_location=torch.device('cpu'))

# If the .pt file contains the model state under a specific key, adjust the key accordingly
model.load_state_dict(custom_model_state)

model.eval()  # Prepare the model for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load and prepare your data
fasta_file = '/scratch/ss11645/GenSLM/MLProject/Seperated_files/h3n2.64000.fasta'
sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, 'fasta')]

# Prepare dataset and dataloader
dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Compute embeddings
embeddings = []
with torch.no_grad():
    for batch in dataloader:
        outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), output_hidden_states=True)
        emb = outputs.hidden_states[-1].detach().cpu().numpy()
        emb = np.mean(emb, axis=2)
        embeddings.append(emb)

# Concatenate all embeddings
embeddings = np.concatenate(embeddings, axis=0)

# Output the shape of the embeddings array
print(embeddings.shape)
```

7. I submitted the job using emb.sh in Sapelo2 cluster

```bash

#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --partition=bahl_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --cpus-per-task=4
#SBATCH --time=500:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=ss11645@uga.edu
#SBATCH --mail-type=END,FAIL    #Mail events (NONE, BEGIN, END, FAIL, ALL)

cd $SLURM_SUBMIT_DIR

ml Python/3.10.4-GCCcore-11.3.0 
.   ~/env/genslm/bin/activate
python /scratch/ss11645/GenSLM/embeddings1.py
````