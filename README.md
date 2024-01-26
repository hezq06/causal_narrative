# causal_narrative
Source code for the paper "Causality in Language Model Features Rediscovers Cortical Hierarchy in Human Narrative Processing".

# Dependency
The following are required packages to run the project. (Inside paranthes are version number used when writing the paper.)
Python (3.8.16)
Datalad (0.16.1)
PyTorch (2.1.0)
Huggingface (4.29.2)
Nilearn (0.10.1)
Sklearn (0.23.0)
PyVista (0.39.1)
The software was originally developed on Rocky Linux 8.7.

# Hardware Requirement
This project assumes a GPU accelerator with cuda capability.
We assume cuda 12.1 when installing pytorch. Please change "env_manage_causal.sh" to match your version of cuda.

# Setting up the environment
```
sh env_manage_causal.sh
source activate causal
sh install.sh
```

The installation would take several minutes.

# Dataset
We use Narrative Dataset:
"The 'Narratives' fMRI dataset for evaluating models of naturalistic language comprehension"
datalad install -r ///labs/hasson/narratives
Then, use "dataled get" commend to get specific dataset.

# Data Preprocess
Use code inside workspace/notebooks/DataPreprocess to do tokenization, alignment, and encode.

The encoding process may take tens of minutes.

# Causal relationship calculation
Use code inside workspace/notebooks/CausalInferMain

Hack into to huggingface opt source at modeling_opt.py, line 715, inside OPTDecoder.forward()
```
############## Edited for causal study 23/08/31 #############
            if self.config.noise_insert:
                noise_para = self.config.noise_para
                if idx == noise_para["insert_layer"]:
                    add_noise = noise_para["add_noise"]
                    hidden_states = hidden_states + add_noise.to(hidden_states.device)
                    # Or Equivalently
                    # insert_position = noise_para["insert_position"]
                    # hidden_states[0,insert_position,:] = hidden_states[0,insert_position,:] + torch.randn_like(hidden_states[0,insert_position,:]) * 0.01
############## End Edited for causal study 23/08/31 #############
```
# Brain fitting
Using code in workspace/python/brain_score_allsubs.py

It may take up to an hour to fit all data.


