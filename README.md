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
"The 'Narratives' fMRI dataset for evaluating models of naturalistic language comprehension".  
To get the dataset, go to the dataset fold and run:
```
datalad install -r ///labs/hasson/narratives
```
Then, go to the folder "dataset" and run:
```
sh get_data.sh
```
The scripy will download about 200GB of processed fsaverage6 fMRI dataset of all subjects with datalad.

We also use GlasserAtlas and LanAAtlas when analysing the data. These atlases can be found in dataset folder.  

# Data Preprocess
Use code inside workspace/notebooks/DataPreprocess.ipynb to do tokenization, alignment, and encode.  
The code would generate opts folder inside dataset folder, and stores tokenization, alignment, and encode data for each task.  
The encoding process may take tens of minutes.

# Causal Relationship Calculation
Use code inside workspace/notebooks/CausalInferMain  
Hack into to huggingface opt source at modeling_opt.py,
```
~/anaconda3/envs/causal/lib/python3.8/site-packages/transformers/models/opt/modeling_opt.py
```
line 715, inside OPTDecoder.forward(), inside loop 
```
for idx, decoder_layer in enumerate(self.layers):
```
after hidden_states = layer_outputs[0], insert:
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

Then, run jupyter notebook workspace/notebooks/CausalInferMain.ipynb.  
This file would create two folders of features for high indegree and low indegree each under each task folder of dataset.  
Note that the sampling process to get causal matrix may take hours.

# Brain fitting
Use code in workspace/python/top/brain_score_allsubs.py and workspace/python/bottom/brain_score_allsubs.py to fit all fMRI data.  
The python file in top folder would fit brain using features with high in-degree.  
The python file in bottom folder would fit brain using features with low in-degree.
The correlation coeffitients are output in coef_mat_for_all_subs.data file.
It may take up to an hour to fit all data.

# Data Visualization
The fitting result can be visualized using:  
workspace/notebooks/DataVisualization.ipynb


