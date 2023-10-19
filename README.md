# causal_narrative
Source code for the paper "Causality in Language Model Features Rediscovers Cortical Hierarchy in Human Narrative Processing".

# Setting up the environment
sh env_manage_causal.sh
sh install.sh

# Dataset
We use Narrative Dataset:
"The 'Narratives' fMRI dataset for evaluating models of naturalistic language comprehension"
datalad install -r ///labs/hasson/narratives
Then, use "dataled get" commend to get specific dataset.

# Data Preprocess
Use code inside workspace/notebooks/DataPreprocess to do tokenization, alignment, and encode.

# Causal relationship calculation
Use code inside workspace/notebooks/CausalInferMain

Hack into to huggingface opt source at modeling_opt.py, line 715, inside OPTDecoder.forward()
############## Edited for causal study 23/08/31 #############
            if self.config.noise_insert:
                noise_para = self.config.noise_para
                if idx == noise_para["insert_layer"]:
                    insert_position = noise_para["insert_position"]
                    # hidden_states[0,insert_position,:] = hidden_states[0,insert_position,:] + torch.randn_like(hidden_states[0,insert_position,:]) * 0.01
                    add_noise = noise_para["add_noise"]
                    hidden_states = hidden_states + add_noise.to(hidden_states.device)
############## End Edited for causal study 23/08/31 #############

# Brain fitting
Using code in workspace/python/brain_score_allsubs.py


