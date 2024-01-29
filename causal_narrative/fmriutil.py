"""
Python package for project brain scale estimate
Using Opt series to estimate brain scale
Developer: Harry He
2023-4-01
"""
from causal_narrative.datautil import *
import numpy as np
import torch
from transformers import OPTModel, OPTConfig, GPT2Model
import nilearn
from nilearn import plotting
from sklearn.decomposition import FastICA

class FmriUtil(object):
    """
    FMRIutil for Narrative dataset
    """

    def __init__(self, config={}):
        self.config(config)

        # self.feature_alg = None
        # self.wrdembed_alg = None
        self.zero_mask = None
        self.opt = None
        self.coefl_validation_mem = []

    def config(self, config):
        current_path = os.getcwd()
        opts_home = os.path.join(current_path, "../../dataset/opts")
        self.data_home = config.get("data_home", opts_home)
        self.opt_model = config.get("opt_model", "opt-125m") # 30b layer 48
        self.k_fold = config.get("k_fold", 5)
        self.pad_id = config.get("pad_id", 1) # Opt pad id
        self.pca_dim = config.get("pca_dim", 20) #
        self.fmri_split = config.get("fmri_split", 1) # trying to split 40962 fsaverage if GPU memory overflow
        self.project_mode = config.get("project_mode", "pca") # pca, ica, slow_feature_analysis_slow, slow_feature_analysis_fast
        self.layer = config.get("layer", 8)

        self.mode_vec_feature = config.get("mode_vec_feature", list(range(-6,0))) # FIR previous encoding encluded
        # self.mode_vec_embed = config.get("mode_vec_embed", [0]) # Wordvector enhancing
        # self.include_embed = config.get("include_embed", False) # Including Wordvector to enhance
        self.mode_vec_pred = config.get("mode_vec_pred", [0,1])  # Wordvector enhancing
        self.include_pred = config.get("include_pred", False)  # Including Wordvector to enhance
        self.t_max = config.get("t_max", 3)
        self.t_min = config.get("t_min", -9)

        self.ldyngpt2_folder = config.get("ldyngpt2_folder", None)
        self.afni_smooth = config.get("afni_smooth", True)
        self.alpha = config.get("alpha", 10.**np.array(range(-1,9)))
        self.alpha_mode = config.get("alpha_mode", "normal") # normal, fix

        self.manual_path = config.get("manual_path", "Ica_PFbot_27946")
        self.shuffle_flag = config.get("shuffle_flag", False)
        self.hemi = config.get("hemi", "L")

    # def prepare_batch_input(self, window=128, task="21styear"):
    #     """
    #     Batch up all input for high through put model parallel
    #     Attention version
    #     :return:
    #     """
    #     if "opt" in self.opt_model:
    #         input_ids_raw = load_data(os.path.join(self.data_home, "%s/opt_tokenization.pickle" % task))
    #     elif "gpt2" in self.opt_model:
    #         input_ids_raw = load_data(os.path.join(self.data_home, "%s/gpt2_tokenization.pickle" % task))
    #
    #     batch_input = torch.zeros([len(input_ids_raw), window]).type(torch.LongTensor)
    #     attention_mask = torch.ones([len(input_ids_raw), window]).type(torch.LongTensor)
    #     for iib in range(len(input_ids_raw)):
    #         if iib < window-1 :
    #             input_ids = input_ids_raw[:(iib + 1)]
    #         else:
    #             input_ids = input_ids_raw[iib - window +1:iib]
    #         input_ids = torch.cat((torch.LongTensor([2]), input_ids))
    #         if len(input_ids)<window: # For generative model, left padding should be used
    #             input_ids = torch.cat((torch.LongTensor([self.pad_id]*(window-len(input_ids))), input_ids)) # 1 is pad id
    #             attention_mask[iib,:window-len(input_ids)]=0
    #         batch_input[iib,:] = input_ids
    #     return batch_input, attention_mask

    def opt_encode_causalstudy(self, window=128, task="21styear", cuda_device="cuda:1", insert_layer=3, model_type = "opt-125m"):
        """
        Hacked opt_model for causal relationship study
        """
        # self.opt = OPTModel.from_pretrained("facebook/opt-125m").to(cuda_device)
        if model_type == "opt-125m":
            model_str = "facebook/%s" % model_type
            model_dim=768
            self.opt = OPTModel.from_pretrained(model_str).to(cuda_device)
            input_ids_raw = load_data(os.path.join(self.data_home, "%s/opt_no_case_tokenization.pickle" % task))
        elif model_type == "opt-350m":
            model_str = "facebook/%s" % model_type
            model_dim = 1024
            self.opt = OPTModel.from_pretrained(model_str).to(cuda_device)
            input_ids_raw = load_data(os.path.join(self.data_home, "%s/opt_tokenization.pickle" % task))
        elif model_type == "gpt2":
            model_str = "gpt2"
            model_dim = 768
            self.opt = GPT2Model.from_pretrained(model_str).to(cuda_device)
            input_ids_raw = load_data(os.path.join(self.data_home, "%s/gpt2_tokenization.pickle" % task))
        else:
            raise Exception("Unknown model_type")

        self.opt.config.output_hidden_states = True

        gptcodel = []
        gptcodel_noise = []
        noise_paral = []
        self.opt.eval()

        for iib in tqdm_notebook(range(len(input_ids_raw))):

            if iib < window:
                input_ids = input_ids_raw[:(iib + 1)]
            else:
                # input_ids = input_ids_raw[iib - window +1:iib] # old, can be wrong
                input_ids = input_ids_raw[iib - window + 1: iib + 1]
            if "opt" in self.opt_model:
                input_ids = torch.cat((torch.LongTensor([2]), input_ids)).view(1, -1).to(cuda_device)
            elif "gpt2" in self.opt_model:
                input_ids = input_ids.view(1, -1).to(cuda_device)

            ##### with no noise
            self.opt.config.noise_insert = False
            with torch.no_grad():
                output = self.opt.to(cuda_device)(input_ids)
            hidden_states = torch.cat(output.hidden_states[:-1], dim=0)
            output = hidden_states[:, -1, :].detach().cpu()
            gptcodel.append(output.unsqueeze(1))  # [lnum, 768]

            ##### with noise
            self.opt.config.noise_insert = True
            seql = len(input_ids[0])
            add_noise = torch.zeros(1, seql, model_dim)
            insert_position = int(np.random.rand() * np.min([iib, 20]))
            add_noise_vec = (torch.rand(model_dim)-0.5) * 0.01
            add_noise[:, -insert_position, :] = add_noise_vec

            noise_para = {
                "insert_layer": insert_layer,
                "insert_position": -insert_position,
                "add_noise": add_noise_vec
            }
            self.opt.config.noise_para = noise_para
            noise_paral.append(noise_para)

            with torch.no_grad():
                output = self.opt.to(cuda_device)(input_ids)
            hidden_states = torch.cat(output.hidden_states[:-1], dim=0)
            output = hidden_states[:, -1, :].detach().cpu()
            gptcodel_noise.append(output.unsqueeze(1))  # [lnum, 768]

        gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
        gptcodeall_noise = torch.cat(gptcodel_noise, dim=1)  # [lnum, 7???, 768]

        return gptcodeall, gptcodeall_noise, noise_paral

    def opt_encode(self, window=128, task="21styear", cuda_device="cuda:1"):
        """
        ttbert uses wordpiece
        :param bert:
        :param window:
        :param task:
        :return:
        """
        if "untrained" in self.opt_model:
            print("Found untrained keyword!")
            opt_model_strip = self.opt_model.replace("_untrained","")
            tmp_opt = OPTModel.from_pretrained("facebook/" + opt_model_strip).to(cuda_device)
            self.opt = OPTModel(tmp_opt.config)
            input_ids_raw = load_data(os.path.join(self.data_home, "%s/opt_tokenization.pickle" % task))
        elif "opt" in self.opt_model and "no_case" in self.opt_model:
            opt_model = self.opt_model.replace("_no_case","")
            self.opt = OPTModel.from_pretrained("facebook/" + opt_model).to(cuda_device)
            input_ids_raw = load_data(os.path.join(self.data_home, "%s/opt_no_case_tokenization.pickle" % task))
        elif "opt" in self.opt_model:
            self.opt = OPTModel.from_pretrained("facebook/" + self.opt_model).to(cuda_device)
            input_ids_raw = load_data(os.path.join(self.data_home, "%s/opt_tokenization.pickle" % task))
        elif self.opt_model == "gpt2":
            self.opt = AutoModel.from_pretrained(self.opt_model).to(cuda_device)
            input_ids_raw = load_data(os.path.join(self.data_home, "%s/gpt2_tokenization.pickle" % task))
        else:
            raise Exception("Unknown opt_model")

        # self.opt = OPTModel.from_pretrained(self.opt_model)
        # self.opt = AutoModel.from_pretrained(self.opt_model).to(cuda_device)
        self.opt.config.output_hidden_states = True
        # input_ids_raw = load_data(os.path.join(self.data_home, "%s/opt_tokenization.pickle" % task))

        gptcodel = []
        self.opt.eval()

        for iib in tqdm(range(len(input_ids_raw))):
            if iib < window:
                input_ids = input_ids_raw[:(iib + 1)]
            else:
                # input_ids = input_ids_raw[iib - window +1:iib] # old, can be wrong
                input_ids = input_ids_raw[iib - window + 1: iib +1]
            if "opt" in self.opt_model:
                input_ids = torch.cat((torch.LongTensor([2]),input_ids)).view(1,-1).to(cuda_device)
            elif "gpt2" in self.opt_model:
                input_ids = input_ids.view(1, -1).to(cuda_device)
            with torch.no_grad():
                output = self.opt.to(cuda_device)(input_ids)
            hidden_states = torch.cat(output.hidden_states[:-1], dim=0)
            output = hidden_states[:, -1, :].detach().cpu()
            gptcodel.append(output.unsqueeze(1))  # [lnum, 768]
        gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
        save_data(gptcodeall,os.path.join(self.data_home, "%s/opt_encode_%s_w%s.data" % (task, self.opt_model, window)))
        return gptcodeall

    def clip_encode_att(self, window=75, task="21styear", cuda_device="cuda:0"):
        """
        ttbert uses wordpiece
        :param bert:
        :param window:
        :param task:
        :return:
        """
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = text_encoder.to(cuda_device)

        # self.opt = OPTModel.from_pretrained(self.opt_model)
        # self.opt = AutoModel.from_pretrained(self.opt_model).to(cuda_device)
        text_encoder.config.output_hidden_states = True
        input_ids_raw = load_data(os.path.join(self.data_home, "%s/clip_tokenization.pickle" % task))

        gptcodel = []
        text_encoder.eval()

        for iib in tqdm(range(len(input_ids_raw))):
            if iib < window:
                input_ids = input_ids_raw[:iib]
                padding = torch.LongTensor((window-iib)*[49407])
                input_ids = torch.cat((torch.LongTensor([49406]), input_ids, torch.LongTensor([49407]), padding)).view(1,-1).to(cuda_device)
                attention_mask = torch.ones(window+2)
                attention_mask[iib-window:]=0
                attention_mask = attention_mask.view(1,-1).to(cuda_device)
            else:
                input_ids = input_ids_raw[iib - window:iib]
                input_ids = torch.cat((torch.LongTensor([49406]), input_ids, torch.LongTensor([49407]))).view(1, -1).to(cuda_device)
                attention_mask = torch.ones(window + 2)
                attention_mask = attention_mask.view(1, -1).to(cuda_device)
            with torch.no_grad():
                output = text_encoder(input_ids, attention_mask)
            # hidden_states = torch.cat(output.hidden_states, dim=0)
            hidden_states = output.pooler_output
            output = hidden_states.detach().cpu()
            # if iib < window:
            #     idpick = iib+1
            # else:
            #     idpick = -1
            # gptcodel.append(output[:, idpick, :].unsqueeze(1))  # [lnum, 768]
            # gptcodel.append(torch.sum(output,dim=1).unsqueeze(1))
            gptcodel.append(output.unsqueeze(1))  # [lnum, 768]
        gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
        save_data(gptcodeall,os.path.join(self.data_home, "%s/learning_dynamics/clip_encode_w%s.data" % (task, window)))
        return gptcodeall

    def t5_encode_att(self, window=76, task="21styear", cuda_device="cuda:0"):
        from transformers import T5EncoderModel
        text_encoder = T5EncoderModel.from_pretrained("DeepFloyd/IF-I-XL-v1.0", subfolder="text_encoder")
        text_encoder = text_encoder.to(cuda_device)

        # self.opt = OPTModel.from_pretrained(self.opt_model)
        # self.opt = AutoModel.from_pretrained(self.opt_model).to(cuda_device)
        # text_encoder.config.output_hidden_states = True
        input_ids_raw = load_data(os.path.join(self.data_home, "%s/t5_tokenization.pickle" % task))

        gptcodel = []
        text_encoder.eval()

        for iib in tqdm(range(len(input_ids_raw))):
            if iib < window:
                input_ids = input_ids_raw[:iib]
                padding = torch.LongTensor((window - iib) * [0])
                input_ids = torch.cat((input_ids, torch.LongTensor([1]), padding)).view(1, -1).to(cuda_device)
                attention_mask = torch.ones(window + 1)
                attention_mask[iib - window:] = 0
                attention_mask = attention_mask.view(1, -1).to(cuda_device)
            else:
                input_ids = input_ids_raw[iib - window:iib]
                input_ids = torch.cat((input_ids, torch.LongTensor([1]))).view(1, -1).to(cuda_device)
                attention_mask = torch.ones(window + 1)
                attention_mask = attention_mask.view(1, -1).to(cuda_device)
            with torch.no_grad():
                output = text_encoder(input_ids, attention_mask)
            hidden_states = output[0]
            output = hidden_states.detach().cpu()
            if iib < window:
                idpick = iib
            else:
                idpick = -1
            gptcodel.append(output[:, idpick, :].unsqueeze(1))  # [lnum, 768]
            # gptcodel.append(torch.sum(output,dim=1).unsqueeze(1))
            # gptcodel.append(output.unsqueeze(1))  # [lnum, 768]
        gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
        save_data(gptcodeall, os.path.join(self.data_home, "%s/t5_encode_w%s.data" % (task, window)))
        return gptcodeall

    def t5_encode_block(self, window=76, task="21styear", cuda_device="cuda:0"):
        from transformers import T5EncoderModel
        text_encoder = T5EncoderModel.from_pretrained("DeepFloyd/IF-I-XL-v1.0", subfolder="text_encoder")
        text_encoder = text_encoder.to(cuda_device)

        # self.opt = OPTModel.from_pretrained(self.opt_model)
        # self.opt = AutoModel.from_pretrained(self.opt_model).to(cuda_device)
        # text_encoder.config.output_hidden_states = True
        input_ids_raw = load_data(os.path.join(self.data_home, "%s/t5_tokenization.pickle" % task))

        gptcodel = []
        text_encoder.eval()

        num_blocks = int(np.ceil(len(input_ids_raw)/window))

        for iib in range(num_blocks):
            input_ids = input_ids_raw[iib*window:(iib+1)*window]
            input_ids = torch.cat((input_ids, torch.LongTensor([1]))).view(1, -1).to(cuda_device)
            attention_mask = torch.ones(len(input_ids))
            attention_mask = attention_mask.view(1, -1).to(cuda_device)
            with torch.no_grad():
                output = text_encoder(input_ids, attention_mask)
            hidden_states = output[0]
            output = hidden_states.detach().cpu().squeeze()
            gptcodel.append(output[:-1,:])
        gptcodeall = torch.cat(gptcodel, dim=0).unsqueeze(0)  # [lnum, 7???, 768]
        assert len(input_ids_raw) == gptcodeall.shape[1]
        save_data(gptcodeall, os.path.join(self.data_home, "%s/t5_encode_w%s.data" % (task, window)))
        return gptcodeall

    def opt_encode_batch(self, window=128, task="21styear", chuck_size=64, cuda_device="cuda:1"):
        """
        ttbert uses wordpiece
        :param bert:
        :param window:
        :param task:
        :return:
        """
        self.opt = OPTModel.from_pretrained("facebook/" + self.opt_model).half().to(cuda_device)
        # self.opt = OPTModel.from_pretrained("facebook/" + self.opt_model).to(cuda_device)
        # self.opt = AutoModel.from_pretrained(self.opt_model).to(cuda_device)
        self.opt.config.output_hidden_states = True
        self.opt.eval()

        print("Preparing batch input ...")
        batch_input = self.prepare_batch_input(window=window, task=task)
        batch_input = batch_input.to(cuda_device)

        num_chuncks = int(len(batch_input) / chuck_size) + 1
        gptcodel = []

        chunck_pt = 0
        mem_offload_flag=False
        for iib in tqdm(range(num_chuncks)):
            input_chuck = batch_input[iib * chuck_size: (iib + 1) * chuck_size, :]
            with torch.no_grad():
                output = self.opt(input_chuck)
                hidden_states = torch.stack(output.hidden_states[:-1])

            output = hidden_states[:, :, -1, :].detach().cpu()
            gptcodel.append(output)  # [lnum, chuck_size ,768]

            if psutil.virtual_memory()[2]>90: # To prevent memory overflow, 90% of memory used
                mem_offload_flag = True
                print('RAM memory % used:', psutil.virtual_memory()[2])
                gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
                save_data(gptcodeall, os.path.join(self.data_home, "%s/opt_encode_batch_%s_w%s_pc%s.data" % (task, self.opt_model, window, chunck_pt)))
                chunck_pt = chunck_pt + 1
                gptcodel = []

        gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
        if mem_offload_flag:
            save_data(gptcodeall, os.path.join(self.data_home, "%s/opt_encode_batch_%s_w%s_pc%s.data" % (task, self.opt_model, window, chunck_pt)))
        else:
            save_data(gptcodeall,os.path.join(self.data_home, "%s/opt_encode_batch_%s_w%s.data" % (task, self.opt_model, window)))

        return True

    def opt_encode_batch_attention(self, window=128, task="21styear", chuck_size=64, cuda_device="cuda:1"):
        """
        ttbert uses wordpiece
        :param bert:
        :param window:
        :param task:
        :return:
        """
        self.opt = OPTModel.from_pretrained("facebook/" + self.opt_model).half().to(cuda_device)
        # self.opt = OPTModel.from_pretrained("facebook/" + self.opt_model).to(cuda_device)
        # self.opt = AutoModel.from_pretrained(self.opt_model).to(cuda_device)
        self.opt.config.output_hidden_states = True
        self.opt.eval()

        print("Preparing batch input ...")
        batch_input, attention_mask = self.prepare_batch_input(window=window, task=task)
        batch_input = batch_input.to(cuda_device)
        attention_mask = attention_mask.to(cuda_device)

        num_chuncks = int(len(batch_input) / chuck_size) + 1
        gptcodel = []

        chunck_pt = 0
        mem_offload_flag=False
        for iib in tqdm(range(num_chuncks)):
            input_chuck = batch_input[iib * chuck_size: (iib + 1) * chuck_size, :]
            input_mask = attention_mask[iib * chuck_size: (iib + 1) * chuck_size, :]
            with torch.no_grad():
                output = self.opt(input_chuck, input_mask)
                hidden_states = torch.stack(output.hidden_states[:-1])

            output = hidden_states[:, :, -1, :].detach().cpu()
            gptcodel.append(output)  # [lnum, chuck_size ,768]

            if psutil.virtual_memory()[2]>90: # To prevent memory overflow, 90% of memory used
                mem_offload_flag = True
                print('RAM memory % used:', psutil.virtual_memory()[2])
                gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
                save_data(gptcodeall, os.path.join(self.data_home, "%s/opt_encode_att_%s_w%s_pc%s.data" % (task, self.opt_model, window, chunck_pt)))
                chunck_pt = chunck_pt + 1
                gptcodel = []

        gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
        if mem_offload_flag:
            save_data(gptcodeall, os.path.join(self.data_home, "%s/opt_encode_att_%s_w%s_pc%s.data" % (task, self.opt_model, window, chunck_pt)))
        else:
            save_data(gptcodeall,os.path.join(self.data_home, "%s/opt_encode_att_%s_w%s.data" % (task, self.opt_model, window)))

        return True

    def opt_encode_llm(self, window=128, task="21styear"):
        """
        ttbert uses wordpiece
        :param bert:
        :param window:
        :param task:
        :return:
        """
        # config = AutoConfig.from_pretrained("facebook/" + self.opt_model)
        config = OPTConfig.from_pretrained("facebook/" + self.opt_model)
        with init_empty_weights():
            self.opt = OPTModel(config)
        self.opt.tie_weights()
        self.opt.config.output_hidden_states = True
        self.opt.eval()

        ## Do it once
        # opt = AutoModel.from_pretrained("facebook/" + self.opt_model)
        # a = input("Wait for weights_location:")

        if self.opt_model == "opt-13b":
            weights_location = "/home/hezq17/.cache/huggingface/hub/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/pytorch_model.bin.index.json"
        elif self.opt_model == "opt-30b":
            weights_location = "/home/hezq17/.cache/huggingface/hub/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546/pytorch_model.bin.index.json"

        # Load the checkpoint and dispatch it to the right devices
        self.opt = load_checkpoint_and_dispatch(
            self.opt, weights_location, device_map="auto", no_split_module_classes=["OPTDecoderLayer"],
            max_memory={0: "30GiB", 1: "30GiB", 2: "30GB","cpu": "300GiB"}
            # offload_folder = "/home/hezq17/MyWorkSpace/brain_estimate/explore/offload"
        )

        input_ids_raw = load_data(os.path.join(self.data_home, "%s/opt_tokenization.pickle" % task))
        gptcodel = []
        chunck_pt = 0
        for iib in tqdm(range(len(input_ids_raw))):
            if iib < window:
                input_ids = input_ids_raw[:iib]
            else:
                input_ids = input_ids_raw[iib - window +1:iib]
            input_ids = torch.cat((torch.LongTensor([2]),input_ids)).view(1,-1).cuda()
            with torch.no_grad():
                output = self.opt(input_ids)
            hidden_states = torch.cat(output.hidden_states[:-1], dim=0)
            output = hidden_states.detach().cpu()
            gptcodel.append(output[:, -1, :].unsqueeze(1))  # [lnum, 768]
            if (iib+1)%1000 == 0:
                gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
                save_data(gptcodeall, os.path.join(self.data_home, "%s/opt_encode_%s_pc%s.data" % (task,self.opt_model, chunck_pt)))
                chunck_pt = chunck_pt+1
                gptcodel = []
        gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
        save_data(gptcodeall, os.path.join(self.data_home, "%s/opt_encode_%s_pc%s.data" % (task,self.opt_model, chunck_pt)))

        datal = []
        for ii in range(chunck_pt+1):
            data = load_data(os.path.join(self.data_home, "%s/opt_encode_%s_pc%s.data" % (task,self.opt_model, ii)))
            datal.append(data)
            # os.remove(os.path.join(self.data_home, "%s/opt_encode_%s_pc%s.data" % (task, self.opt_model, ii)))
        data_all = torch.cat(datal, dim=1)
        save_data(data_all, os.path.join(self.data_home, "%s/opt_encode_%s.data" % (task,self.opt_model)))

        return gptcodeall

    def opt_encode_llm_batch(self, window=128, task="21styear", chuck_size=64, model_ready=False):
        """
        ttbert uses wordpiece
        :param bert:
        :param window:
        :param task:
        :return:
        """
        if not model_ready:
            config = AutoConfig.from_pretrained("facebook/" + self.opt_model)
            # print(config)
            with init_empty_weights():
                self.opt = OPTForCausalLM(config)
                # self.opt = OPTModel(config)
            self.opt.tie_weights()
            self.opt.config.output_hidden_states = True

            # Do it once
            # opt = AutoModel.from_pretrained("facebook/" + self.opt_model)
            # a = input("Wait for weights_location:")

            if self.opt_model == "opt-13b":
                weights_location = "/home/hezq17/.cache/huggingface/hub/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/pytorch_model.bin.index.json"
            elif self.opt_model == "opt-30b":
                weights_location = "/home/hezq17/.cache/huggingface/hub/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546/pytorch_model.bin.index.json"
            elif self.opt_model == "opt-66b":
                weights_location = "/home/hezq17/.cache/huggingface/hub/models--facebook--opt-66b/snapshots/7259969061237fe940036d22bea0fd349e4485e9/pytorch_model.bin.index.json"

            # Load the checkpoint and dispatch it to the right devices
            print("Loading checkpoint and dispatch ...")
            self.opt = load_checkpoint_and_dispatch(
                self.opt, weights_location, device_map="auto", no_split_module_classes=["OPTDecoderLayer"],
                max_memory={0: "7GiB", 1: "20GiB", 2: "20GB","cpu": "300GiB"},
                # max_memory={0: "20GiB", 1: "20GiB", "cpu": "300GiB"},
                # max_memory={0: "10GB", "cpu": "300GiB"},
                dtype=torch.half
                # offload_folder = "/home/hezq17/MyWorkSpace/brain_estimate/explore/offload"
            )
            self.opt.eval()

        print("Preparing batch input ...")
        batch_input = self.prepare_batch_input(window=window, task=task)
        batch_input = batch_input.cuda()

        num_chuncks = int(len(batch_input)/chuck_size)+1
        gptcodel = []
        chunck_pt = 0
        mem_offload_flag = False

        for iib in tqdm(range(num_chuncks)):
            input_chuck = batch_input[iib*chuck_size: (iib+1)*chuck_size, :]
            with torch.no_grad():
                output = self.opt(input_chuck)
            outputl = []
            for ii in range(len(output.hidden_states)-1):
                outputl.append(output.hidden_states[ii][:,-1,:].detach().cpu())
            # hidden_states = torch.stack(output.hidden_states[:-1])
            output = torch.stack(outputl)
            gptcodel.append(output)  # [lnum, chuck_size ,768]

            if psutil.virtual_memory()[2] > 90:  # To prevent memory overflow, 90% of memory used
                mem_offload_flag = True
                print('RAM memory % used:', psutil.virtual_memory()[2])
                gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
                save_data(gptcodeall, os.path.join(self.data_home, "%s/opt_encode_%s_pc%s.data" % (task, self.opt_model, chunck_pt)))
                chunck_pt = chunck_pt + 1
                gptcodel = []

        gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
        if mem_offload_flag:
            save_data(gptcodeall,
                      os.path.join(self.data_home, "%s/opt_encode_%s_pc%s.data" % (task, self.opt_model, chunck_pt)))
        else:
            save_data(gptcodeall, os.path.join(self.data_home, "%s/opt_encode_%s.data" % (task, self.opt_model)))

        return True

    def opt_embed(self, task="21styear"):
        """
        Producing only the word embedding of each word
        :param task:
        :return:
        """
        self.opt = OPTModel.from_pretrained("facebook/" + self.opt_model)
        input_ids_raw = load_data(os.path.join(self.data_home, "%s/opt_tokenization.pickle" % task))
        embedcodeall = self.opt.decoder.embed_tokens.cuda()(input_ids_raw.view(1,-1).cuda()).squeeze()
        embedcodeall = embedcodeall.detach().cpu()
        save_data(embedcodeall, os.path.join(self.data_home, "%s/opt_embed_%s.data" % (task, self.opt_model)))
        return embedcodeall

    def prepare_feature_alltasks(self, window=128):

        if self.project_mode != "manual":
            pM = self.get_shared_pm(window=window)
        else:
            pM = None

        ndp = NarrativeDataPreprocess()
        self.feature_alltasks=dict([])
        for task in ndp.tasks:
            feature_alg, zero_mask, train_bool, valid_bool = self.prepare_feature(window=window, task=task, pM=pM)
            self.feature_alltasks[task] = (feature_alg, zero_mask, train_bool, valid_bool)
        save_data(self.feature_alltasks, "feature_alltasks.data")

    # def get_shared_pm(self, window=128):
    #     """
    #     Get shared projection matrix
    #     :return:
    #     """
    #     ndp = NarrativeDataPreprocess()
    #
    #     feature_alg_l = []
    #
    #     for task in ndp.tasks:
    #
    #         if "untrained" in self.opt_model:
    #             print("Found untrained keyword!")
    #             opt_model_strip = self.opt_model.replace("_untrained", "")
    #             mat = load_data(os.path.join(self.data_home, "%s/opt_alignmat.pickle" % task))
    #             feature_all = load_data(os.path.join(self.data_home, "%s/opt_encode_%s_w%s.data" % (task, opt_model_strip, window)))
    #         elif "opt" in self.opt_model and "no_case" in self.opt_model:
    #             mat = load_data(os.path.join(self.data_home, "%s/opt_no_case_alignmat.pickle" % task))
    #             feature_all = load_data(os.path.join(self.data_home, "%s/learning_dynamics/opt_encode_%s_w%s.data" % (task, self.opt_model, window)))
    #         elif "opt" in self.opt_model:
    #             mat = load_data(os.path.join(self.data_home, "%s/opt_alignmat.pickle" % task))
    #             feature_all = load_data(
    #                 os.path.join(self.data_home, "%s/opt_encode_%s_w%s.data" % (task, self.opt_model, window)))
    #         elif self.opt_model == "gpt2":
    #             mat = load_data(os.path.join(self.data_home, "%s/gpt2_alignmat.pickle" % task))
    #             feature_all = load_data(
    #                 os.path.join(self.data_home, "%s/opt_encode_%s_w%s.data" % (task, self.opt_model, window)))
    #         elif "ldyngpt2" in self.opt_model:
    #             mat = load_data(os.path.join(self.data_home, "%s/gpt2_alignmat.pickle" % task))
    #             feature_all = load_data(
    #                 os.path.join(self.data_home, "%s/learning_dynamics/opt_encode_%s_w%s.data" % (task, self.opt_model, window)))
    #         elif "clip" in self.opt_model:
    #             mat = load_data(os.path.join(self.data_home, "%s/clip_alignmat.pickle" % task))
    #             feature_all = load_data(
    #                 os.path.join(self.data_home, "%s/learning_dynamics/clip_encode_w%s.data" % (task, window)))
    #         elif "t5" in self.opt_model:
    #             mat = load_data(os.path.join(self.data_home, "%s/t5_alignmat.pickle" % task))
    #             feature_all = load_data(
    #                 os.path.join(self.data_home, "%s/t5_encode_w%s.data" % (task, window)))
    #         else:
    #             raise Exception("Unknown self.opt_model")
    #
    #         if self.layer is None:
    #             layer = int((8 / 12) * feature_all.shape[0]) + 1
    #         else:
    #             layer = self.layer
    #
    #         feature = feature_all[layer, :, :].type(torch.FloatTensor).numpy()
    #         feature_alg = mat.T.dot(feature)
    #
    #         if self.project_mode == "pca_shared":
    #             zero_feature = ((feature_alg == 0)[:, 0]).astype(int).reshape(-1, 1)
    #             feature_alg = np.concatenate([zero_feature, feature_alg], axis=-1)
    #             feature_alg_l.append(feature_alg)
    #
    #         elif self.project_mode in ["pca_shared_zero","pca_shared_zero2"]:
    #             zero_mask = (feature_alg != 0)[:, 0]
    #             feature_alg_l.append(feature_alg[zero_mask])
    #
    #     feature_alg_l = np.concatenate(feature_alg_l, axis=0)
    #     feature_alg, pM = pca_proj(scipy.stats.zscore(feature_alg_l.T), self.pca_dim)
    #     save_data(pM,"pca_proj_mat.data")
    #     save_data(feature_alg_l,"feature_alg_l_pca_shared_zero.data")

        # return pM

    def get_shared_pm(self, window=128):

        ndp = NarrativeDataPreprocess()
        feature_alg_l = []

        for task in ndp.tasks:

            if "opt" in self.opt_model and "no_case" in self.opt_model:
                mat = load_data(os.path.join(self.data_home, "%s/opt_no_case_alignmat.pickle" % task))
                feature_all = load_data(os.path.join(self.data_home, "%s/learning_dynamics/opt_encode_%s_w%s.data" % (task, self.opt_model, window)))
            else:
                raise Exception("Unknown self.opt_model")

            if self.layer is None:
                layer = int((8 / 12) * feature_all.shape[0]) + 1
            else:
                layer = self.layer

            feature = feature_all[layer, :, :].type(torch.FloatTensor).numpy()
            if self.project_mode in ["mat_zscoret_pca","mat_zscore_pca","pca_shared_zero","mat_pca","pca_shared_zerot","mat_pca_zscore",\
                                     "mat_pca_zscoresep","mat_ica_zscoresep","manual"]:
                feature_alg = mat.T.dot(feature)
                zero_mask = (feature_alg != 0)[:, 0]
                feature_alg_l.append(feature_alg[zero_mask,:])
            else:
                raise Exception("Unknown project_mode")

        feature_alg_l = np.concatenate(feature_alg_l, axis=0)

        if self.project_mode in ["mat_zscoret_pca","pca_shared_zero","pca_shared_zerot"]:
            _, pM = pca_proj(scipy.stats.zscore(feature_alg_l.T), self.pca_dim)
        elif self.project_mode == "mat_zscore_pca":
            _, pM = pca_proj(scipy.stats.zscore(feature_alg_l).T, self.pca_dim)
        elif self.project_mode in ["mat_pca","mat_pca_zscore","mat_pca_zscoresep"]: # Pick mat_pca_zscore
            _, pM = pca_proj(feature_alg_l.T, self.pca_dim) # used to be this
        elif self.project_mode in ["mat_ica_zscoresep"]:
            res, pMpca = pca_proj(feature_alg_l.T, self.pca_dim)
            ica = FastICA(n_components=10)
            X_ica = ica.fit_transform(res.T)
            pM = (pMpca, ica)
        elif self.project_mode in ["manual"]:
            pM = None
        else:
            raise Exception("Unknown project_mode")

        return pM

    def prepare_feature(self, window=128, task="21styear", pM=None):

        if self.project_mode != "manual":
            if "opt" in self.opt_model and "no_case" in self.opt_model:
                mat = load_data(os.path.join(self.data_home, "%s/opt_no_case_alignmat.pickle" % task))
                feature_all = load_data(os.path.join(self.data_home, "%s/learning_dynamics/opt_encode_%s_w%s.data" % (task, self.opt_model, window)))
            else:
                raise Exception("Unknown self.opt_model")

            if self.layer is None:
                layer = int((8 / 12) * feature_all.shape[0]) + 1
            else:
                layer = self.layer

            feature = feature_all[layer, :, :].type(torch.FloatTensor).numpy()
        else:
            layer = self.layer

        if self.project_mode == "mat_zscoret_pca":
            feature_alg = mat.T.dot(feature)
            zero_mask = (feature_alg != 0)[:, 0]
            zero_feature = ((feature_alg == 0)[:, 0]).astype(int).reshape(-1, 1)
            feature_alg[zero_mask,:] = scipy.stats.zscore(feature_alg[zero_mask,:], axis=1)
            feature_alg = feature_alg.dot(pM.T)
            feature_alg = np.concatenate([zero_feature, feature_alg], axis=-1)
        elif self.project_mode == "mat_zscore_pca":
            feature_alg = mat.T.dot(feature)
            zero_mask = (feature_alg != 0)[:, 0]
            zero_feature = ((feature_alg == 0)[:, 0]).astype(int).reshape(-1, 1)
            feature_alg[zero_mask, :] = scipy.stats.zscore(feature_alg[zero_mask, :], axis=0)
            feature_alg = feature_alg.dot(pM.T)
            feature_alg = np.concatenate([zero_feature, feature_alg], axis=-1)
        elif self.project_mode == "pca_shared_zero": # best, but logic wrong
            feature_alg = mat.T.dot(feature)
            zero_mask = (feature_alg != 0)[:, 0]
            zero_feature = ((feature_alg == 0)[:, 0]).astype(int).reshape(-1, 1)
            feature_alg = feature_alg.dot(pM.T)
            feature_alg = np.concatenate([zero_feature, feature_alg], axis=-1)
            feature_alg = scipy.stats.zscore(feature_alg)
        elif self.project_mode == "pca_shared_zerot":
            feature_alg = mat.T.dot(feature)
            zero_mask = (feature_alg != 0)[:, 0]
            zero_feature = ((feature_alg == 0)[:, 0]).astype(int).reshape(-1, 1)
            feature_alg = feature_alg.dot(pM.T)
            feature_alg = np.concatenate([zero_feature, feature_alg], axis=-1)
            feature_alg = scipy.stats.zscore(feature_alg, axis=-1)
        elif self.project_mode == "mat_pca":
            feature_alg = mat.T.dot(feature)
            zero_mask = (feature_alg != 0)[:, 0]
            zero_feature = ((feature_alg == 0)[:, 0]).astype(int).reshape(-1, 1)
            feature_alg = feature_alg.dot(pM.T)
            feature_alg = np.concatenate([zero_feature, feature_alg], axis=-1)
        elif self.project_mode == "mat_pca_zscore": ############### Pick this as best ###############
            feature_alg = mat.T.dot(feature)
            zero_mask = (feature_alg != 0)[:, 0]
            zero_feature = ((feature_alg == 0)[:, 0]).astype(int).reshape(-1, 1)
            feature_alg = feature_alg.dot(pM.T)
            feature_alg = np.concatenate([zero_feature, feature_alg], axis=-1)
            feature_alg = scipy.stats.zscore(feature_alg)
        elif self.project_mode == "mat_pca_zscoresep":
            feature_alg = mat.T.dot(feature)
            zero_mask = (feature_alg != 0)[:, 0]
            zero_feature = ((feature_alg == 0)[:, 0]).astype(int).reshape(-1, 1)
            zero_feature = scipy.stats.zscore(zero_feature)
            feature_alg = feature_alg.dot(pM.T)
            feature_alg[zero_mask,:] = scipy.stats.zscore(feature_alg[zero_mask,:])
            feature_alg = np.concatenate([zero_feature, feature_alg], axis=-1)
        elif self.project_mode in ["mat_ica_zscoresep"]:
            feature_alg = mat.T.dot(feature)
            zero_mask = (feature_alg != 0)[:, 0]
            zero_feature = ((feature_alg == 0)[:, 0]).astype(int).reshape(-1, 1)
            zero_feature = scipy.stats.zscore(zero_feature)
            pMpca, ica = pM
            feature_alg = feature_alg.dot(pMpca.T)
            feature_alg = ica.transform(feature_alg)
            feature_alg[zero_mask, :] = scipy.stats.zscore(feature_alg[zero_mask, :])
            feature_alg = np.concatenate([zero_feature, feature_alg], axis=-1)
        elif self.project_mode == "manual":
            feature_alg, zero_mask = load_data(os.path.join(self.data_home, "%s/%s/opt_encode_%s_l%s.data" % (task, self.manual_path, self.opt_model, layer)))
            # feature_alg, zero_mask = load_data(os.path.join(self.data_home, "%s/%s/opt_encode_%s_l8.data" % (
            # task, self.manual_path, self.opt_model)))
        else:
            raise Exception("Unknown project mode")

        save_data(feature_alg, "test_data_feature_alg_l.data")

        feature_alg_cut, embed_alg_cut = self.produce_feature_fir(feature_alg, feature_alg)

        train_bool, valid_bool = kfold_gen(feature_alg_cut.shape[0], self.k_fold)

        return feature_alg_cut, zero_mask, train_bool, valid_bool


    def gii_preparation(self, task="21styear", sub="sub-001", align_shape=1, zero_mask=(True)):

        if self.afni_smooth:

            if task not in ["pieman"]:
                gii = load_data(
                    "../workspace/dataset/narratives/derivatives/afni-smooth/%s/func/%s_task-%s_space-fsaverage6_hemi-%s_desc-clean.func.gii" % (
                        sub, sub, task, self.hemi), engine="gii")
            else:
                try:
                    gii = load_data(
                        "../workspace/dataset/narratives/derivatives/afni-smooth/%s/func/%s_task-%s_run-1_space-fsaverage6_hemi-%s_desc-clean.func.gii" % (
                            sub, sub, task, self.hemi), engine="gii")
                except:
                    gii = load_data(
                        "../workspace/dataset/narratives/derivatives/afni-smooth/%s/func/%s_task-%s_space-fsaverage6_hemi-%s_desc-clean.func.gii" % (
                            sub, sub, task, self.hemi), engine="gii")

        else:

            if task not in ["pieman"]:
                gii = load_data(
                    "../workspace/dataset/narratives/derivatives/afni-nosmooth/%s/func/%s_task-%s_space-fsaverage6_hemi-%s_desc-clean.func.gii" % (
                        sub, sub, task, self.hemi), engine="gii")
            else:
                try:
                    gii = load_data(
                        "../workspace/dataset/narratives/derivatives/afni-nosmooth/%s/func/%s_task-%s_run-1_space-fsaverage6_hemi-%s_desc-clean.func.gii" % (
                            sub, sub, task, self.hemi), engine="gii")
                except:
                    gii = load_data(
                        "../workspace/dataset/narratives/derivatives/afni-nosmooth/%s/func/%s_task-%s_space-fsaverage6_hemi-%s_desc-clean.func.gii" % (
                            sub, sub, task, self.hemi), engine="gii")

        giir = self.shape_adjust(gii, align_shape)

        # giir = np.roll(gii, -6, axis=1) # should we roll zero mask due to hemodynamics?
        # giir = gii[:, zero_mask]
        # giir = np.roll(giir, 6, axis=1)

        gii_cut = self.produce_gii_cut(giir) # align to feature cut due to time rolling

        return gii_cut


    def fmri_regression(self, window=128):
        """

        :param task:
        :param layer:
        :param mode_vec_feature: How many delays included with the FIR model
        :param mode_vec_embed: Which word embeddings included
        :param include_embed: If word embedding is included
        :return:
        """

        self.prepare_feature_alltasks(window=window)

        coef_ll = []
        pM_ll = []
        hype_ind_ll = []

        ndp = NarrativeDataPreprocess()

        for sub in tqdm(ndp.sub_task_dict.keys()):
            print("Handling person ",sub)

            feature_alg_l = []
            gii_alg_l = []
            train_bool_l = []
            valid_bool_l = []
            task_l = []

            for task in ndp.sub_task_dict[sub]:

                feature_alg, zero_mask, train_bool, valid_bool = self.feature_alltasks[task]
                gii_alg = self.gii_preparation(task=task, sub=sub, align_shape=len(zero_mask), zero_mask=zero_mask)
                feature_alg_l.append(feature_alg)
                gii_alg_l.append(gii_alg)
                train_bool_l.append(train_bool)
                valid_bool_l.append(valid_bool)

            feature_alg = np.concatenate(feature_alg_l, axis=0)
            if self.shuffle_flag:
                np.random.shuffle(feature_alg)
            gii_alg = np.concatenate(gii_alg_l, axis=1)
            train_bool = np.concatenate(train_bool_l, axis=1)
            valid_bool = np.concatenate(valid_bool_l, axis=1)

            coef_l = []
            pM_l = []
            hype_ind_l = []
            for fold in range(self.k_fold):
                print("Fitting fold ", fold)
                train_mask = train_bool[fold]
                valid_mask = valid_bool[fold]
                coefl, pM, hype_ind = self.cal_regression(feature_alg, gii_alg, train_mask, valid_mask, task_l=task_l, current_subj = sub)  # nvox,
                coef_l.append(coefl)
                pM_l.append(pM)
                # hype_ind_l.append(hype_ind)
            coef_l = np.array(coef_l)  # k_fold, nvox
            pM_l = np.array(pM_l)# k_fold, 1+ fir*pca, nvox
            # hype_ind_l = np.array(hype_ind_l)

            coef_ll.append(np.nanmean(coef_l, axis=0))
            pM_ll.append(np.nanmean(pM_l, axis=0)) # sub, 1+ fir*pca, nvox
            # hype_ind_ll.append(np.nanmean(hype_ind_l, axis=0))

        coef_ll = np.array(coef_ll)  # n_person, nvox
        coef_pm = np.nanmean(coef_ll, axis=0)
        coef_vm = np.nanmean(coef_ll, axis=1)
        pM_ll = np.array(pM_ll)
        # hype_ind_pm = np.nanmean(hype_ind_ll, axis=0)

        return coef_pm, coef_vm, pM_ll

    def niplot(self, coef_np, output_file="coef_mean.png", threshold=0.005):

        fsaverage = nilearn.datasets.fetch_surf_fsaverage("fsaverage6")
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, coef_np, hemi='left',
            title="Some title", colorbar=True, vmax=0.23,
            threshold=threshold, bg_map=fsaverage.sulc_left, output_file=output_file)

    def cal_regression(self, feature_alg_cut, gii_cut, train_mask, valid_mask, nest_fold=4, task_l=None, current_subj=None):
        if self.alpha_mode == "fix":
            # return self.cal_regression_simple(feature_alg_cut, gii_cut, train_mask, valid_mask)
            hype_ind = load_data("/home/hezq17/MyWorkSpace/brain_estimate/explore/python/manuel_feature_align/hype_ind_pm.data")
            hype_ind = np.round(hype_ind)
            hype_ind = torch.from_numpy(hype_ind).cuda()
            coefl, pM, hype_ind = self.cal_regression_fix(feature_alg_cut, gii_cut, train_mask, valid_mask, hype_ind)
            return coefl, pM, hype_ind

        elif self.alpha_mode == "nest":
            if self.fmri_split == 1:
                coefl, pM, hype_ind = self.cal_regression_nest(feature_alg_cut, gii_cut, train_mask, valid_mask, nest_fold=nest_fold)
                return coefl, pM, hype_ind
            else:
                coefl_l = []
                pM_l = []
                hype_ind_l = []
                for ii_split in range(self.fmri_split):
                    len_vox = int(np.ceil(gii_cut.shape[0]/self.fmri_split))
                    gii_cut_split = gii_cut[ii_split*len_vox:(ii_split+1)*len_vox, :]
                    coefl, pM, hype_ind = self.cal_regression_nest(feature_alg_cut, gii_cut_split, train_mask, valid_mask, nest_fold=nest_fold, task_l=task_l, current_subj=current_subj)
                    coefl_l.append(coefl)
                    pM_l.append(pM)
                    hype_ind_l.append(hype_ind)
                coefl_l = np.concatenate(coefl_l)
                pM = np.concatenate(pM_l, axis=1)
                hype_ind_l = np.concatenate(hype_ind_l)
                return coefl_l, pM, hype_ind_l
        else:
            raise Exception("Unknown alpha_mode")

    def cal_regression_simple(self, feature_alg_cut, gii_cut, train_mask, valid_mask):
        pM = ridge_regression_cuda(feature_alg_cut[train_mask, :], np.nan_to_num(scipy.stats.zscore(gii_cut[:, train_mask].T)), alpha=100.0)
        y_pred = ridge_predict_cuda(feature_alg_cut[valid_mask, :], pM)
        coefl = cal_batch_corr_cuda(gii_cut[:, valid_mask], y_pred)
        return coefl.cpu().numpy()

    def cal_regression_fix(self, feature_alg_cut, gii_cut, train_mask, valid_mask, hype_ind):
        """
        Fixed alpha with pre-calculated hype_ind
        """

        X = feature_alg_cut[train_mask, :]
        y = np.nan_to_num(scipy.stats.zscore(gii_cut[:, train_mask].T))

        ## After getting hype_ind

        pM = ridge_regression_alphavox_cuda(X, y, hype_ind, alpha=self.alpha)
        y_pred = ridge_predict_cuda(feature_alg_cut[valid_mask, :], pM)
        coefl = cal_batch_corr_cuda(gii_cut[:, valid_mask], y_pred)
        coefl = coefl.detach().cpu().numpy()
        pM = pM.detach().cpu().numpy()
        # data_mem = (current_subj, task_l, valid_mask, pM, coefl)
        # self.coefl_validation_mem.append(data_mem)
        return coefl, pM, hype_ind

    def cal_regression_nest(self, feature_alg_cut, gii_cut, train_mask, valid_mask, nest_fold=4, task_l=None, current_subj=None):

        X = feature_alg_cut[train_mask, :]
        y = np.nan_to_num(scipy.stats.zscore(gii_cut[:, train_mask].T))
        nest_train, nest_valid = kfold_gen(X.shape[0], nest_fold)

        Xnt = []
        ynt = []
        for fold in range(nest_fold):
            Xnt.append(X[nest_train[fold], :])
            ynt.append(y[nest_train[fold], :])
        Xnt = np.stack(Xnt)
        ynt = np.stack(ynt)
        pM = ridge_regression_hypervalid_cuda(Xnt, ynt, alpha=self.alpha)

        Xnv = []
        ynv = []
        for fold in range(nest_fold):
            Xnv.append(X[nest_valid[fold], :])
            ynv.append(y[nest_valid[fold], :])
        Xnv = np.stack(Xnv)
        ynv = np.stack(ynv)
        y_pred = ridge_predict_hypervalid_cuda(Xnv, pM)

        coefl = cal_batch_corr_hypervalid_cuda(y_pred, ynv) # n_hype, n_folds, n_voxels
        coefl = torch.mean(coefl, dim = 1) # n_hype, n_voxels
        hype_ind = torch.argmax(coefl, dim=0)

        ## After getting hype_ind

        pM = ridge_regression_alphavox_cuda(X, y, hype_ind, alpha=self.alpha)
        y_pred = ridge_predict_cuda(feature_alg_cut[valid_mask, :], pM)
        coefl = cal_batch_corr_cuda(gii_cut[:, valid_mask], y_pred)
        coefl = coefl.detach().cpu().numpy()
        pM = pM.detach().cpu().numpy()
        hype_ind = hype_ind.detach().cpu().numpy()
        # data_mem = (current_subj, task_l, valid_mask, pM, coefl)
        # self.coefl_validation_mem.append(data_mem)
        return coefl, pM, hype_ind

    def produce_feature_fir(self, feature_alg, wrdembed_alg):
        """

        :param gii: [voxels, t]
        :param zero_feature: [t, 1]
        :param feature_alg: [t, 20]
        :param mode_vec: [-6]
        :return:
        """
        gii_cut = []
        # zero_feature_cut = []
        feature_alg_cut = []
        embed_alg_cut = []
        t_tot = feature_alg.shape[0]
        # t_max = np.max(mode_vec_feature + mode_vec_embed)
        # t_min = np.min(mode_vec_feature + mode_vec_embed)
        t_max = self.t_max
        t_min = self.t_min

        for t in range(t_tot):
            if t+t_min>=0 and t+t_max<t_tot:
                # zero_feature_sub = []
                feature_alg_sub = []
                embed_alg_sub = []
                for t_mode in self.mode_vec_feature:
                    t_pick = t+t_mode
                    # zero_feature_sub.append(zero_feature[t_pick,:])
                    feature_alg_sub.append(feature_alg[t_pick, :])
                for t_mode in self.mode_vec_pred:
                    t_pick = t+t_mode
                    embed_alg_sub.append(wrdembed_alg[t_pick, :])
                # zero_feature_cut.append(np.concatenate(zero_feature_sub))
                feature_alg_cut.append(np.concatenate(feature_alg_sub))
                embed_alg_cut.append(np.concatenate(embed_alg_sub))
                # feature_alg_cut.append(np.mean(feature_alg_sub, axis=0)) # very bad
                # embed_alg_cut.append(np.mean(embed_alg_sub, axis=0))
        return np.array(feature_alg_cut), np.array(embed_alg_cut)


    def produce_gii_cut(self, gii):
        """

        :param gii: [voxels, t]
        :param zero_feature: [t, 1]
        :param feature_alg: [t, 20]
        :param mode_vec: [-6]
        :return:
        """
        gii_cut = []
        # zero_feature_cut = []
        feature_alg_cut = []
        embed_alg_cut = []
        t_tot = gii.shape[1]
        # t_max = np.max(mode_vec_feature + mode_vec_embed)
        # t_min = np.min(mode_vec_feature + mode_vec_embed)
        t_max=self.t_max
        t_min=self.t_min

        for t in range(t_tot):
            if t+t_min>=0 and t+t_max<t_tot:
                # zero_feature_sub = []
                feature_alg_sub = []
                embed_alg_sub = []
                gii_cut.append(gii[:, t])
                # zero_feature_cut.append(np.concatenate(zero_feature_sub))
        return np.array(gii_cut).T

    def shape_adjust(self, gii, align_shape):
        """
        Due to gii shape variance, gii shape should be aligned
        :param gii:
        :param iis:
        :return:
        """
        if align_shape != gii.shape[1]:
            # print(align_shape, gii.shape[1])
            if  align_shape > gii.shape[1]:
                sdiff = align_shape - gii.shape[1]
                gii = np.concatenate((gii,gii[:, -sdiff:]), axis=1)
            else:
                sdiff = gii.shape[1] - align_shape
                gii = gii[:,:-sdiff]
        return gii

def gen_manuel_feature(layer=9, transform = None, save_flag=False, ica_flag = False):

    assert not(ica_flag and transform!=None), "Transform and ica should not use together."

    ndp = NarrativeDataPreprocess()
    data_home = "/home/hezq17/MyWorkSpace/brain_estimate/data/narrative"
    opt_model = "opt-125m_no_case"
    window = 128
    pca_dim = 10
    feature_alg_l = []
    for task in ndp.tasks:
        mat = load_data(os.path.join(data_home, "%s/opt_no_case_alignmat.pickle" % task))
        feature_all = load_data(
            os.path.join(data_home, "%s/learning_dynamics/opt_encode_%s_w%s.data" % (task, opt_model, window)))
        feature = feature_all[layer, :, :].type(torch.FloatTensor).numpy()
        feature_alg = mat.T.dot(feature)
        zero_mask = (feature_alg != 0)[:, 0]
        feature_alg_l.append(feature_alg[zero_mask])
    feature_alg_l = np.concatenate(feature_alg_l, axis=0)
    res, pM = pca_proj(feature_alg_l.T, pca_dim)
    if ica_flag:
        ica = FastICA(n_components=10, max_iter=1000)
        X_ica = ica.fit_transform(res.T)
    else:
        ica = None

    feature_alg_dict={}

    for task in ndp.tasks:
        mat = load_data(os.path.join(data_home, "%s/opt_no_case_alignmat.pickle" % task))
        feature_all = load_data(
            os.path.join(data_home, "%s/learning_dynamics/opt_encode_%s_w%s.data" % (task, opt_model, window)))
        feature = feature_all[layer, :, :].type(torch.FloatTensor).numpy()
        feature_alg = mat.T.dot(feature)

        zero_mask = (feature_alg != 0)[:, 0]
        zero_feature = ((feature_alg == 0)[:, 0]).astype(int).reshape(-1, 1)
        zero_feature = scipy.stats.zscore(zero_feature)

        feature_alg = feature_alg.dot(pM.T)

        if ica_flag:
            feature_alg = ica.transform(feature_alg)

        if transform is not None:
            transformation_matrix, bias_vector = transform
            feature_alg = np.dot(feature_alg, transformation_matrix.T) + bias_vector

        feature_alg[zero_mask, :] = scipy.stats.zscore(feature_alg[zero_mask, :])
        feature_alg_zero = np.concatenate([zero_feature, feature_alg], axis=-1)
        feature_alg_zero = scipy.stats.zscore(feature_alg_zero)  # mat_pca_zscoresep # not big deal

        feature_alg_dict[task] = (feature_alg_zero, zero_mask)

        if save_flag:
            save_data((feature_alg_zero, zero_mask), os.path.join(data_home, "%s/manual_feature_mat_ica_zscoresep/opt_encode_%s_l%s.data" % (task, opt_model, layer)))
            save_data((pM, ica, transform), os.path.join(data_home, "%s/manual_feature_mat_ica_zscoresep/transform_l%s" % (task, layer)))

    return feature_alg_dict

def fit_linear(m_target, m2):

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Create and fit the model
    model = LinearRegression(fit_intercept=True).fit(m2, m_target)

    # Get the transformation matrix and bias vector
    transformation_matrix = model.coef_
    bias_vector = model.intercept_

    # Apply the transformation to m2
    m2_transformed = np.dot(m2, transformation_matrix.T) + bias_vector

    # Calculate the RMS error
    rms_error = np.sqrt(mean_squared_error(m_target, m2_transformed))

    print(f"Transformation Matrix: \n{transformation_matrix}")
    print(f"Bias Vector: {bias_vector}")
    print(f"RMS Error: {rms_error}")

    transform = (transformation_matrix, bias_vector)

    return m2_transformed, transform

def linear_regress(Ztd, Yt):
    """
    Yt = Bd Zdt + U
    Bd = Yt Z'td (Zdt Z'td)^-1
    """
    ZZp = Ztd.T.dot(Ztd)
    ZZpi = np.linalg.pinv(ZZp)
    Bd = Yt.dot(Ztd).dot(ZZpi)
    Yft = Bd.dot(Ztd.T)
    return Bd, Yft

