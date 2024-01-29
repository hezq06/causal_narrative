"""
Python package for project brain scale estimate
Using Opt series to estimate brain scale
Utility for handling dataset
Developer: Harry He
2023-4-01
"""
import os, glob, copy, shutil
import pickle, json, csv
import nibabel as nib
from tqdm import tqdm, tqdm_notebook
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, OPTModel
import numpy as np
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn import datasets
import nibabel as nib
import pyvista as pv
from ncautil.ncamath import *
import scipy
# import cortex
from collections import Counter
import time

def save_data(data,file,large_data=False, engine="pickle"):
    if engine=="pickle":
        try:
            pickle.dump(data, open(file, "wb"))
            print("Data saved to ", file)
        except:
            pickle.dump(data, open(file, "wb"), protocol=4)
            print("Large Protocal 4 Data saved to ", file)
    elif engine=="json":
        json.dump(data, open(file, "w"))
        print("Data saved to ", file)
    else:
        raise Exception("Unknown Engine.")

def load_data(file, engine="pickle", print_flag=True):
    if engine == "pickle":
        data = pickle.load(open(file, "rb"))
    elif engine=="json":
        data = json.load(open(file, "r"))
    elif engine=="tsv":
        data = []
        with open(file) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                data.append(row)
    elif engine=="gii":
        surf_data = nib.load(file)
        try:
            data = surf_data.agg_data()
        except:
            data = surf_data.get_fdata()
    elif engine=="nii":
        data = nib.load(file)
    elif engine=="txt":
        # Open the file
        with open(file, 'r') as fp:
            # Read the file
            data = fp.read()
    else:
        raise Exception("Unknown Engine.")
    if print_flag:
        print("Data load from ", file)
    return data

def kfold_gen(length, n_fold=5):
    """
    Create n_fold cross validation masks
    :param length:
    :param n_fold:
    :return:
    """
    valid_bool = np.zeros((n_fold, length)).astype(bool)
    train_bool = np.zeros((n_fold, length)).astype(bool)

    sep = int(length/n_fold)
    for ii in range(n_fold):
        valid_bool[ii, ii*sep:(ii+1)*sep] = True

    train_bool = ~valid_bool

    train_bool[:, n_fold * sep:] = False # throw away remaining data points
    valid_bool[:, n_fold * sep:] = False

    return train_bool, valid_bool

def detect_nan(npmat):
    """
    Check if contains nan
    :param npmat:
    :return:
    """
    if np.isnan(npmat).any():
        npmat = np.nan_to_num(npmat)
    assert not np.isnan(npmat).any()

class NarrativeDataPreprocess(object):
    """
    Preproces of narrative dataset
    HBTdataset preparation for transcripts
    Model output alighment to fMRI TR data
    """
    def __init__(self, config={}):
        self.config(config)
        if "opt" in self.opt_model:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False) # Others have issue? According to Alpa (Not true)
        elif self.opt_model=="gpt2":
            self.tokenizer = AutoTokenizer.from_pretrained(self.opt_model, use_fast=False)
        elif "clip" in self.opt_model:
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        elif "t5" in self.opt_model:
            self.tokenizer = T5Tokenizer.from_pretrained("DeepFloyd/IF-I-XL-v1.0", subfolder="text_encoder")
        self.prepare_meta()

    def config(self, config):
        current_path = os.getcwd()
        narrative_home = os.path.join(current_path, "../../dataset/narratives")
        self.narrative_home = config.get("narrative_home", narrative_home)
        opts_home = os.path.join(current_path, "../../dataset/opts")
        self.opts_home = config.get("narrative_home", opts_home)
        # self.task = config.get("task", "pieman")
        # self.task_aligndata = config.get("task_aligndata", "pieman")
        # self.task_path = config.get("task_path", "stimuli/gentle")
        # self.subj = config.get("subj", "sub-001")
        # self.hemi_side = config.get("hemi_side", "L")
        self.trt = config.get("trt", 1.5) # time duration of a tr is 1.5

        self.opt_model=config.get("opt_model", "opt-30b") # gpt2, clip
        self.tasks_exclude = ["notthefalllongscram", "notthefallshortscram", "schema"]

        if "opt" in self.opt_model:
            self.gpt_helpers = [4, 6, 22, 480, 60, 12, 1917, 35, 2901, 113, 1174, 23962, 72, 116, 36856, 27223, 131,
                                734, 328, 126, 48474, 36, 43, 482, 43775, 108, 43003, 46353, 845, 50121, 50118, 128, 6600,
                                955, 111, 578, 20551, 37008, 93, 2383, 9957, ] # helpers for Opt
        elif "gpt2" in self.opt_model:
            self.gpt_helpers = [12, 11, 13, 1, 366, 1701, 526, 553, 30, 438, 1377, 25, 1058, 2474, 1399, 9962, 11097, 9313, 26,
                                2162, 986, 2644, 0, 5145, 1906, 784, 42720, 7, 8, 357, 1267, 1539, 30823, 6, 705, 35751, 24426,
                                1911, 30, 628, 198, 201, 8348, 2637, 960, 532, 851, 19056, 20995, 11496] # helpers for Gpt2
        elif "clip" in self.opt_model:
            self.gpt_helpers = [269, 267, 2432, 257, 3823, 268, 3283, 281, 2947, 959, 20215, 1081, 286, 12089, 5240, 282,
                                678, 256, 1224, 0, 263, 264, 5276, 29315, 262, 30, 44493, 33993, 2811, 49407, 2005, 13610,
                                5938, 6718, 12, 41558]
        elif "t5" in self.opt_model:
            self.gpt_helpers = [5, 6, 1636, 96, 976, 18, 4609, 10, 4720, 121, 233, 10011, 535, 58, 117, 55, 104, 41, 61,
                                31, 1280, 318]

    def prepare_meta(self):
        self.task_meta = load_data(os.path.join(self.narrative_home,"code/task_meta.json"), engine="json")
        self.tasks = list(self.task_meta.keys())
        tasks_copy = copy.deepcopy(self.tasks)
        print(self.tasks), print(self.tasks_exclude)
        for task in tasks_copy:
            if task in self.tasks_exclude:
                self.tasks.remove(task)
                print("Task %s removed"%task)

        self.task_sub_dict = dict([])
        for task in self.tasks:
            self.task_sub_dict[task] = list(self.task_meta[task].keys())

        scan_exclude = load_data(os.path.join(self.narrative_home,"code/scan_exclude.json"), engine="json")
        for task in self.tasks:
            scan_exclude_dict = scan_exclude[task]
            exclude_subs = list(scan_exclude_dict.keys())
            for sub in exclude_subs:
                if sub in self.task_sub_dict[task]:
                    self.task_sub_dict[task].remove(sub)
                    # print("Task, sub pair %s, %s removed."%(task, sub))

        self.sub_task_dict = dict([])
        for task, subl in self.task_sub_dict.items():
            for sub in subl:
                if sub not in self.sub_task_dict.keys():
                    self.sub_task_dict[sub] = []
                self.sub_task_dict[sub].append(task)

        # Check if the directory exists
        if not os.path.exists(self.opts_home):
            # Create the directory
            os.makedirs(self.opts_home)
        for task in self.tasks:
            if task not in self.tasks_exclude:
                task_path = os.path.join(self.opts_home, task)
                if not os.path.exists(task_path):
                    # Create the directory
                    os.makedirs(task_path)

        # Copy milkywayoriginal to be milkyway

        # Define the source and destination directories
        milkyway_original = os.path.join(self.narrative_home,"stimuli/gentle/milkywayoriginal")
        milkyway = os.path.join(self.narrative_home,"stimuli/gentle/milkyway")

        # Check if destination directory exists, if not, create it
        if not os.path.exists(milkyway):
            os.makedirs(milkyway)

            # Copy each file from the source directory to the destination directory
            for filename in os.listdir(milkyway_original):
                file_path = os.path.join(milkyway_original, filename)

                # Check if it's a file and not a directory
                if os.path.isfile(file_path):
                    shutil.copy(file_path, milkyway)

    def text_clean(self, raw_file):
        raw_file = raw_file.replace("‘", "\'")
        raw_file = raw_file.replace("’", "\'")
        raw_file = raw_file.replace("“", "\"")
        raw_file = raw_file.replace("”", "\"")
        if "clip" in self.opt_model or "t5" in self.opt_model or "no_case" in self.opt_model:
            raw_file = raw_file.lower() # clip and t5 seems to has no capital.
        if "t5" in self.opt_model:
            raw_file = raw_file.replace("12,","12") # handle strange t5 tokenizer "12,"

        return raw_file

    def tokenization(self):
        for task in self.tasks:
            if task not in self.tasks_exclude:
                data = load_data(os.path.join(self.narrative_home,"stimuli/gentle/%s/align.json") % task, engine="json")
                text = self.text_clean(data["transcript"])
                if "opt" in self.opt_model:
                    input_ids = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()[1:]
                    save_data(input_ids, os.path.join(os.path.join(self.opts_home,task),
                                           "%s_tokenization.pickle"%self.opt_model))
                elif "gpt2" in self.opt_model:
                    input_ids = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()[1:]
                    save_data(input_ids, os.path.join(os.path.join(self.opts_home,task),
                                                  "gpt2_tokenization.pickle"))
                elif "clip" in self.opt_model:
                    input_ids = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()[1:-1]
                    save_data(input_ids, os.path.join(os.path.join(self.opts_home, task),
                                           "clip_tokenization.pickle"))
                elif "t5" in self.opt_model:
                    input_ids = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()[:-1]
                    save_data(input_ids, os.path.join(os.path.join(self.opts_home,task),
                                           "t5_tokenization.pickle"))

    def get_fsaverage6_gii_file(self, task):
        subj = self.task_sub_dict[task][0]
        path_to_list = os.path.join(self.narrative_home, "derivatives/afni-nosmooth/%s/func"%subj)
        file_to_list = "%s_task-%s*space-fsaverage6_hemi-L_desc-clean.func.gii"%(subj, task)
        path_to_list = os.path.join(path_to_list, file_to_list)
        flist = glob.glob(path_to_list)
        # print(flist)
        return flist

    def run_alignment(self, task):

        aligndata = load_data(os.path.join(self.narrative_home,"stimuli/gentle/%s/align.json" % task), engine="json")
        # enc_ids = load_data(os.path.join("/home/hezq17/MyWorkSpace/brain_estimate/data/narrative/%s" % task,
        #              "opt_tokenization.pickle"))
        enc_ids = load_data(os.path.join(os.path.join(self.opts_home,task),
                                         "%s_tokenization.pickle"%self.opt_model))

        align_pt = 0
        for item in aligndata["words"]:
            item["word"] = self.text_clean(item["word"])
            item["temp_word"] = item["word"]

        for ii in tqdm_notebook(range(len(enc_ids))):
            cwdpt = aligndata["words"][align_pt]
            insert_flag = self.insert_bpe_ptl(cwdpt, enc_ids[ii])
            if not insert_flag:
                align_pt = align_pt + 1
                cwdpt = aligndata["words"][align_pt]
                insert_flag = self.insert_bpe_ptl(cwdpt, enc_ids[ii])
                if not insert_flag:
                    print(ii, align_pt)
                assert insert_flag

        # Cross check
        totlen = 0
        for item in aligndata["words"]:
            totlen = totlen + len(item['bpe_ptl'])
        assert totlen == len(enc_ids)
        self.totlen = totlen
        return aligndata

    def gptid_to_vocab(self, id):
        token = self.tokenizer._convert_id_to_token(id)
        if token.startswith("Ġ"):
            token = token[1:]
        elif "clip" in self.opt_model and token.endswith("</w>"):
            token = token.replace('</w>', "")
        elif "t5" in self.opt_model and token.startswith("▁"):
            token = token.replace('▁', "")
        return token

    def insert_bpe_ptl(self, cwdpt, bpe, debug=False): # Use debug = True when finding gpt_helpers

        bpe = bpe.item()
        if cwdpt.get("bpe_ptl", None) is None:
            cwdpt["bpe_ptl"] = []
            cwdpt["vocabl"] = []
        if cwdpt["temp_word"].startswith(self.gptid_to_vocab(bpe).strip()):
            cwdpt["bpe_ptl"].append(bpe)
            cwdpt["vocabl"].append(self.gptid_to_vocab(bpe))
            if debug:
                print(cwdpt["temp_word"], self.gptid_to_vocab(bpe).strip())
            stripl = len(self.gptid_to_vocab(bpe).strip())
            cwdpt["temp_word"] = cwdpt["temp_word"][stripl:]
            # if len(cwdpt["temp_word"])>0:
            #     cwdpt["temp_word"] = "##" + cwdpt["temp_word"]
            if debug:
                print("Replaced", cwdpt["temp_word"], self.gptid_to_vocab(bpe).strip())
            return True
        elif bpe in self.gpt_helpers:
            cwdpt["bpe_ptl"].append(bpe)
            cwdpt["vocabl"].append(self.gptid_to_vocab(bpe))
            if debug:
                print(cwdpt["temp_word"], self.gptid_to_vocab(bpe).strip())
            return True
        else:
            if debug:
                print("False", cwdpt["temp_word"], self.gptid_to_vocab(bpe).strip(), "bpe: ", bpe)
            return False

    def gen_alignment_mats(self):
        for task in self.tasks:
            if task not in self.tasks_exclude:
                mat = self.gen_alignment_mat(task)
                save_data(mat, os.path.join(os.path.join(self.opts_home,task),
                                                  "%s_alignmat.pickle"%self.opt_model))

    def gen_alignment_mat(self, task):
        """
        Generate alignment matrix, input shape: bpe, outputshape
        :return:
        """
        flist = self.get_fsaverage6_gii_file(task)
        data = load_data(flist[0], engine="gii")
        trslen = data.shape[-1]
        aligndata = self.run_alignment(task)
        bpelen = self.totlen
        alignmat = np.zeros((bpelen, trslen))

        bpe_pt = 0
        start_mem = 0
        for item in aligndata["words"]:
            startt = item.get('start', start_mem)
            start_mem = startt
            ctr = int(np.floor(startt / self.trt))
            lenbpe = len(item['bpe_ptl'])
            for iib in range(lenbpe):
                alignmat[bpe_pt + iib, ctr] = 1
            bpe_pt = bpe_pt + lenbpe
        alignmat = alignmat / (np.sum(alignmat, axis=0, keepdims=True) + 1e-9)
        return alignmat

class GptEncoder(object):
    def __init__(self, config={}):
        self.config(config)
        self.gpt = AutoModel.from_pretrained(self.model_name).to(self.cuda_device)
        self.gpt.config.output_hidden_states = True
        self.gpt = self.gpt.to(self.cuda_device)
        self.gpt.eval()
        self.wiki = WikiText103Dataset(config)

    def config(self, config):
        self.model_name = config.get("model_name", "gpt2")
        self.cuda_device = config.get("cuda_device", "cuda:0")
        self.batch_size = config.get("batch_size", 16)
        # self.pad_id = config.get("pad_id", 1)  # Opt pad id
        self.pad_id = config.get("pad_id", 50256)

    # def encode(self):
    #     wiki_loader = self.wiki.get_dataloader(self.batch_size)
    #     hidden_states_l = []
    #     for input_ids in tqdm_notebook(wiki_loader):
    #         input_ids = input_ids.to(self.cuda_device)
    #         with torch.no_grad():
    #             output = self.gpt(input_ids)
    #         hidden_states = torch.stack(output.hidden_states, dim=0).detach().cpu()
    #         # print(hidden_states.shape)
    #         hidden_states_l.append(hidden_states)
    #     hidden_states_l = torch.cat(hidden_states_l,dim=1)
    #     print(hidden_states_l.shape)
    #     return hidden_states_l

    def encode(self, window=128, dix=0):
        """
        ttbert uses wordpiece
        :param bert:
        :param window:
        :param task:
        :return:
        """

        gptcodel = []
        input_ids_l = []
        input_ids_raw = self.wiki[dix].to(self.cuda_device)

        for iib in tqdm(range(len(input_ids_raw))):
            if iib < window -1:
                # input_ids = input_ids_raw[:(iib + 1)]
                continue # skip shorter ones
            else:
                input_ids = input_ids_raw[iib - window +1:iib]
            input_ids = input_ids.view(1, -1)
            with torch.no_grad():
                output = self.gpt(input_ids)
            hidden_states = torch.cat(output.hidden_states[:-1], dim=0)
            gptcodel.append(hidden_states[:, -1, :].unsqueeze(1).detach().cpu())  # [lnum, 768]
            input_ids_l.append(input_ids.view(-1)[-1].item())
        gptcodeall = torch.cat(gptcodel, dim=1).numpy()  # [lnum, 7???, 768]
        return gptcodeall, input_ids_l

    def prepare_batch_input(self, input_ids_raw, window=128):
        """
        Batch up all input for high through put model parallel
        :return:
        """

        batch_input = torch.zeros([len(input_ids_raw), window]).type(torch.LongTensor)
        for iib in range(len(input_ids_raw)):
            if iib < window-1 :
                input_ids = input_ids_raw[:(iib + 1)]
            else:
                input_ids = input_ids_raw[iib - window +1:iib]
            # input_ids = torch.cat((torch.LongTensor([2]), input_ids)) # for opt only
            if len(input_ids)<window: # For generative model, left padding should be used
                input_ids = torch.cat((torch.LongTensor([self.pad_id]*(window-len(input_ids))), input_ids)) # 1 is pad id
            batch_input[iib,:] = input_ids
        return batch_input

    def opt_encode_batch(self, window=128, task="21styear", chuck_size=64, cuda_device="cuda:1"):
        """
        ttbert uses wordpiece
        :param bert:
        :param window:
        :param task:
        :return:
        """
        # self.opt = OPTModel.from_pretrained("facebook/" + self.opt_model).half().to(cuda_device)
        # self.opt = OPTModel.from_pretrained("facebook/" + self.opt_model).to(cuda_device)
        self.opt = AutoModel.from_pretrained(self.opt_model).to(cuda_device)
        self.opt.config.output_hidden_states = True
        self.opt.eval()

        print("Preparing batch input ...")
        batch_input = self.prepare_batch_input(window=window, task=task)
        batch_input = batch_input.to(cuda_device)

        num_chuncks = int(len(batch_input) / chuck_size) + 1
        gptcodel = []

        for iib in tqdm(range(num_chuncks)):
            input_chuck = batch_input[iib * chuck_size: (iib + 1) * chuck_size, :]
            with torch.no_grad():
                output = self.opt(input_chuck)
                hidden_states = torch.stack(output.hidden_states[:-1])

            output = hidden_states[:, :, -1, :].detach().cpu()
            gptcodel.append(output)  # [lnum, chuck_size ,768]

        gptcodeall = torch.cat(gptcodel, dim=1)  # [lnum, 7???, 768]
        return gptcodeall

class FSPlot(object):
    """
    A utility to plot fMRI result in fsaverage6 space
    """
    def __init__(self, config={}):
        fsaverage6 = datasets.fetch_surf_fsaverage("fsaverage6")
        brain = load_data(fsaverage6['infl_left'], engine="gii")
        labels, ctab, names = nib.freesurfer.read_annot('/home/hezq17/dataset/GlasserAtlas/lh.HCPMMP1.annot') # Glasser Atlas
        lana = nib.load("/home/hezq17/dataset/LanAAtlas/LH_LanA_n804.nii.gz").get_fdata().reshape(-1)
        # Down sampling to fs6
        self.length = len(brain[0])
        lana6 = lana[:len(brain[0])]
        self.labels6 = labels[:len(brain[0])]

        # Array 1: list of vertices coordinates
        vertices = brain[0]

        # Array 2: list of 3 vertices labels
        faces = brain[1]
        num_verts = np.ones(faces.shape[0]) * 3
        faces = np.concatenate([num_verts.reshape(-1, 1), faces], axis=1).astype(int)

        # points, namel = cal_atlas_center(labels, vertices, names)
        self.points, self.namel = self.cal_atlas_center(self.labels6, vertices, names)

        # Create a new mesh (PolyData)
        self.mesh = pv.PolyData(vertices, faces)
        # mesh["lana"] = lana
        self.mesh["lana"] = lana6
        self.mesh["labels"] = self.labels6 / 180

        # edge_faces = detect_atlas_edges(labels,faces)
        edge_faces = self.detect_atlas_edges(self.labels6, faces)
        self.mesh_edge = pv.PolyData(vertices, edge_faces)

        # Visualize the mesh
        self.left = [-1, 0, 0]
        # mesh.plot(scalars = "labels", cpos=left, cmap=lut)

    def get_language_network(self):
        self.language_network={
            12:"55b", 79:"IFJa", 81:"IFSp", 74:"44", 75:"45", 76:"47l", 131:"TGd", 123:"STGa", 125:"A5",
            128:"STSda", 176:"STSva", 132:"TE1a", 129:"STSdp", 130:"STSvp", 25:"PSL", 28:"STV", 139:"TPOJ1",
            150:"PGi", 43:"SCEF", 26:"SFL"
        }
        return self.language_network

    def get_brain_mask_with_labels(self, labels):
        assert type(labels) == list
        masks = []
        for label in labels:
            mask = self.get_brain_mask_with_label(label)
            masks.append(mask)

        allmask = np.logical_or.reduce(masks)
        return allmask

    def get_brain_mask_with_label(self, label):
        iil = []
        for ii, name in enumerate(self.namel):
            if label == name:
                iil.append(ii)
        mask = np.zeros(self.length,dtype=bool)
        for ii in iil:
            mask[self.labels6==ii]=True
        assert mask.any()
        return mask

    def cal_atlas_center(self, labels, vertices, names):
        """
        calculate the center of each atlas area
        return the index of point which is closes to the center, and the atlas label
        """
        points = []
        namel = []
        for label in set(labels):
            mask = (labels == label)
            picked = vertices[mask]
            picked_ids = np.array(range(len(vertices)))[mask]
            mean_posi = np.mean(picked, axis=0)
            dist = np.sum((picked - mean_posi) ** 2, axis=1)
            min_point = np.argmin(dist)
            min_point = picked_ids[min_point]
            name = names[label][2:-4]
            points.append(min_point)
            namel.append(name.decode('ascii'))
        return np.array(points), namel

    def detect_atlas_edges(self, labels, faces):
        """
        Return simplified faces contains only edge points
        """
        edge_faces = []
        for face in faces:
            face_labels = labels[face[1:]]
            if len(set(face_labels)) > 1:
                edge_faces.append(face)
        return edge_faces

    def plot(self, fsscalars, clim=None):
        plotter = pv.Plotter()
        plotter.add_point_labels(self.mesh.points[self.points], self.namel, point_size=10, font_size=20)
        # plotter.add_mesh(mesh, scalars = "labels", cmap=lut, opacity = "linear")
        scalar_bar_args = {"vertical": True, "height": 0.8, "position_y":0.1,
                           "title_font_size": 20, "label_font_size": 20, "color": "black"}

        # cmap = "inferno", "seismic"
        cmap = "seismic"
        if clim is None:
            plotter.add_mesh(self.mesh, scalars=fsscalars, cmap=cmap, scalar_bar_args=scalar_bar_args)
        else:
            plotter.add_mesh(self.mesh, scalars=fsscalars, cmap=cmap, clim=clim, scalar_bar_args=scalar_bar_args)
        plotter.add_mesh(self.mesh_edge, color=[0.2, 0.2, 0.2])

        # Set the camera view
        # camera_position = [(-300, 0, 0), (0, -10, 0), (0, 0, 1)] # left
        camera_position = [(320, 0, 0), (0, 10, 0), (0, 0, 1)]
        plotter.background_color = 'white'
        plotter.camera_position = camera_position

        # plotter.show(cpos=self.left)
        plotter.show(cpos=camera_position)

    def plot_atlas(self):
        plotter = pv.Plotter()
        plotter.add_point_labels(self.mesh.points[self.points], self.namel, point_size=10, font_size=20)
        # plotter.add_mesh(mesh, scalars = "labels", cmap=lut, opacity = "linear")
        plotter.add_mesh(self.mesh, scalars="lana", cmap="seismic")
        plotter.add_mesh(self.mesh_edge, color="black")
        plotter.show(cpos=self.left)











