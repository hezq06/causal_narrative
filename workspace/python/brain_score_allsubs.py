from brain_estimate.fmriutil import *
from brain_estimate.datautil import *
import argparse

# ### Do Opt encoding with window 1024

# opt_ls = ["opt-125m"]
# ndp = NarrativeDataPreprocess()
#
# for opt_model in opt_ls:
#     print("Current opt model,",opt_model)
#     for task in ndp.tasks:
#         if task not in ndp.tasks_exclude:
#             futil = FmriUtil({"opt_model": opt_model})
#             gptcodeall = futil.opt_encode(window=1024, task=task, cuda_device="cuda:2") #<6.7b
torch.cuda.set_device("cuda:0")


parser = argparse.ArgumentParser()

# Add arguments
#parser.add_argument("--layer", required=True, help="Layer ID")
#args = parser.parse_args()
#layer_id = args.layer

config={
    "opt_model": "opt-125m_no_case",
    "mode_vec_feature":list(range(-9,-3)),
    "include_pred":False,
    "project_mode":"manual",
    "alpha_mode": "fix",
    "afni_smooth":True,
    "layer":8,
    "manual_path":"multilayer_4c9_bottom"
}

futil = FmriUtil(config)
coef_pm, coef_vm, pM_ll  = futil.fmri_regression(window=128)
futil.niplot(coef_pm)
print(np.mean(coef_pm))
save_data((coef_pm, coef_vm, pM_ll),"coef_mat_for_all_subs.data")
