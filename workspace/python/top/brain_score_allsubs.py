from causal_narrative.fmriutil import *
from causal_narrative.datautil import *
import argparse

torch.cuda.set_device("cuda:1")

parser = argparse.ArgumentParser()

# Add arguments
#parser.add_argument("--layer", required=True, help="Layer ID")
#args = parser.parse_args()
#layer_id = args.layer

current_path = os.getcwd()
narrative_home = os.path.join(current_path, "../../../dataset/narratives")
opts_home = os.path.join(current_path, "../../../dataset/opts")
config={
    "narrative_home": narrative_home,
    "data_home": opts_home,
    "opt_model": "opt-125m_no_case",
    "mode_vec_feature":list(range(-9,-3)),
    "include_pred":False,
    "project_mode":"manual",
    "alpha_mode": "nest",
    "afni_smooth":True,
    "layer":8,
    "manual_path":"multilayer_4c9_opt_top"
}

futil = FmriUtil(config)
coef_pm, coef_vm, pM_ll  = futil.fmri_regression(window=128)
futil.niplot(coef_pm)
print(np.mean(coef_pm))
save_data((coef_pm, coef_vm, pM_ll),"coef_mat_for_all_subs.data")
