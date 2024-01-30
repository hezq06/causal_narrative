source activate causal
datalad install -r ///labs/hasson/narratives
datalad get ./narratives/stimuli/gentle/*
datalad get ./narratives/derivatives/afni-smooth/sub-*/func/sub-*-fsaverage6_hemi-*clean.func.gii


