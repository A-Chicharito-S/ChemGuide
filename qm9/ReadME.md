# Inference with GeoLDM+xtb
the necessary packages for running the code are specified in ``requirements.txt`` of the GeoLDMwithXTB file

The checkpoints (too big to zip) are provided by the authors of GeoLDM and need downloading, the specific instructions are shown in the repo of GeoLDM

the code command (exemplary) to run GeoLDM with xtb is: (clf_scale is the scaling factor)

``python eval_sample_xtb.py --grad_clip_threshold=1. --clf_scale=0.0001 --guidance_steps=800 --num_samples=300``

The calling stack is:

eval_sample_xtb.py ---> qm9/sampling_xtb.py ---> equivariant_diffusion/cond_en_diffusion_xtb.py ---> (calls the ``cond_sample`` function of the ``EnLatentDiffusion`` model)