# baryon_paper

This repository hosts code and data associated with the publication 
Connor et al. (2024). It is a reproduction package of the 
analysis and figures in that work.

## src/frbdm_mcmc_jit.py
This program has code for JAX-compiled MCMC fitting code using emcee. It's still terribly slow due to computing a 2D integral for each new parameter, so if anybody wants to submit a PR to speed things up please do! That said, I never got it running on GPU, so maybe it just requires accelerated hardware.