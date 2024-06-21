# frb_baryon_connor2024

This repository hosts code and data associated with the publication 
Connor et al. (2024). It is a reproduction package of the 
analysis and figures in that work. 

## installation instructions

You can install in a virtual environment or with Poetry, which is a dependency management and packaging tool for Python. I usually use Poetry.

python3 -m venv baryon_env
source baryon_env/bin/activate
pip install -r requirements.txt

Or if you want to use Poetry,

pip install poetry

alternatively, 

curl -sSL https://install.python-poetry.org | python3 -

poetry install
poetry shell

## src/frbdm_mcmc_jit.py
This program has code for JAX-compiled MCMC fitting code using emcee. It's still terribly slow due to computing a 2D integral for each new parameter, so if anybody wants to submit a PR to speed things up please do! That said, I never got it running on GPU, so maybe it just requires accelerated hardware.

Example usage:

To run on all FRBs in the dataset with no selection criteria

python frbdm_mcmc_jit.py 

To run only on DSA-110 discovered FRBs beyond z=0.0125 for 2000 MCMC steps

python frbdm_mcmc_jit.py --zmin 0.0125 --nmcmc 2000 --tel dsa-110

To run on all FRBs between 0.25 < z < 0.50 excluding, say, FRB20200430A

python frbdm_mcmc_jit.py --zmin 0.25 --zmax 0.50 --nmcmc 2000 --tel all --exclude FRB20200430A

