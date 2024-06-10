import os

import numpy as np

f = open('./namelist.txt','r')

for name in f:
    print("Excluding %s" % name)
    os.system('python frbdm_mcmc.py --zmin 0.0125 --nmcmc 750 --exclude %s' % name)
    print("\n\n\n")
