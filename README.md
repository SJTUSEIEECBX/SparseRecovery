# SparseRecovery

This is a project for sparse recovery problem. The codes contains data generation, recovery algorithms and other utilities. As long as it is not 
for collabration at the very beginning, The code is super lack of readability, which is going to be improved in the future. A few instruction 
might be helpful to conprehend the codes now. The utils.py file contains the function to generate physical wireless channel, and do the 
recovery work by an AMP algorithm. They are well packed together. The SporadicTraffic.ipynb file is the main experiment playground, where 
you can find the usages for most of the useful component in this repository. **Please make sure to raise an issue if you have any question about 
the codes.**

You should at least have following libraries installed in your python enviorment:
- pytorch
- numpy
- matplotlib
- scipy
- tensorboardX
- tqdm
