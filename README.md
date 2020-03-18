# SparseRecovery

This is a project for sparse recovery problem. The codes contains data generation, recovery algorithms and other utilities. As long as it is not 
for collabration at the very beginning, The code is pretty lack of readability, which is going to be improved in the future. Basically, the only file you should run now is SporadicTraffic.ipynb, which is a playground especially convenient for the training and testing of neural networks. There are now instructions about how to run the code in that file. There are some trash at the back of the file and never mind, they are just for some quick check. The utils.py file contains the function to generate physical wireless channel, and do the 
recovery work by GMMV-AMP algorithm (Thank greatly to Mr. Ke for kindly providing me the MATLAB code for these). They are well packed together. **Please make sure to raise an issue if you have any question about 
the codes.**

You should at least have following libraries installed in your python enviorment:
- pytorch
- numpy
- matplotlib
- scipy
- tensorboardX
- tqdm

Please contact me if you are interest in MIMO or sparse recovery, I will be more than glad to discuss with you.
