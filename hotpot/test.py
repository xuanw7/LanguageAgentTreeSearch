import pickle 
import numpy as np
import re


with open('accuracy.txt','rb') as f:
    test_accs = pickle.load(f)
    print(test_accs)

with open('terminate_count.txt','rb') as f:
    test_terminate= pickle.load(f)
    print(test_terminate)