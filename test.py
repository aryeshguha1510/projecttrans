from datasets import load_dataset
import numpy as np
import pandas as pd
ds = load_dataset('csebuetnlp/BanglaNMT', "BanglaNMT")

train_data=ds['train']
val_data=ds['validation']
test_data=ds['test'] 
print(train_data)

