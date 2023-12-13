import os
import wfdb
import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path
from collections import Counter
import itertools

def categorize_records(folder_path):
    """ 
    Cateogorize record files with their lables.

    """

    lables=[]
    file_names=[]
    files=os.listdir(folder_path)
    files=[f[:-4] for f in files if f.endswith(".hea")]
    for file in files:        
        file_names.append(file)
        lable=wfdb.rdrecord(os.path.join(folder_path,file)).comments
        lables.append(lable)
    records_category=dict(zip(file_names,lables))
    return records_category

test_categorize_records=categorize_records("D:/RA/Project/Database/CPSC2021/Training_set_II")
print(len(test_categorize_records))

Counter(itertools.chain.from_iterable(test_categorize_records.values()))

print(test_categorize_records)