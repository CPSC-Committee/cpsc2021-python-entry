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
    Categorize record files with their labels.

    Parameters:
    - folder_path (str): The path to the folder containing WFDB records.

    Returns:
    - records_category (dict): A dictionary mapping file names to their corresponding labels.
    """
    labels = []
    file_names = []
    files = os.listdir(folder_path)
    files = [f[:-4] for f in files if f.endswith(".hea")]
    
    for file in files:        
        file_names.append(file)
        try:
            label = wfdb.rdrecord(os.path.join(folder_path, file)).comments
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    records_category = dict(zip(file_names, labels))
    return records_category

def get_duration(folder_path):
    """
    Calculate the duration of each record in the specified folder_path.

    Parameters:
    - folder_path (str): The path to the folder containing WFDB records.

    Returns:
    - duration_list (list): A list of durations corresponding to each record.
    """
    duration_list=[]
    files=os.listdir(folder_path)
    files=[f[:-4] for f in files if f.endswith(".hea")]

    for file in files:
        record_path=os.path.join(folder_path,file)
        try:
            record_data=wfdb.rdsamp(record_path)
            sample_point=record_data[0][0]
            sampling_frequency=record_data[1]['fs']
            duration=sample_point/sampling_frequency
            duration_list.append(duration)
        except Exception as e:
            print(f'Error processing {record_path}:{e}')
    return duration_list

test_categorize_records=categorize_records("D:/RA/Project/Database/CPSC2021/Training_set_II")
print(len(test_categorize_records))

Counter(itertools.chain.from_iterable(test_categorize_records.values()))

print(test_categorize_records)
get_duration("D:/RA/Project/Database/CPSC2021/Training_set_II")
get_duration("D:/RA/Project/Database/CPSC2021/Training_set_I")