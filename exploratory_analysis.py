import os
import wfdb
import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

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

    records_category = []    
    for file in files:
        try:
            label = wfdb.rdrecord(os.path.join(folder_path, file)).comments
            record_info = {'file_name': file, 'label': label}
            records_category.append(record_info)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return records_category

def get_duration(folder_path):
    """
    Calculate the duration of each record in the specified folder_path.

    Parameters:
    - folder_path (str): The path to the folder containing WFDB records.

    Returns:
    - duration_list (list): A list of durations corresponding to each record.
    """
    records_duration = []
    files = os.listdir(folder_path)
    files = [f[:-4] for f in files if f.endswith(".hea")]

    for file in files:
        record_path = os.path.join(folder_path, file)
        try:
            record_data = wfdb.rdsamp(record_path)
            sample_point = record_data[0]
            sampling_frequency = record_data[1]['fs']
            duration = len(sample_point) / sampling_frequency
            record_info = {'file_name': file, 'duration': duration}
            records_duration.append(record_info)
        except Exception as e:
            print(f"Error processing {record_path}: {e}")

    return records_duration

def get_annotation(folder_path):

    """
    Extract annotations for record files in the specified folder_path.

    Parameters:
    - folder_path (str): The path to the folder containing WFDB annotation files.

    Returns:
    - annotations_category (dict): A dictionary mapping file names to their corresponding annotations and their summary.
    """

    records_annotations = []
    annotations_summary = []
    file_names = []
    files = os.listdir(folder_path)
    files = [f[:-4] for f in files if f.endswith(".atr")]

    for file in files:
        file_names.append(file)
        record_path = os.path.join(folder_path, file)
        try:
            record_annotation = wfdb.rdann(record_path, "atr")

            # Extract annotation information
            annotation_symbols = record_annotation.symbol
            annotation_sample = record_annotation.sample            

            # Example: Create a summary using Counter
            summary = Counter(annotation_symbols)

            # Append relevant information to the lists
            records_annotations.append({
                'symbols': annotation_symbols,
                'sample': annotation_sample                
            })

            annotations_summary.append(summary)
        except Exception as e:
            print(f"Error processing {record_path}: {e}")

    # Pair each file with its corresponding annotations and summary
    annotations_category = {file_name: {'annotations': ann, 'summary': summ}
                            for file_name, ann, summ in zip(file_names, records_annotations, annotations_summary)}

    return annotations_category
    
def create_records_dataframe(folder_path):
    # Load data into dictionaries
    categorized_records = categorize_records(folder_path)
    duration_data = get_duration(folder_path)
    annotation_data = get_annotation(folder_path)
    # Create a DataFrame
    df = pd.DataFrame({'folder_name':os.path.basename(folder_path),
                        'file_name': [record['file_name'] for record in categorized_records],
                        'labels': [record['label'][0] for record in categorized_records],
                        'duration(s)': [record['duration'] for record in duration_data],
                        'annotation': [annotation_data[file]['annotations'] for file in [record['file_name'] for record in categorized_records]],
                        'summary_annotation': [annotation_data[file]['summary'] for file in [record['file_name'] for record in categorized_records]]
                    })
    return df

def visualize_dataset(df):
    plt.hist(df['duration(s)'], bins=10, color='blue', alpha=0.7)
    plt.title('Distribution of Record Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.show()


    plt.figure(figsize=(10, 6))
    sns.countplot(x='labels', data=df, palette='viridis')
    plt.title('Distribution of Labels of ')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.show()   
    return 

TSOne_df=create_records_dataframe("D:/RA/Project/Database/CPSC2021/Training_set_I")
#TSOne_df.to_excel("CPSC2021_TS1.xlsx",index=False)
visualize_dataset(TSOne_df)

TSTwo_df=create_records_dataframe("D:/RA/Project/Database/CPSC2021/Training_set_II")
#TSOne_df.to_excel("CPSC2021_TS1.xlsx",index=False)
visualize_dataset(TSTwo_df) 

