# Python example code for the 4th China Physiological Signal Challenge 2021

## What's in this repository?

We implemented a threshold-based classifier that uses the coefficient of sample entropy (cosEn) of the ECG lead signals as features. This simple example illustrates how to format your Python entry for the Challenge. However, it is not designed to score well (or, more accurately, designed not to do well), so you should not use it as a baseline for your model's performance.

The code uses two main scripts, as described below, to run and test your algorithm for the 2021 Challenge.

## How do I run these scripts?

You can run this baseline method by installing the requirements

    pip install requirements.txt

and running 

    python entry_2021.py <data_path> <result_save_path>

where <data_path> is the folder path of the test set, <result_save_path> is the folder path of your detection results. 

## How do I run my code and save my results?

Please edit entry_2021.py to implement your algorithm. You should save the results as ‘.json’ files by record. The format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]] }. The name of the result file should be the same as the corresponding record file.

After obtaining the test results, you can evaluate the scores of your method by running

    python score_2021.py <ans_path> <result_save_path>

where <ans_path> is the folder save the answers, which is the same path as <data_path> while the data and annotations are stored with 'wfdb' format. <result_save_path> is the folder path of your detection results.

## Useful links

- [MATLAB example code for The China Physiological Signal Challenge (CPSC2021)](https://github.com/CPSC-Committee/cpsc2021-matlab-entry)