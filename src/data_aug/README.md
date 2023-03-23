# Data Augmentation 

The above files contains the programs for data augmentation using resampling, and manipulation and summarization of augmented data. For more information about GECA data augmentation, refer to https://github.com/yangs12/GECA-modified.git.

## Usage
1. resample_from_data.py: This is the main file that generates resampled outputs from the previously trained base model in model_path. The maximum length for each sentence depends on the reference sentence from the original dataset. The generation method can be easily modified in the code.
```
python resample_from_data.py 
```

2. compare_text.py: This program compares text and count for repeated sentences in the augmented file vs the original training file.
```
python compare_text.py 
```

3. data_summary.py: This program counts the total number of words in each of the files in a list of folders (test_dir), and output the total number of words in each folder.
```
python data_summary.py 
```

4. geca_data_ext.py: The program takes files from from_dir and extract the same number of words required by the experiments (default as 12k)
```
python geca_data_ext.py 
```

5. real_data_ext.py: The program takes files from from_dir and extract the same number of words required by the custom count_dir

```
python real_data_ext.py 
```
