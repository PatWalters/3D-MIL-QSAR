#!/usr/bin/env  python

import pandas as pd

def generate_splits(infile_name):
    df = pd.read_csv(infile_name)
    dataset_list = sorted(df.Dataset.unique())
    split_dict = {}
    for dataset in dataset_list:
        split_dict[dataset] = {}
        tmp_df = df.query(f"Dataset == '{dataset}'").copy()
        tmp_df.reset_index(drop=True,inplace=True)
        # process random splits
        rnd_list = []
        for i in range(0,10):
            col = f"RND_{i:02d}"
            train_list = list(tmp_df.query(f"{col} == 'train'").index)
            test_list = list(tmp_df.query(f"{col} == 'test'").index)
            rnd_list.append([train_list, test_list])
        split_dict[dataset]['RND'] = rnd_list
        # process scaffold splits
        scaf_list = []
        for i in range(0,10):
            col = f"SCAF_{i:02d}"
            train_list = list(tmp_df.query(f"{col} == 'train'").index)
            test_list = list(tmp_df.query(f"{col} == 'test'").index)
            scaf_list.append([train_list, test_list])
        split_dict[dataset]['SCAF'] = scaf_list
    return split_dict


if __name__ == "__main__":
    generate_splits("cv_splits.csv")

    

    
        
        
    
    


