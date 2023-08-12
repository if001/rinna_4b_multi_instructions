from datasets import load_dataset, Dataset, concatenate_datasets

def _load_dataset(data_path):
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):                                
                data = load_dataset("json", data_files=data_path)
        else:
                data = load_dataset(data_path)        
        return data


def load_merged_dataset(ds1, ds2, val_set_size, ds1_len=0, verbose=False):        
        data1 = _load_dataset(ds1)        
        if ds1_len != 0:
                data1 = data1["train"].shuffle().select(range(ds1_len))                

        ## remove column if need
        # data1 = data1.remove_columns([col for col in data1.column_names if col not in ['input', 'output', 'index', 'instruction']])
        if verbose:
                print('data set1: ', data1)
        data2 = _load_dataset(ds2)
        data2 = data2["train"].shuffle()
        if verbose:
                print('data set2: ', data2)

        train_val = concatenate_datasets([data1, data2])
        train_val = train_val.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle()
        if verbose:
                print('merged train: ', train_data)
        val_data = train_val["test"].shuffle()
        if verbose:
                print('merged val: ', val_data)

        return train_data, val_data

def main():
        train, val = load_merged_dataset(
                './dataset/databricks-dolly-15k-ja.json', 
                './dataset/agent_dataset.json', 
                1,
                3,
                verbose=True)
        
        #for v in train:
        #        print(v)
if __name__ == '__main__':
        main()