from datasets import load_dataset, Dataset, concatenate_datasets

def _load_dataset(data_path):
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):                                
                data = load_dataset("json", data_files=data_path)
        else:
                data = load_dataset(data_path)        
        return data

def load_merged_dataset(dataset_paths, val_set_size, verbose=False, data_set_formatter = None):
        datasets = []
        for dataset_path in dataset_paths:
                ds = _load_dataset(dataset_path)
                ds = ds["train"].shuffle()
                if data_set_formatter:
                        ds = data_set_formatter(dataset_path, ds)

                if verbose:
                        print(f'ds {dataset_path}: {ds}')

                datasets.append(ds)

        train_val = concatenate_datasets(datasets)
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

def load_dolly_and_agent(dataset_paths, select_len = 10, val_set_size = 2, verbose=False):
        def formatter(dataset_path, dataset):
                if "databricks-dolly-15k-ja" in dataset_path:                        
                        dataset = dataset.shuffle().select(range(select_len))
                return dataset
        
        train, val = load_merged_dataset(dataset_paths, val_set_size= val_set_size, verbose=verbose, data_set_formatter=formatter)
        return train, val

def main():        
        train, val = load_dolly_and_agent(
                ['./dataset/databricks-dolly-15k-ja.json', './dataset/agent_dataset.json'], 
                verbose=True)                                         
        
        #for v in train:
        #        print(v)
if __name__ == '__main__':
        main()