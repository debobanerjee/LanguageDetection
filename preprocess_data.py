import pandas as pd
import argparse
import os
import json
from huggingface_hub import hf_hub_download
from torch.utils.tensorboard import SummaryWriter
import yaml
import logging
logger = logging.getLogger('preprocess_data')
logger.setLevel(logging.INFO)
from utils import settings

def custom_load_dataset(split="train", args=None):
    REPO_ID = args["dataset_repoid"]
    FILENAME = f"{split}.csv"

    if not os.path.exists(f"{args['datadir']}/{split}.csv"):
        df = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type=args["dataset"]))
        df.to_csv(f"{args['datadir']}/{split}.csv", index=False)
    else:
        df = pd.read_csv(f"{args['datadir']}/{split}.csv")

    return df
    
def get_label_info(dataset, args=None):
    labels = list(set(dataset[args["label_column"]].to_list()))
    label2id = {label:index for label, index in zip(labels, list(range(len(labels))))}
    id2label = {label2id[label]:label for label in label2id}
    
    return label2id, id2label

def save_label_info(label2id, id2label, args=None):
    os.makedirs(args["preprocessed_datadir"], exist_ok=True)
    with open(args["preprocessed_datadir"] + "/label_info.json", "w") as f:
        json.dump({"label2id": label2id, 
                   "id2label": id2label}, f)
    
        
def add_label_id_to_dataset(dataset, label2id, args=None):
    dataset["label"] = dataset[args["label_column"]].apply(lambda x: label2id[x])
    return dataset[["text", "label"]]
    
def main(args):
    logger.addHandler(settings.get_logger_handler_data_preprocessing())
    datadir = args["datadir"]
    train_dataset = custom_load_dataset(split="train", args=args)
    valid_dataset = custom_load_dataset(split="valid", args=args)
    test_dataset = custom_load_dataset(split="test", args=args)

    dataset = pd.concat([train_dataset, valid_dataset, test_dataset])
    label2id, id2label = get_label_info(dataset, args=args)
    save_label_info(label2id, id2label, args=args)

    train_dataset = add_label_id_to_dataset(train_dataset, label2id, args)
    valid_dataset = add_label_id_to_dataset(valid_dataset, label2id, args)
    test_dataset = add_label_id_to_dataset(test_dataset, label2id, args)

    train_dataset.to_csv(args["preprocessed_datadir"] + "/train_dataset.csv", index=False)
    valid_dataset.to_csv(args["preprocessed_datadir"] + "/valid_dataset.csv", index=False)
    test_dataset.to_csv(args["preprocessed_datadir"] + "/test_dataset.csv", index=False)

if __name__ == "__main__":
    with open('./config.yml', 'r') as file:
        args = yaml.safe_load(file)["args"]["preprocessing"]
    main(args)




