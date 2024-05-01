from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import pandas as pd
import os
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import Dataset as torchDS
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam, AdamW
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import get_scheduler, get_linear_schedule_with_warmup
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from model import LanguageDetectionModel
from tqdm.auto import tqdm
import json
import evaluate
from torch.utils.tensorboard import SummaryWriter
import yaml
import logging
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
from utils import settings, custom_trainer, custom_eval
  
def main(args):
    experiment = args["experiment"]

    #create directories for logs, results, and models
    os.makedirs(f"output/logs/experiment-{str(experiment)}/", exist_ok=True)
    os.makedirs(f"output/results/experiment-{str(experiment)}", exist_ok=True)
    os.makedirs(f"output/models/experiment-{str(experiment)}", exist_ok=True)

    # set custom handler to logger
    logger.addHandler(settings.get_logger_handler(experiment))
    
    # load Label Info
    with open(args["preprocessed_datadir"] + "/label_info.json", "r") as f:
        label_info = json.load(f)
        num_labels = len(label_info["label2id"])
        
    # update args with label info
    args["id2label"] = label_info["id2label"]
    args["label2id"] = label_info["label2id"]
    args["num_labels"] = args["num_labels"]

    logger.info(f"Experimental Settings")
    logger.info(f"{args}")
    
    # load train, valid, and test dataset
    if args["train_valid_step"] and args["test_step"]:
        train_dataset = pd.read_csv(args["preprocessed_datadir"] + "/train_dataset.csv")
        valid_dataset = pd.read_csv(args["preprocessed_datadir"] + "/valid_dataset.csv")
        test_dataset = pd.read_csv(args["preprocessed_datadir"] + "/test_dataset.csv")
    
    # Load test dataset
    if not args["train_valid_step"] and args["test_step"]:
        test_dataset = pd.read_csv(args["preprocessed_datadir"] + "/test_dataset.csv")

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["model_str"])
    tokenizer.model_max_len = args["max_seq_len"]

    # create a HuggingFace Dataset Dictionary containing train, valid, and test datasets
    if args["train_valid_step"] and args["test_step"]:
        dataset_hf = DatasetDict({"train": HFDataset.from_pandas(train_dataset),
                                  "valid": HFDataset.from_pandas(valid_dataset),
                                  "test": HFDataset.from_pandas(test_dataset)})

    if not args["train_valid_step"] and args["test_step"]:
        dataset_hf = DatasetDict({"test": HFDataset.from_pandas(test_dataset)})

    # tokenize the dataset
    tokenized_dataset = dataset_hf.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=args["max_seq_len"]), batched=True)

    # set the dataset format to make it compatible with pytorch
    tokenized_dataset.set_format("torch", columns = ["input_ids", "attention_mask", "label"])

    # create a data collator for batch-wise processing of the input data
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    # create pytorch dataloader for train and valid sets
    if args["train_valid_step"] and args["test_step"]:
        train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=args["batch_size"], shuffle=True, collate_fn=data_collator)
        valid_dataloader = DataLoader(tokenized_dataset["valid"], batch_size=args["batch_size"]*2, shuffle=False, collate_fn=data_collator)

    # create pytorch dataloader for test set
    if args["test_step"]:
        test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args["batch_size"]*2, shuffle=False, collate_fn=data_collator)

    
    # load the model
    model = LanguageDetectionModel(args["model_str"], embedding_dim = args["embedding_dim"], num_labels=num_labels, dropout_prob = args["dropout_prob"])

    # load the model into gpu or cpu (as per availability)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # load relevant metrics
    acc_metric, f1_metric = evaluate.load("accuracy", trust_remote_code=True), evaluate.load("f1", trust_remote_code=True)
    
    # Training and Validation Step
    if args["train_valid_step"]:
        writer = SummaryWriter(f"output/logs/experiment-{str(experiment)}-tensorboard")
        train_results = custom_trainer.train(model, train_dataloader, valid_dataloader, train_metric={"acc": acc_metric, "f1": f1_metric}, valid_metric={"acc": acc_metric, "f1": f1_metric}, device=device, args=args, writer=writer)
        writer.close()
        
    # Post-Training Evaluation Step
    if args["test_step"]:
        custom_eval.test(model, test_dataloader, metric={"acc": acc_metric, "f1": f1_metric}, device=device, args=args)
    
if __name__ == "__main__":
    with open('./config.yml', 'r') as file:
        args = yaml.safe_load(file)["args"]["config-5"]
        args["lr"] = float(args["lr"])
        args["weight_decay"] = float(args["weight_decay"])
    main(args)
   
    