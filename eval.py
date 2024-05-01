from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import pandas as pd
import os
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_metric
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
import logging
logger = logging.getLogger(__name__)
from torch.utils.tensorboard import SummaryWriter
import yaml
import evaluate

def test(model, best_model_epoch, test_dataloader, metric=None, device=None, args=None):
    experiment = args["experiment"]
    CHECKPOINT_PATH = f"output/models/experiment-{experiment}"
    
    checkpoint = torch.load(f"{CHECKPOINT_PATH}/xlm-roberta-base-language-detection-epoch-{str(best_model_epoch)}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded Checkpoint: {CHECKPOINT_PATH}/xlm-roberta-base-language-detection-epoch-{str(best_model_epoch)}.pth")
    
    model.eval()
    for batch in test_dataloader:
        if args["precision"] == "half":
            model.half()
        elif args["precision"] == "full":
            model.float()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        for key in metric:
            metric[key].add_batch(predictions = predictions, references = batch["labels"])
        # break
    # Compute test results
    results = {"testAcc": metric["acc"].compute()["accuracy"],
                "testF1": metric["f1"].compute(average=args['f1_metric_average'])["f1"],
                "experimental_settings": args}
   
    # Log Test Results
    logger.info(f"\nTest Results: {results}\n")

    with open(f"output/results/experiment-{experiment}/results_test.jsonl", "a") as f:
        json.dump(results, f)
        f.write("\n")
              
def main(args):
    experiment = args["experiment"]
    os.makedirs(f"output/logs/experiment-{str(experiment)}/", exist_ok=True)
    os.makedirs(f"output/results/experiment-{str(experiment)}", exist_ok=True)
    os.makedirs(f"output/models/experiment-{str(experiment)}", exist_ok=True)
                
    logging.basicConfig(filename=f"output/logs/experiment-{str(experiment)}/language_detection.log", level=logging.INFO)
    logger.info(f"Experimental Settings")
    logger.info(f"{args}")
    # load Label Info
    with open(args["preprocessed_datadir"] + "/label_info.json", "r") as f:
        label_info = json.load(f)
        num_labels = len(label_info["label2id"])
    
    # Load test dataset
    if not args["train_valid_step"] and args["test_step"]:
        test_dataset = pd.read_csv(args["preprocessed_datadir"] + "/test_dataset.csv")

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["model_ckpt"])
    tokenizer.model_max_len = args["max_seq_len"]

    if not args["train_valid_step"] and args["test_step"]:
        dataset_hf = DatasetDict({"test": HFDataset.from_pandas(test_dataset)})

    # tokenize the dataset
    tokenized_dataset = dataset_hf.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=args["max_seq_len"]), batched=True)

    # set the dataset format to make it compatible with pytorch
    tokenized_dataset.set_format("torch", columns = ["input_ids", "attention_mask", "label"])

    # create a data collator for batch-wise processing of the input data
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    # create pytorch dataloader for test set
    if args["test_step"]:
        test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args["batch_size"]*2, shuffle=False, collate_fn=data_collator)

    # load the model
    model = LanguageDetectionModel(args["model_ckpt"], embedding_dim = args["embedding_dim"], num_labels=num_labels, dropout_prob = args["dropout_prob"])

    # load the model into gpu or cpu (as per availability)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # load relevant metrics
    acc_metric, f1_metric = load_metric("accuracy", trust_remote_code=True), load_metric("f1", trust_remote_code=True)
    
    # Training and Validation Step
    if args["train_valid_step"]:
        writer = SummaryWriter(f"output/logs/experiment-{str(experiment)}-tensorboard")
        train_results = train(model, train_dataloader, valid_dataloader, train_metric={"acc": acc_metric, "f1": f1_metric}, valid_metric={"acc": acc_metric, "f1": f1_metric}, device=device, args=args, writer=writer)
        writer.close()
        
    # Post-Training Evaluation Step
    if args["test_step"]:
        with open(f"output/results/experiment-{experiment}/results_train_valid.jsonl") as f:
            results = [json.loads(row.rstrip("\n")) for row in f.readlines()]
        best_model_epoch = results[-1]["best_model_epoch"]
        
        test(model, best_model_epoch, test_dataloader, metric={"acc": acc_metric, "f1": f1_metric}, device=device, args=args)
    
if __name__ == "__main__":
    with open('./config.yml', 'r') as file:
        args = yaml.safe_load(file)["args"]["config-eval-4"]
        args["lr"] = float(args["lr"])
        args["weight_decay"] = float(args["weight_decay"])
    main(args)
    