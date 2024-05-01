import torch
from torch.optim import Adam, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import json
# import evaluate
import utils.settings as settings
import logging
logger = logging.getLogger('custom_trainer')
logger.setLevel(logging.INFO)


def train(model, 
          train_dataloader, 
          valid_dataloader, 
          train_metric=None, 
          valid_metric=None, 
          device=None, 
          args=None, 
          writer=None):
    """ Customer trainer for the LangaugeDetectionModel """
    
    experiment = args["experiment"]
    logger.addHandler(settings.get_logger_handler(experiment))
    CHECKPOINT_PATH = f"output/models/experiment-{experiment}"
    last_epoch = args["starting_epoch"] - 1

    if args["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=args["lr"])
    elif args["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    else:
        raise NotImplementedError

    num_training_steps = (args["num_epochs"] - args["starting_epoch"]) * len(train_dataloader)

    if last_epoch > -1:
        checkpoint = torch.load(f"{CHECKPOINT_PATH}/xlm-roberta-base-language-detection-epoch-{str(last_epoch)}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded Checkpoint: {CHECKPOINT_PATH}/xlm-roberta-base-language-detection-epoch-{str(last_epoch)}.pth")

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_training_steps,
        last_epoch = last_epoch
    )

    progress_bar_train = tqdm(range(num_training_steps))
    progress_bar_valid = tqdm(range((args["num_epochs"] - args["starting_epoch"]) * len(valid_dataloader)))
    
    best_model_epoch = 0
    best_valid_f1 = 0
    for epoch in range(args["starting_epoch"], args["num_epochs"]):
        # training loop
        train_running_loss = 0.0
        model.train()
        for idx, batch in enumerate(train_dataloader):
            if args["precision"] == "half":
                model.half()
            elif args["precision"] == "full":
                model.float()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            train_loss = outputs.loss
            train_loss.backward()
            model.float()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_running_loss += train_loss.item()

            train_logits = outputs.logits
            train_predictions = torch.argmax(train_logits, dim=-1)
            for key in train_metric:
                train_metric[key].add_batch(predictions = train_predictions, references = batch["labels"])

            logger.info(f"Epoch: {epoch}, Batch: {idx}, Train Loss: {train_running_loss/(idx+1)}")
            progress_bar_train.update(1)
            # break

        trainLoss = train_running_loss/len(train_dataloader)

        # compute train results of the current eppoch
        results = {"epoch": epoch,
                   "trainLoss": trainLoss,
                   "trainAcc": train_metric["acc"].compute()["accuracy"],
                   "trainF1": train_metric["f1"].compute(average=args["f1_metric_average"])["f1"]}

        logger.info(f"Results: {results}")

        writer.add_scalar("Loss/train", trainLoss, epoch)
        writer.add_scalar("Accuracy/train", results["trainAcc"], epoch)
        writer.add_scalar("F1/train", results["trainF1"], epoch)

        # validation loop
        model.eval()
        valid_running_loss = 0.0
        for idx, batch in enumerate(valid_dataloader):
            if args["precision"] == "half":
                model.half()
            elif args["precision"] == "full":
                model.float()
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            valid_loss = outputs.loss
            valid_running_loss += valid_loss.item()
            valid_logits = outputs.logits
            valid_predictions = torch.argmax(valid_logits, dim=-1)
            for key in valid_metric:
                valid_metric[key].add_batch(predictions = valid_predictions, references = batch["labels"])
            logger.info(f"Epoch: {epoch}, Batch: {idx}, Valid Loss: {valid_running_loss/(idx+1)}")
            progress_bar_valid.update(1)
            # break
        validLoss = valid_running_loss/len(valid_dataloader)

        # Compute valid results of the current epoch
        results["validLoss"] = validLoss
        results["validAcc"] = valid_metric["acc"].compute()["accuracy"]
        results["validF1"] = valid_metric["f1"].compute(average=args["f1_metric_average"])["f1"]

        # update the best model epoch
        if best_valid_f1 < results["validF1"]:
            best_valid_f1 = results["validF1"]
            best_model_epoch = epoch

        results["best_valid_f1"] = best_valid_f1
        results["best_model_epoch"] = best_model_epoch
        results["experimental_settings"] = args

        logger.info(f"Results: {results}")

        writer.add_scalar("Loss/valid", validLoss, epoch)
        writer.add_scalar("Accuracy/valid", results["validAcc"], epoch)
        writer.add_scalar("F1/valid", results["validF1"], epoch)

        # Save Results of the current epoch
        with open(f"output/results/experiment-{str(experiment)}/results_train_valid.jsonl", "a") as f:
            json.dump(results, f)
            f.write("\n")
        if epoch in [0, best_model_epoch]:
            # Save Model
            torch.save({
                "epoch": epoch,
                "model_str": args["model_str"],
                "max_seq_len": args["max_seq_len"],
                "embedding_dim": args["embedding_dim"],
                "id2label": args["id2label"],
                "label2id": args["label2id"],
                "num_labels": args["num_labels"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "trainLoss": trainLoss,
                }, f"{CHECKPOINT_PATH}/xlm-roberta-base-language-detection-best-model.pth")
        
    
    # Save best model epoch
    results = {"best_model_epoch": best_model_epoch}
    with open(f"output/results/experiment-{str(experiment)}/results_train_valid.jsonl", "a") as f:
        json.dump(results, f)
        f.write("\n")

    return results