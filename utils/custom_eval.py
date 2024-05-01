import torch
from tqdm.auto import tqdm
import json
import utils.settings as settings
import logging
logger = logging.getLogger('custom_eval')
logger.setLevel(logging.INFO)

def test(model, 
         test_dataloader, 
         metric=None, 
         device=None, 
         args=None):

    """ Custom Evaluator for LanguageDetectionModel """
    
    experiment = args["experiment"]
    logger.addHandler(settings.get_logger_handler(experiment))
    
    # load model weights from checkpoint
    CHECKPOINT_PATH = f"output/models/experiment-{experiment}"
    checkpoint = torch.load(f"{CHECKPOINT_PATH}/xlm-roberta-base-language-detection-best-model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded Checkpoint: {CHECKPOINT_PATH}/xlm-roberta-base-language-detection-epoch-best-model.pth")

    # Compute Test Results
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

    # Compute test results
    results = {"testAcc": metric["acc"].compute()["accuracy"],
                "testF1": metric["f1"].compute(average=args['f1_metric_average'])["f1"],
                "experimental_settings": args}
   
    # Log Test Results
    logger.info(f"\nTest Results: {results}\n")

    # Save Test Results
    with open(f"output/results/experiment-{experiment}/results_test.jsonl", "a") as f:
        json.dump(results, f)
        f.write("\n")