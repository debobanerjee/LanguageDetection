from sklearn.metrics import classification_report
from model import LanguageDetectionModelInference
import yaml
import pandas as pd
import json
import os

def main(args):
    test_dataset = pd.read_csv(args["preprocessed_datadir"] + "/test_dataset.csv")
    y_true, documents = list(map(int, test_dataset["label"].values.tolist())), test_dataset["text"].values.tolist()

    # set the checkpoint path
    checkpoint_path = "output/models/experiment-4/xlm-roberta-base-language-detection-best-model.pth"
    
    # create an instance of the language detection model inference
    model = LanguageDetectionModelInference(checkpoint_path)

    # generate classification report
    report = model.get_classification_report(y_true, documents=documents, 
                                             disgits=args["classfication_report"]["digits"],  
                                             output_dict=args["classification_report"]["output_dict"])
    
    os.makedirs(args["results_dir"], exist_ok=True)
    if args["classification_report"]["output_dict"]:
        with open(args["results_dir"]+"/classification_report.json", "w") as f:
            json.dump(report, f)
    else:
        print(report)
    
if __name__ == "__main__":
    with open("./config.yml", "r") as f:
        args = yaml.safe_load(f)["args"]["config-4"]
    main(args)

