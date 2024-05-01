import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer, DataCollatorWithPadding
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset as HFDataset
from datasets import DatasetDict
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from datasets.utils.logging import disable_progress_bar

class LanguageDetectionModel(nn.Module):
    def __init__(self, model_str, embedding_dim = 768, num_labels=20, dropout_prob = 0.5):
        super(LanguageDetectionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.model = AutoModel.from_pretrained(model_str, 
                                               config = AutoConfig.from_pretrained(
                                                            model_str, 
                                                            output_attention = True,
                                                            output_hidden_state = True))
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(embedding_dim, num_labels)
        
        
    def forward(self, input_ids = None, attention_mask = None, labels = None):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        
        last_hidden_state = outputs[0]
        
        sequence_outputs = self.dropout(last_hidden_state)
        
        logits = self.classifier(sequence_outputs[:, 0, :].view(-1, self.embedding_dim))
        
        loss = None
        
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(loss = loss, logits = logits, hidden_states = outputs.hidden_states, attentions = outputs.attentions)

class LanguageDetectionModelInference():
    def __init__(self, checkpoint_path: str, precision: str="full"):
        checkpoint = torch.load(f"{checkpoint_path}")
        self.max_seq_len = checkpoint["max_seq_len"]
        self.id2label = checkpoint["id2label"]
        self.label2id = checkpoint["label2id"]

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_str"])
        self.tokenizer.model_max_len = checkpoint["max_seq_len"]
        
        self.data_collator = DataCollatorWithPadding(tokenizer = self.tokenizer)
        
        self.model = LanguageDetectionModel(checkpoint["model_str"], 
                                            embedding_dim=checkpoint["embedding_dim"],
                                            num_labels=checkpoint["num_labels"])
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if precision == "half":
            self.model.half()
        
        try:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        except:
            self.device = "cpu"
            
        self.model.to(self.device)
        
    def predict(self, documents: list=None, batch_size: int=16, prob: bool=False):
        self.model.eval()
        disable_progress_bar()
        dataset_hf = DatasetDict({"test": HFDataset.from_pandas(pd.DataFrame(documents, columns=["text"]))})
        
        if len(documents) >= 1:
            tokenized_dataset = dataset_hf.map(lambda batch: self.tokenizer(batch["text"], truncation=True, max_length=self.max_seq_len), batched=True)
        else:
            raise ValueError('Empty Input!! Input "documents" list should have at least one text document')
            
        tokenized_dataset.set_format("torch", columns = ["input_ids", "attention_mask"])

        if len(documents) < batch_size:
            batch_size = len(documents)
            
        dataloader = DataLoader(
                tokenized_dataset['test'], batch_size = batch_size, collate_fn = self.data_collator
        )
        
        pred_labels = []
        prob_list = []
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                
            logits = outputs.logits
            if prob:
                prob_list += torch.nn.functional.softmax(logits, dim=1).tolist() 
            pred_labels += [self.id2label[str(pred_id)] for pred_id in torch.argmax(logits, dim = -1).tolist()]
        
        if prob:
            label_prob_scores = []
            for row in prob_list:
                label_prob_scores.append({self.id2label[str(idx)]:prob for idx, prob in enumerate(row)})
            return pred_labels, label_prob_scores
        
        return pred_labels

    def get_pred_label_ids(self, pred_labels: list):
        return [int(self.label2id[label]) for label in pred_labels]

    def get_classification_report(self, y_true: list, documents: list=None, output_dict=False):
        pred_labels = self.predict(documents)
        y_pred = self.get_pred_label_ids(pred_labels)
        target_names = list(self.id2label.values())
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=output_dict)
        return report
        
        

