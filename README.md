# LanguageDetection

### Instal Required Packages
```
pip install -r requirements.txt
```

### Data Preprocessing
#### Set configuration for data preprocessing inside `config.yml` file
```yaml
args
    preprocessing:
        dataset_repoid: "papluca/language-identification"
        repo_type: "dataset"
        preprocessed_datadir: "data/preprocessed"
        datadir: "data"
        label_column: "labels"
```
#### Start Data Preprocessing
```
python preprocess_data.py
```

### Training 
#### Set training configuration inside `config.yml` file
```yaml
args
    config
        experiment: 1
        dataset_repoid: "papluca/language-identification"
        model_str: "FacebookAI/xlm-roberta-base"
        batch_size: 8
        num_epochs: 5
        starting_epoch: 0
        lr: 2e-6
        optimizer: "adam"
        embedding_dim: 768
        max_seq_len: 512
        dropout_prob: 0.1
        f1_metric_average: "weighted"
        weight_decay: 1e-5
        precision: "full"
        train_valid_step: True
        test_step: True
        num_labels: 20
        datadir: "data"
        preprocessed_datadir: "data/preprocessed"
        model_ckpt_dir: "output/models/experiment-1"
        results_dir: "output/results/experiment-1/"
        log_dir: "output/logs/experiment-1"
```

```
python main.py
```

### Getting Started with Model Inference
```python
from model import LanguageDetectionModelInference

# set the checkpoint path
checkpoint_path = "output/models/experiment-4/xlm-roberta-base-language-detection-epoch-4-updated-with-meta-info.pth"

# create an instance of the language detection model inference
model = LanguageDetectionModelInference(checkpoint_path)

# list of documents
documents = ["os chefes de defesa da estónia, letónia, lituânia, alemanha, itália, espanha e eslováquia assinarão o acordo para fornecer pessoal e financiamento para o centro.", 
             "размерът на хоризонталната мрежа може да бъде по реда на няколко километра ( km ) за на симулация до около 100 km за на симулация .", 
             "很好，以前从不去评价，不知道浪费了多少积分，现在知道积分可以换钱，就要好好评价了，后来我就把这段话复制走了，既能赚积分，还省事，走到哪复制到哪，最重要的是，不用认真的评论了，不用想还差多少字，直接发出就可以了，推荐给大家！！", 
             "สำหรับ ของเก่า ที่ จริงจัง ลอง honeychurch ของเก่า ที่ ไม่ 29 สำหรับ เฟอร์นิเจอร์ และ เงิน ไท ร้อง บริษัท ที่ 122 สำหรับ ลาย คราม", 
             "Alles in allem ein super schönes Teil, deshalb die 2 Sterne! Denn: Voice Control?! Nein, ein absoluter Witz. Die reagiert nämlich nur bedingt und wenn sie gerade meint. Sprachbefehle sind, egal wie man sie ausspricht, ein Glückstreffer. Meine Freundin sagte z.B. zu mir- naja ist eben ein Weib. Daraufhin schaltete sich der Akkuträger aus bzw fragte ob ich mir sicher bin ob ich ihn ausmachen möchte.... Zusätzlich kam das Teil bei mir mit kaputtem Glastank an. Da Amazon nicht selbst der Verkäufer ist, gibt es nur die Option der Rücksendung. Schade, denn das Gerät sieht super aus und liegt schön in der Hand. Allerdings ist eben die Sprachsteuerung eine Katastrophe. Bin echt enttäuscht...", 
             "Einer Freundin Geschenk da sie Flugbegleiterin ist und es gepasst hat. Allerdings hat der Anhänger nach 4-5 Wochen angefangen an den Ecken und Kanten braun zu wirken", 	
             "Didnt really seem to work much."]

# document labels
labels = ["pt", "bg", "zh", "th", "de", "de", "en"]

predictions = model.predict(documents=documents)
print(f"Labels: {labels}\nPredictions :{predictions}")
```
