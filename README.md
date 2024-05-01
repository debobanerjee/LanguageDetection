# LanguageDetection

## Model Description
The [XLM-RoBERTa-base](https://huggingface.co/FacebookAI/xlm-roberta-base) model (pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages) has been adopted to build a language detection model - **XLM-RoBERTa-base-Language-Detection** (similar to https://huggingface.co/papluca/xlm-roberta-base-language-detection).

## Datasets
For fine-tuning and evaluation, the [Language Identification](https://huggingface.co/datasets/papluca/language-identification) dataset has been used, which consists of 90k (train/valid/test split are 70k/10k/10k, respectively) text sequences spanning 20 different languages.

## For Fine-tuning and Evaluation, first clone this repository
```
git clone https://github.com/debobanerjee/LanguageDetection.git
```
## create a virtual environment using conda
```
conda create -n lang-detect python=3.11 -c anaconda
```
## Install Required Packages
```
cd LanguageDetection
pip install -r requirements.txt
```

## Data Preprocessing
### Set configuration for data preprocessing inside `config.yml` file
```yaml
# Sample Configuration for data preprocessing
args:
    preprocessing:
        dataset_repoid: "papluca/language-identification"
        repo_type: "dataset"
        preprocessed_datadir: "data/preprocessed"
        datadir: "data"
        label_column: "labels"
```
### Start Data Preprocessing
```
python preprocess_data.py
```

## Training 
#### Fine-tuning has been done on the following GPU configuration
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:61:00.0 Off |                    0 |
| N/A   55C    P0    34W / 250W |      0MiB / 16280MiB |      3%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Set training configuration inside `config.yml` file
```yaml
# Sample training configuration
args:
    config-1:
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
        classification_report:
            digits: 3
            output_dict: False
        num_labels: 20
        datadir: "data"
        preprocessed_datadir: "data/preprocessed"
        model_ckpt_dir: "output/models/experiment-1"
        results_dir: "output/results/experiment-1/"
        log_dir: "output/logs/experiment-1"
```
### Start Training
```
python main.py
```

## Evaluation
##### Set `train_valid_step: False` inside `config.yml` 
### Start Evaluation
```
python main.py
```
### Generate Classification Report
#### For printing the classification report execute
```
python generate_classification_report.py
```
#### For saving classification report in a json file
##### Set `outfit_dict: True` inside `config.yml`, then execute
```
python generate_classification_report.py
```

### Results
```
precision    recall  f1-score   support

          pt      0.998     0.998     0.998       500
          pl      0.998     1.000     0.999       500
          tr      0.994     1.000     0.997       500
          el      1.000     1.000     1.000       500
          bg      0.996     1.000     0.998       500
          vi      1.000     1.000     1.000       500
          ru      1.000     0.998     0.999       500
          en      0.998     1.000     0.999       500
          es      0.998     1.000     0.999       500
          fr      1.000     1.000     1.000       500
          th      1.000     1.000     1.000       500
          zh      1.000     1.000     1.000       500
          it      1.000     0.992     0.996       500
          sw      0.986     0.996     0.991       500
          de      1.000     1.000     1.000       500
          ja      1.000     1.000     1.000       500
          nl      1.000     0.998     0.999       500
          ur      0.996     0.970     0.983       500
          ar      1.000     0.998     0.999       500
          hi      0.978     0.992     0.985       500

    accuracy                          0.997     10000
   macro avg      0.997     0.997     0.997     10000
weighted avg      0.997     0.997     0.997     10000
```

## Getting Started with Model Inference
#### Download the model weights from the shared [directory](https://drive.google.com/file/d/1_c9gaM9x7xWU_GAYyDawzaYMF3JqqF2j/view?usp=sharing). 
#### Change directory to LanguageDetection
```
cd LanguageDetection
```
#### Create the following models directory
```
mkdir -p output/models/experiment-4
```
#### Copy the downloaded model file into `output/models/experiment-4` directory.

#### Execute the below code
```python
from model import LanguageDetectionModelInference

# set the checkpoint path
checkpoint_path = "output/models/experiment-4/xlm-roberta-base-language-detection-best-model.pth"

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
