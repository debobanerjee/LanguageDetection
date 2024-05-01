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

### Results on the Test Set
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
#### Output
```python
Labels: ['pt', 'bg', 'zh', 'th', 'de', 'de', 'en']
Predictions :['pt', 'bg', 'zh', 'th', 'de', 'de', 'en']
```
#### For getting the Probability Scores, execute the following code
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

predictions, prob_scores = model.predict(documents=documents, prob=True)
print(f"Labels: {labels}\nPredictions :{predictions}")
print(f"Prob Scores: {prob_scores}")
```
#### Output
```python
Labels: ['pt', 'bg', 'zh', 'th', 'de', 'de', 'en']
Predictions :['pt', 'bg', 'zh', 'th', 'de', 'de', 'en']
Prob Scores: [{'pt': 0.9997825026512146, 'pl': 1.2613561921170913e-05, 'tr': 9.72355337580666e-06, 'el': 4.6459908844553865e-06, 'bg': 1.3339752513275016e-05, 'vi': 9.555774340697099e-06, 'ru': 1.735125988489017e-05, 'en': 9.923269317368977e-06, 'es': 3.237215059925802e-05, 'fr': 1.6904778021853417e-05, 'th': 2.3382708604913205e-06, 'zh': 3.827940417977516e-06, 'it': 2.9998876925674267e-05, 'sw': 1.1206372619199101e-05, 'de': 2.897447302530054e-06, 'ja': 3.1505769584327936e-06, 'nl': 1.8337595975026488e-05, 'ur': 7.4920703809766565e-06, 'ar': 6.85104987496743e-06, 'hi': 4.7762723625055514e-06}, {'pt': 1.691111538093537e-05, 'pl': 2.03679664991796e-05, 'tr': 7.269316938618431e-06, 'el': 1.4159442798700184e-05, 'bg': 0.9998072981834412, 'vi': 5.735988452215679e-06, 'ru': 3.190490315319039e-05, 'en': 8.126859938784037e-06, 'es': 5.802130544907413e-06, 'fr': 4.445161721378099e-06, 'th': 7.621899385412689e-06, 'zh': 8.594674909545574e-06, 'it': 7.88268334872555e-06, 'sw': 5.323835921444697e-06, 'de': 4.672691829910036e-06, 'ja': 4.58588556284667e-06, 'nl': 7.0730980041844305e-06, 'ur': 1.0712201401474886e-05, 'ar': 1.6647538359393366e-05, 'hi': 4.776458808919415e-06}, {'pt': 3.107967586402083e-06, 'pl': 2.739164301601704e-06, 'tr': 1.87203181667428e-06, 'el': 1.8634054868016392e-06, 'bg': 5.940046776231611e-06, 'vi': 5.975677595415618e-06, 'ru': 6.9721841100545134e-06, 'en': 9.646357284509577e-06, 'es': 7.222527074191021e-06, 'fr': 2.155838046746794e-06, 'th': 2.4796513571345713e-06, 'zh': 0.999923586845398, 'it': 3.983682745456463e-06, 'sw': 8.228495289586135e-07, 'de': 4.452786015463062e-06, 'ja': 3.485435627226252e-06, 'nl': 1.1444723213571706e-06, 'ur': 1.9280353171780007e-06, 'ar': 7.22137747288798e-06, 'hi': 3.4698809940891806e-06}, {'pt': 3.1402846616401803e-06, 'pl': 4.293893198337173e-06, 'tr': 6.275267423916375e-06, 'el': 1.6778054487076588e-05, 'bg': 7.590105724375462e-06, 'vi': 7.96166295913281e-06, 'ru': 1.2394767509249505e-05, 'en': 3.4305085137020797e-06, 'es': 1.0552742423897143e-05, 'fr': 3.944877789763268e-06, 'th': 0.9998767375946045, 'zh': 6.32304954706342e-06, 'it': 5.04939271195326e-06, 'sw': 7.738027306913864e-06, 'de': 3.240026671846863e-06, 'ja': 2.6590519155433867e-06, 'nl': 1.976752173504792e-06, 'ur': 3.817181550402893e-06, 'ar': 4.955375516146887e-06, 'hi': 1.1122567229904234e-05}, {'pt': 2.494320597179467e-06, 'pl': 5.6009189393080305e-06, 'tr': 1.4527884559356607e-05, 'el': 2.708122565309168e-06, 'bg': 2.498739604561706e-06, 'vi': 4.573015303321881e-06, 'ru': 1.091173635359155e-05, 'en': 8.107120265776757e-06, 'es': 6.015117833158001e-06, 'fr': 7.063579687383026e-06, 'th': 2.214829237345839e-06, 'zh': 8.173837159120012e-06, 'it': 7.624611043866025e-06, 'sw': 2.6713441911851987e-06, 'de': 0.9998854398727417, 'ja': 5.406694981502369e-06, 'nl': 1.143759254773613e-05, 'ur': 3.930259481421672e-06, 'ar': 5.243257419351721e-06, 'hi': 3.378183691893355e-06}, {'pt': 2.525995796531788e-06, 'pl': 6.004279839544324e-06, 'tr': 1.826863081078045e-05, 'el': 2.824646344379289e-06, 'bg': 2.5898864350892836e-06, 'vi': 5.4705269576516e-06, 'ru': 1.175388115370879e-05, 'en': 6.892401415825589e-06, 'es': 6.013076017552521e-06, 'fr': 7.554986950708553e-06, 'th': 2.8034774004481733e-06, 'zh': 5.757461167377187e-06, 'it': 8.763520781940315e-06, 'sw': 2.9815721518389182e-06, 'de': 0.9998759031295776, 'ja': 5.33535376234795e-06, 'nl': 1.5681445802329108e-05, 'ur': 4.169396561337635e-06, 'ar': 5.229569524090039e-06, 'hi': 3.37095571012469e-06}, {'pt': 1.5856498066568747e-05, 'pl': 1.977251667995006e-05, 'tr': 6.857124390080571e-06, 'el': 8.66918435349362e-06, 'bg': 4.9924428822123446e-06, 'vi': 4.174776677245973e-06, 'ru': 1.072639315680135e-05, 'en': 0.9997865557670593, 'es': 1.8128785086446442e-05, 'fr': 1.3372810826695058e-05, 'th': 4.562361482385313e-06, 'zh': 8.566461474401876e-06, 'it': 5.738758773077279e-06, 'sw': 1.261060606339015e-05, 'de': 1.1187005839019548e-05, 'ja': 7.924895726318937e-06, 'nl': 1.546534440421965e-05, 'ur': 2.7696769393514842e-05, 'ar': 1.0056736755359452e-05, 'hi': 7.161486337281531e-06}]
```

#### Or edit the `getting_started_model_inference.py` file and run it as shown below
```
python getting_started_model_inference.py
```
