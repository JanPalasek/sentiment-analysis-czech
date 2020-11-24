# sentiment-analysis-czech
The most of NLP studies is performed using the english language. However, the rise of multilingual models allows us
to perform them on other languages. I decided to test it. This project performs sentiment analysis, classifying
sentences into three categories: negative, positive and neutral, using the recent, multi-lingual model, [xlm-roberta base](https://arxiv.org/pdf/1911.02116.pdf).

The sentiment analysis is done on datasets from [https://github.com/kysely/sentiment-analysis-czech](https://github.com/kysely/sentiment-analysis-czech). The
benchmarks will be done in the corresponding article soon.

## How to install the project
Install the requirements using following script.

```shell script
python -m pip install -r "requirements.txt"
```

Furthermore, we consider *src* as the root directory in this tutorial.

### Requirements
They are listed in *requirements.txt* file.

- Python >= 3.6
- sklearn
- huggingface/transformers
- Tensorflow
- Tensorflow Addons

## Datasets
There are 5 datasets available:
1. *csfd* - contains list of csfd review comments, split into negative, positive and neutral,
2. *facebook* - contains list of facebook comments,
3. *mall* - contains list of mall.cz reviews,
4. *all* - contains merged datasets from *csfd, facebook* and *mall*,
5. *handpicked_test* - contains list of handpicked tests used for analysis article.

Those datasets can be found in corresponding directory in *data*. There are also predefined train/test/validation splits in order
for the replication analysis to be easier (since seed generation works differently on different OS even using the same seed).

### How to create my own dataset
If you'd like to create a new dataset for sentiment analysis, create a new directory in *data* dir,
create files *negative.txt*, *neutral.txt* and *positive.txt* and fill them with corresponding sentences (one sentence for each line).

In order to use it in the training, we need to create 3 important csv files for training, validation and testing: *data_train.csv*,
*data_val.csv* and *data_test.csv*. Their creation is handled by *create_dataset.py*.

```shell script
python create_dataset.py --dataset="{DATASET_NAME}"
```

The script combines the contents of *negative.txt, neutral.txt, positive.txt*, creates a labeled dataset from it, shuffles it 
and splits it with ratio train/test/split 0.7/0.15/0.15 (this ratio can be changed in *create_dataset.py*).


#### Configuration files
To be able to train and evaluate the dataset, you also need to create its configuration file in *config* folder named
*{DATASET_NAME}.json*. The existing ones can be taken as an inspiration.

## Training
Training is done via fine-tuning a pretrained model. In this analysis, *xlm-roberta* was used for all datasets since it achieved better results.
The prepared configs can be found in directories *config*, specifying which model will be taken with its parameters. Model's
default parameters can be found in *config/model* directory.

Training can be done using following shell script:

```shell script
python train.py --dataset="{DATASET_NAME}"
```

In order to override default parameters specified by config files, you can send them to *train.py*
script as the script parameters. For example

```shell script
python train.py --dataset="{DATASET_NAME}" --model="bert" --dropout=0.1 --epochs=10
```

### Imbalanced dataset problem
If the dataset is imbalanced in favour of one or more classes (like *mall.cz* dataset), you can define its class weights.
Those can be automatically computed and assigned to corresponding json configuration file by following invocation

```shell script
python calculate_class_weight.py --dataset_full_path="{PATH_TO_DATASET_FILE}" --config_path="{PATH_TO_JSON_CONFIG}"
```

As variable *{PATH_TO_DATASET_FILE}* it is good to assign path to the csv training or the full data, config path should be
path to the corresponding json config file.

## Evaluation
In order to evaluate dataset, you need to define config similar to the ones in *config* directory.
Then the dataset can be evaluated using following command:

```shell script
python evaluate.py --dataset="{DATASET_NAME}"
```

The script outputs metrics into the standard output and additionally csv file with every sentence with expected and actual predictions. The
csv file can be found in newly created directory in *src/logs* directory.


