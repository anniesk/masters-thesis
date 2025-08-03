import datasets
import transformers
from pprint import PrettyPrinter
import logging
import argparse
import pandas as pd
import numpy as np
import json
import torch
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score, precision_recall_curve
from collections import defaultdict
import matplotlib.pyplot as plt

""" Toxicity classifier

This script is to be used for toxicity classification with jigsaw toxicity dataset in English (which is the original language)
 and Finnish (to which the data was translated using DeepL). The data is accepted in a .jsonl format and the data can be found in the data folder of the repository.

The labels of the dataset are:
- label_identity_attack
- label_insult
- label_obscene
- label_severe_toxicity
- label_threat
- label_toxicity
- label_clean
+ no labels means that the text is clean


The script includes a binary classification where if there is a label for the text it is considered toxic and if there are no labels the text is clean/non-toxic.

List for necessary packages to be installed (could also check import list):
- pandas
- transformers
- datasets
- numpy
- torch

Information about the arguments to use with script can be found by looking at the argparse arguments with 'python3 toxic_classifier.py -h'.
"""


parser = argparse.ArgumentParser(
        description="A script for classifying toxic data in a binary manner",
        epilog="Made by Anni Eskelinen"
    )
parser.add_argument('--files',nargs="+", required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--save_name', required=True)
parser.add_argument('--jigsaw', action='store_true', default=False)
parser.add_argument('--batch', type=int, default=8,
    help="The batch size for the model"
)
parser.add_argument('--epochs', type=int, default=3,
    help="The number of epochs to train for"
)
parser.add_argument('--learning', type=float, default=8e-6,
    help="The learning rate for the model"
)
parser.add_argument('--loss', action='store_true', default=False,
        help="If used different class weights are used for the loss function")
args = parser.parse_args()
print(args)
#usage as args.VARIABLENAME


label_names = [
    'label_identity_attack',
    'label_insult',
    'label_obscene',
    'label_severe_toxicity',
    'label_threat',
    'label_toxicity'
]

def json_to_dataset(data):
    """ Reads the data from .jsonl format and turns it into a dataset using pandas.
    
    Parameters
    ----------
    data: str
        path to the file from which to get the data

    Returns
    -------
    dataset: Dataset
        the data in dataset format
    """

    # first I need to read the json lines
    with open(data[0], 'r') as json_file:
        json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]
    with open(data[1], 'r') as json_file:
        json_list = list(json_file)
    lines2 = [json.loads(jline) for jline in json_list]
    # there is now a list of dictionaries

    df=pd.DataFrame(lines+lines2) # probably a better way but let's do this
    # previous way of binarizing
    df['labels'] = df[label_names].values.tolist()

    # change to binary: if toxic 1 if clean 0
    # first get sum of labels
    df['labels'] = df.labels.map(sum) #df[label_names].sum(axis=1)

    # check that the ratio between clean and toxic is still the same! (it is)
    train_toxic = df[df["labels"] > 0]
    train_clean = df[df["labels"] == 0]

    # new binarization only focusing on toxicity label because e.g., something can be obscene but not toxic
    # df['labels'] = None
    # df.loc[df['label_toxicity'] == 0, 'labels'] = 0
    # df.loc[df['label_toxicity'] == 1, 'labels'] = 1

    # train_toxic = df[df['labels'] == 1]
    # train_clean = df[df['labels'] == 0]


    # print("toxic: ", len(train_toxic))
    # print("clean: ", len(train_clean))

    # # then change bigger than 0 to 1 and 0 stays 0
    df.loc[df["labels"] > 0, "labels"] = 1

    # only keep the columns text and one_hot_labels
    df = df[['text', 'labels']]
    #print(df.head())

    dataset = datasets.Dataset.from_pandas(df)

    return dataset

# FOR TOXIC COMMENT COLLECTION
def json_to_multidf(data):

    # first I need to read the json lines
    with open(data, 'r', encoding="utf-8") as json_file:
        json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]
    # there is now a list of dictionaries

    df=pd.DataFrame(lines)

    grouped_datasets = df.groupby('file')
    multi_df = {}
    for file, group in grouped_datasets:
       multi_df[file] = df[df.file == file]

    return multi_df

# change the "toxic" and "non-toxic" to 1 and 0

def df_to_dataset(filename, multi_df):
    """ Reads the data from pandas dataframe format and turns it into a dataset.

    Parameters
    ----------
    data: str
        path to the file from which to get the data

    Returns
    -------
    dataset: Dataset
        the data in dataset format
    """

    if type(filename) is list:
        for i in range(len(filename)):
            df1 = multi_df[filename[i]]
            if i != 0:
                # concat the rest 
                df = pd.concat([df,df1])
            else:
                # make the first the og
                df = df1
    else:
        # if only one filename
      df = multi_df[filename]

    df['labels'] = None
    df.loc[df['label'] == "non-toxic", 'labels'] = 0
    df.loc[df['label'] == "toxic", 'labels'] = 1

    # only keep the columns text and one_hot_labels
    df1 = df[['text', 'labels']]
    #print(df.head())

    dataset = datasets.Dataset.from_pandas(df1)

    return dataset, df


# def make_class_weights(train):
#     """Calculates class weights for the loss function based on the train split. """

#     labels = train["labels"] # get all labels from train split
#     n_samples = (len(labels))
#     n_classes = 2
#     c=Counter(labels)
#     w1=n_samples / (n_classes * c[0])
#     w2=n_samples / (n_classes * c[1])
#     weights = [w1,w2]
#     class_weights = torch.tensor(weights).to("cuda:0") # have to decide on a device

#     print(class_weights)
#     return class_weights


def compute_metrics(pred):
    """Computes the metrics"""

    labels = pred.label_ids
    probs = pred.predictions
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro') # micro seemed to always be the same
    # micro or macro
    acc = accuracy_score(labels, preds)
    #roc_auc = roc_auc_score(y_true=labels, y_score=probs, average = 'micro', multi_class='ovr')
    wacc = balanced_accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'weighted_accuracy': wacc,
        #'roc_auc': roc_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)


# class newTrainer(transformers.Trainer):
#     """A custom trainer to use a different loss and to use different class weights"""

#     def __init__(self, class_weights, **kwargs):
#         super().__init__(**kwargs)
#         self.class_weights = class_weights

#     def compute_loss(self, model, inputs, return_outputs=False):
#         """Computes loss with different class weights"""

#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         # include class weights in loss computing
#         if args.loss == True:
#             loss_fct = torch.nn.CrossEntropyLoss(weight = self.class_weights)
#         else:
#             loss_fct = torch.nn.CrossEntropyLoss()
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
#             labels.view(-1))
#         return (loss, outputs) if return_outputs else loss



def main():
    # this should prevent any caching problems I might have because caching does not happen anymore
    datasets.disable_caching()

    pprint = PrettyPrinter(compact=True).pprint
    logging.disable(logging.INFO)

    if args.jigsaw == True:
        train = json_to_dataset(["train_fi_deepl.jsonl", "test_fi_deepl.jsonl"])

    # toxic comment collection
    data = "combined-translated-unified-fixed.jsonl"
    multi_df = json_to_multidf(data)
    filename = args.files # 'novak2021sl.csv' # ['ousidhoum2019en_with_stopwords.csv', 'ousidhoum2019fr.csv']

    dataset, df = df_to_dataset(filename, multi_df)

    # put the datasets together and shuffle
    if args.jigsaw == True:
        full_dataset = datasets.concatenate_datasets([train, dataset])
    else:
        full_dataset = dataset

    full_dataset = full_dataset.shuffle(seed=42) # shuffle so the dataset examples are not in order

    toxic = full_dataset.filter(lambda example: example["labels"] == 1)
    clean = full_dataset.filter(lambda example: example["labels"] == 0)
    print("clean:", len(clean))
    print("toxic:", len(toxic))


    train, dev = full_dataset.train_test_split(test_size=0.2).values() 
    dataset = datasets.DatasetDict({"train":train,"dev":dev})


    #class_weights = make_class_weights(train)

    print(dataset)

    #build model

    model_name = args.model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(
            example["text"],
            max_length=512,
            truncation=True
        )
        
    dataset = dataset.map(tokenize)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, cache_dir="new_cache_dir/", label2id={"not toxic": 0, "toxic": 1}, id2label={0: "not toxic", 1: "toxic"})

    # Set training arguments 
    trainer_args = transformers.TrainingArguments(
        f"checkpoints/{args.save_name}",
        eval_strategy="steps",
        eval_steps=25000,
        logging_strategy="steps",  # number of epochs = how many times the model has seen the whole training data
        logging_steps=25000,
        save_strategy="best", # save when best metric achieved
        load_best_model_at_end=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning,
        metric_for_best_model = "eval_f1", # this changes the best model to take the one with the best (biggest) f1 instead of the default: 
        #best (smallest) training or eval loss (seems to be random?)
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32
    )

    data_collator = transformers.DataCollatorWithPadding(tokenizer)
    # Argument gives the number of steps of patience before early stopping
    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience=5
    )
    training_logs = LogSavingCallback()


    eval_dataset=dataset["dev"] 

    trainer = transformers.Trainer(
        #class_weights=class_weights,
        model=model,
        args=trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer = tokenizer,
        callbacks=[early_stopping, training_logs]
    )

    trainer.train()

    #trainer.model.save_pretrained(f"models/{args.save_name}")
    trainer.save_model(f"models/{args.save_name}")
    print("saved")


    eval_results = trainer.evaluate(dataset["dev"]) #.select(range(20_000)))
    #pprint(eval_results)
    print('F1_micro:', eval_results['eval_f1'])

    # see how the labels are predicted
    test_pred = trainer.predict(dataset['dev'])
    trues = test_pred.label_ids
    predictions = test_pred.predictions # no softmax needed, only if I want the probabilities saved
    print(predictions)
    preds = predictions.argmax(-1)

    print(classification_report(trues, preds, target_names=["not toxic", "toxic"]))

   # print("roc_auc:", eval_results["eval_roc_auc"])

if __name__ == "__main__":
    main()