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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score
from collections import defaultdict
import matplotlib.pyplot as plt

# do from new labels
def do_label(example):
    if example["new label"] == "not-toxic":
        example["int_label"] = 0
    else:
        example["int_label"] = 1
    return example

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
    with open(data, 'r') as json_file:
        json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]
    # there is now a list of dictionaries

    df=pd.DataFrame(lines) # probably a better way but let's do this
    # previous way of binarizing
    df['labels'] = df[label_names].values.tolist()

    # change to binary: if toxic 1 if clean 0
    # first get sum of labels
    df['labels'] = df.labels.map(sum) #df[label_names].sum(axis=1)

    # check that the ratio between clean and toxic is still the same! (it is)
    train_toxic = df[df["labels"] > 0]
    train_clean = df[df["labels"] == 0]

    # # then change bigger than 0 to 1 and 0 stays 0
    df.loc[df["labels"] > 0, "labels"] = 1

    # only keep the columns text and one_hot_labels
    df = df[['text', 'labels']]
    #print(df.head())

    dataset = datasets.Dataset.from_pandas(df)

    return dataset






# this should prevent any caching problems I might have because caching does not happen anymore
datasets.disable_caching()

# parse arguments
parser = argparse.ArgumentParser(
            description="A script for predicting toxic texts based on a toxicity classifier and finding the best threshold",
            epilog="Made by Anni Eskelinen"
        )
parser.add_argument('--model', required=True,
    help="the model name")
parser.add_argument('--threshold', type=float, default=0.5,
    help="the threshold for the predictions")
parser.add_argument('--data', required=True, nargs="+",
    help="the file name of the raw text to use for predictions")
parser.add_argument('--tokenizer', required=True,
    help="the tokenizer to use for tokenizing new text")
args = parser.parse_args()
print(args)

pprint = PrettyPrinter(compact=True).pprint

# instantiate model, this is pretty simple
model=transformers.AutoModelForSequenceClassification.from_pretrained(args.model)

trainer = transformers.Trainer(
    model=model
) 

# read the data in
data = args.data

if data[0] == "all_reannotated.tsv":
    # read re-annotated data as tsv
    dataset = datasets.load_dataset("csv", data_files=args.data, sep="\t")
    dataset = dataset.map(do_label)

elif data[0] == "jigsaw":
    dataset = json_to_dataset("test_fi_deepl.jsonl")
    texts = dataset["text"]
    trues = dataset["labels"]

else:
    # toxic comment collection
    filename = "combined-translated-unified-fixed.jsonl"
    multi_df = json_to_multidf(filename)

    dataset, df = df_to_dataset(data, multi_df)
    texts = dataset["text"]
    trues = dataset["labels"]

tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

def tokenize(example):
    return tokenizer(
        example["text"],
        padding='max_length', # this got it to work, data_collator could have helped as well?
        max_length=512,
        truncation=True,
    )

#map all the examples
dataset = dataset.map(tokenize)


if data[0] == "all_reannotated.tsv":
    texts = dataset["train"]["text"]
    trues = dataset["train"]["int_label"] 
    old_labels = dataset["train"]["label"]

    dataset = dataset.remove_columns(["text", "label", "new label"])


threshold = args.threshold

# see how the labels are predicted
if data[0] == "all_reannotated.tsv":
    test_pred = trainer.predict(dataset["train"])
else:
    test_pred = trainer.predict(dataset)



# import torch.nn.functional as F
# tensor = torch.from_numpy(predictions)
# probabilities = F.softmax(tensor, dim=1) # turn to probabilities using softmax
# probabilities = probabilities.tolist()
# print(probabilities[:10]) # this is now a tensor with two probabilities per example (two labels)
# print(predictions[:10])

# if threshold == 0.5:
#     preds = predictions.argmax(-1) # the -1 gives the indexes of the predictions, takes the one with the biggest number
#     # argmax can be used on the probabilities as well although the tensor needs to changed to numpy array first
# else:
#     # idea that if there is no high prediction for clean label then we set it to toxic or the other way around
#     # set p[0] or p[1] depending on which we wanna concentrate on
#     # could switch the other way as a test^^^ TODO next
#     preds = [0 if p[1] < threshold else np.argmax(p) for p in probabilities]  # if toxic below threshold count as clean (set index to 0)

# # get all labels and their probabilities
# all_label_probs = []
# for prob in probabilities:
#     all_label_probs.append(tuple(zip(prob, ["clean", "toxic"])))

# # get predicted labels
# labels = []
# idx2label = dict(zip(range(2), ["clean", "toxic"]))
# for val in preds: # index
#     labels.append(idx2label[val])

# # now just loop to a list, get the probability with the indexes from preds
# probs = []
# for i in range(len(probabilities)):
#     probs.append(probabilities[i][preds[i]]) # preds[i] gives the correct index for the current probability



def predictions_to_csv(trues, preds, texts):
    """ Saves a dataframe to .csv with texts, correct labels and predicted labels to see what went right and what went wrong.
    
    Modified from https://gist.github.com/rap12391/ce872764fb927581e9d435e0decdc2df#file-output_df-ipynb

    Parameters
    ---------
    trues: list
        list of correct labels
    preds: list
        list of predicted labels
    dataset: Dataset
        the dataset from which to get texts

    """

    # idx2label = dict(zip(range(2), ["clean", "toxic"]))
    # print(idx2label)

    # # Gathering vectors of label names using idx2label (modified single-label version)
    # true_labels, pred_labels = [], []
    # for val in trues:
    #     true_labels.append(idx2label[val])
    # for val in preds:
    #     pred_labels.append(idx2label[val])

    predictions = preds.argmax(-1)
    import torch.nn.functional as F
    tensor = torch.from_numpy(preds)
    softmaxed = F.softmax(tensor, dim=1) 
    probabilities = [max(sublist) for sublist in softmaxed]
    probabilities = [tensor.item() for tensor in probabilities]


    # Converting lists to df
    comparisons_df = pd.DataFrame({'text': texts, 'true_label': trues, 'pred_label':predictions, "probability": probabilities})
    comparisons_df.to_csv('all_comparison.csv')
    #print(comparisons_df.head())



def get_predictions(dataset, trainer, pprint):
    test_pred = trainer.predict(dataset['test'])
    # this actually has metrics because the labels are available so evaluating is technically unnecessary since this does both! (checked documentation)

    predictions = test_pred.predictions # logits
    print(predictions) # to look at what they look like

    import torch.nn.functional as F
    tensor = torch.from_numpy(predictions)
    probabilities = F.softmax(tensor, dim=1) # turn to probabilities using softmax

    print(probabilities) # this is now a tensor with two probabilities per example (two labels)

    #THIS
    preds = predictions.argmax(-1) # the -1 gives the indexes of the predictions, takes the one with the biggest number
     # argmax can be used on the probabilities as well although the tensor needs to changed to numpy array first

    # OR THIS
    # # idea that if there is no high prediction for e.g. clean label then we set it to toxic (or the other way around)
    # threshold = 0.5
    # # set p[0] or p[1] depending on which we wanna concentrate on
    # preds = [1 if p[0] < threshold else np.argmax(p) for p in probabilities] 

    # # TODO could implement in regular evaluation as well with threshold optimization? to see whether it improves the results or not (seems confusing so probably no)
 

    labels = []
    idx2label = dict(zip(range(2), ["clean", "toxic"]))
    for val in preds: # index
        labels.append(idx2label[val])

    # now just loop to a list, get the probability with the indexes from preds
    probabilities = probabilities.tolist()
    probs = []
    for i in range(len(probabilities)):
        probs.append(probabilities[i][preds[i]]) # preds[i] gives the correct index for the current probability

    texts = dataset["train"]["text"]
    # lastly use zip to get tuples with (text, label, probability)
    prediction_tuple = tuple(zip(texts, labels, probs))

    # make into list of tuples
    toxic = [item for item in prediction_tuple
          if item[1] == "toxic"]
    clean = [item for item in prediction_tuple
          if item[1] == "clean"]

    # now sort by probability, descending
    toxic.sort(key = lambda x: float(x[2]), reverse=True)
    clean.sort(key = lambda x: float(x[2]), reverse=True)
    clean2 = sorted(clean, key = lambda x: float(x[2])) # ascending

    # beginning most toxic, middle "neutral", end most clean
    # all = toxic + clean2

    pprint(toxic[:5])
    pprint(toxic[-5:]) # these two middle are the closest to "neutral" where the threshold is
    pprint(clean[-5:])
    pprint(clean[:5])


def compute_metrics(pred, trues):
    """Computes the metrics"""

    #labels = pred.label_ids
    probs = pred.predictions
    preds = pred.predictions.argmax(-1)

    # import torch.nn.functional as F
    # tensor = torch.from_numpy(probs)
    # probabilities = F.softmax(tensor, dim=1) # turn to probabilities using softmax
    # probabilities = probabilities.tolist()

    precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, average='macro') 
    # micro or macro
    acc = accuracy_score(trues, preds)
    #roc_auc = roc_auc_score(y_true=labels, y_score=probabilities, average = 'micro', multi_class='ovr')
    wacc = balanced_accuracy_score(trues, preds)

    print(classification_report(trues, preds, target_names=["not toxic", "toxic"]))

    return {
        'accuracy': acc,
        'weighted_accuracy': wacc,
        #'roc_auc': roc_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


predictions = test_pred.predictions
preds = predictions.argmax(-1)

import torch.nn.functional as F
tensor = torch.from_numpy(predictions)
softmaxed = F.softmax(tensor, dim=1) 


print(compute_metrics(test_pred, trues))



labels = ["Non-toxic", "Toxic"]


from sklearn.metrics import PrecisionRecallDisplay,precision_recall_curve, roc_curve

# precision recall curve
precision = dict()
recall = dict()
for i in range(len(labels)):
    precision[i], recall[i], _ = precision_recall_curve(trues,
                                                        softmaxed[:, i])
    plt.plot(recall[i], precision[i], lw=2, label=labels[i])
    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.savefig('prec-rec-curve.png')


# roc curve
# fpr = dict()
# tpr = dict()

# for i in range(len(labels)):
#     fpr[i], tpr[i], _ = roc_curve(trues,
#                                   softmaxed[:, i])
#     plt.plot(fpr[i], tpr[i], lw=2, label=labels[i])

# plt.xlabel("false positive rate")
# plt.ylabel("true positive rate")
# plt.legend(loc="best")
# plt.title("ROC curve")

# plt.savefig('roc-curve.png')






#get_predictions(dataset, trainer, pprint)


predictions_to_csv(trues, predictions, texts)




# from sklearn.metrics import ConfusionMatrixDisplay
# disp = ConfusionMatrixDisplay.from_predictions(trues, preds)
# disp.plot()
# plt.savefig('cfmatrix.png')