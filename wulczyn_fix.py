import json
import bz2
import gzip
import argparse
import ast
import tqdm
import re

# python3 wulczyn_fix.py --en_jsonl_train ../wikipedia-toxicity-data-fi/train_en.jsonl.bz2 --en_jsonl_test ../wikipedia-toxicity-data-fi/test_en.jsonl.bz2 --fi_jsonl_train ../wikipedia-toxicity-data-fi/train_fi_deepl.jsonl.bz2 --fi_jsonl_test ../wikipedia-toxicity-data-fi/test_fi_deepl.jsonl.bz2 --wulczyn data/wulczyn.tsv.gz

# takes so long because loops upon loops I could maybe optimize this by using something other than loops

parser = argparse.ArgumentParser()
parser.add_argument('--en_jsonl_train', required=True,
    help="The English jsonl train file")
parser.add_argument('--en_jsonl_test', required=True,
    help="The English jsonl test file")
parser.add_argument('--fi_jsonl_train', required=True,
    help="The Finnish jsonl train file")
parser.add_argument('--fi_jsonl_test', required=True,
    help="The Finnish jsonl test file")
parser.add_argument('--wulczyn', required=True,
    help="The wulzcyn file from the toxic comment collection")

args = parser.parse_args()


# open the english wikipedia data that we had with multilabel jsonl
with bz2.open(args.en_jsonl_train, 'rt') as json_file:
    json_list = list(json_file)
    en_lines = [json.loads(jline) for jline in json_list] # list of dictionaries
with bz2.open(args.en_jsonl_test, 'rt') as json_file:
    json_list = list(json_file)
    en_lines = en_lines + [json.loads(jline) for jline in json_list] # list of dictionaries


# open the finnish translated jsonl data
with bz2.open(args.fi_jsonl_train, 'rt') as json_file:
    json_list = list(json_file)
    fi_lines = [json.loads(jline) for jline in json_list] # list of dictionaries
with bz2.open(args.fi_jsonl_train, 'rt') as json_file:
    json_list = list(json_file)
    fi_lines = fi_lines + [json.loads(jline) for jline in json_list] # list of dictionaries


# open the wulzcyn data from the toxic comment collection
with gzip.open(args.wulczyn, 'rt') as f:
    data = f.readlines()
for i in range(len(data)):
    data[i] = data[i].replace("\n", "")
    data[i]=data[i].split("\t")

print(data[0])
print(len(data))
# idnum file    lang    source  text    labels

final = []     # list of dictionaries

# THIS TAKES FOREVER, NEED TO FIGURE OUT A MORE EFFICIENT WAY, TWO BIG FILES, IS THERE A MORE EFFICIENT WAY?
# - could sort the lists by the texts? then it would maybee be faster
# using sets impossible because dictionaries are not hashable
# list of lists? like the wulczyn one

# match text from english to the wulzcyn file and take english id and everything else from wulczyn
for one in tqdm.tqdm(en_lines):
    for two in data:
        # here I probably need to normalize to get the texts to be the same? bc I doubt they look exactly the same
        if re.sub("[^a-zA-Z0-9]","",one["text"]) == re.sub("[^a-zA-Z0-9]","",two[4]):
            temp = {}
            temp["file"] = two[1]
            temp["id"] = one["id"]
            temp["lang"] = two[2]
            temp["source"] = two[3]
            res = ast.literal_eval(two[5])
            temp["labels"] = res
            # no need for text yet
            final.append(temp)

print("found texts")
print(len(final))
# results in having some ids maybe three times
# if id three times, put labels into one of those and delete duplicates
id_list = []
remove_list = []
for one in tqdm.tqdm(final):
    # if id already was then skip
    if one["id"] in id_list:
        continue
    for index, two in enumerate(final):
        # double loop of the same file
        if one["id"] == two["id"]:
            # append new label
            if one["labels"] == two["labels"]:
                continue
            else:
                one["labels"] = one["labels"] + two["labels"]
                remove_list.append(index)
         
    id_list.appen(one["id"])

deleted_duplicates = [v for i, v in enumerate(final) if i not in remove_list]
print("deleted")
print(len(deleted_duplicates))

new_wulczyn=open("data/wulczyn.jsonl","wt")

# get the finnish text by looking at id and then appending to the dictionary 
for one in tqdm.tqdm(deleted_duplicates):
    for two in fi_lines:
        if one["id"] == two["id"]:
            one["text"] = fi_lines["text"]
            break
    # save jsonline to file
    line=json.dumps(one,ensure_ascii=False,sort_keys=True)
    print(line,file=new_wulczyn)
