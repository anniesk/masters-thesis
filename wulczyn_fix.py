import json
import bz2
import gzip
import argparse
import ast
import tqdm
import re

# python3 wulczyn_fix.py --en_jsonl_train ../wikipedia-toxicity-data-fi/train_en.jsonl.bz2 --en_jsonl_test ../wikipedia-toxicity-data-fi/test_en.jsonl.bz2 --fi_jsonl_train ../wikipedia-toxicity-data-fi/train_fi_deepl.jsonl.bz2 --fi_jsonl_test ../wikipedia-toxicity-data-fi/test_fi_deepl.jsonl.bz2 --wulczyn data/wulczyn.tsv.gz

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

temp_list = []     # list of dictionaries

print("doing re.sub")
# use the re.sub here to make texts the same for sorting, I don't need the texts so doesn't matter if they are different
for index, one in enumerate(en_lines):
    en_lines[index]["text"] = re.sub("[^a-zA-Z0-9]","",one["text"])
for index, one in enumerate(data):
    data[index][4] = re.sub("[^a-zA-Z0-9]","",one[4])

new_data = sorted(data, key=lambda d: str(d[4])) 

# TODO 1% in five minutes, so slowww -> will take like 8 hours ughhh if this pace keeps up
# one option to take each wulczyn part separately which would maybe make this faster?? ask maybe filip for help with this?
print("removing")
#some texts are there many times in the wulczyn, have to take out
text_list = []
remove_list = []
for index1, one in enumerate(tqdm.tqdm(new_data)):
    # if id already was then skip
    count = 0
    if one[4] in text_list:
        continue
    for index2, two in enumerate(new_data[index1:]):
        # double loop of the same file
        if one[4] == two[4]:
            count += 1
            # append new label
            if one[5] == two[5]:
                continue
            else:
                new_data[index1][5] = one[5] + two[5]
                remove = index1 + index2 # like this because index2 would otherwise start from zero
                remove_list.append(remove)
            if count == 2: # two here because the first is in the outer loop
                break
         
    text_list.append(one[4])

data_dedup = [v for i, v in enumerate(data) if i not in remove_list]
print("deleted")
print(len(data_dedup))



print("sorting")
# sort the list of dictionaries by text field to maybe make this faster
new_en = sorted(en_lines, key=lambda d: str(d["text"])) 
#new_data = sorted(data_dedup, key=lambda d: str(d[4])) # unnecessary because sorted above


# TODO THIS TAKES FOREVER, NEED TO FIGURE OUT A MORE EFFICIENT WAY, TWO BIG FILES, IS THERE A MORE EFFICIENT WAY?
# tried sorting, makes it way faster already but could be faster still
# if there is only data from train then I would not have to do test here and would be faster?


print("matching texts")
# match text from english to the wulzcyn file and take english id and everything else from wulczyn

for one in tqdm.tqdm(new_en):
    for two in new_data:
        count = 0
        if one["text"] == two[4]:
            count += 1
            temp = {}
            temp["file"] = two[1]
            temp["id"] = one["id"]
            temp["lang"] = two[2]
            temp["source"] = two[3]
            res = ast.literal_eval(two[5])
            temp["labels"] = res
            # no need for text yet
            temp_list.append(temp)
            if count == 3: # unfortunately this did not make this faster
                break

print("found texts")
print(len(temp_list))

new_wulczyn=open("data/wulczyn.jsonl","wt")

# get the finnish text by looking at id and then appending to the dictionary 

print("sorting again")
# sort to make them in the same order even though the hashes are nonsense
new_temp = sorted(temp_list, key=lambda d: str(d['id'])) 
new_fi = sorted(fi_lines, key=lambda d: str(d['id'])) 

print("matching ids")
for one in tqdm.tqdm(new_temp):
    for two in new_fi:
        if one["id"] == two["id"]:
            one["text"] = fi_lines["text"]
            break # break free from inner loop when found
    # save jsonline to file
    line=json.dumps(one,ensure_ascii=False,sort_keys=True)
    print(line,file=new_wulczyn)
