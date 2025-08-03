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
# idnum source    lang    file  text    labels

print("doing re.sub")
# use the re.sub here to make texts the same for sorting, I don't need the texts so doesn't matter if they are different
for index, one in enumerate(en_lines):
    en_lines[index]["text"] = re.sub("[^a-zA-Z0-9]","",one["text"])
print(en_lines[0])
for index, one in enumerate(data):
    data[index][4] = re.sub("[^a-zA-Z0-9]","",one[4])
print(data[0])

# TODO 1% in five minutes, so slowww -> will take like 8 hours ughhh if this pace keeps up
# one option to take each wulczyn part separately which would maybe make this faster?? ask maybe filip for help with this?
wulczyn1 = []
wulczyn2 = []
wulczyn3 = []
for index1, one in enumerate(tqdm.tqdm(data)):
    if "toxic" in one[3]:
        wulczyn1.append(one)
    elif "aggressive" in one[3]:
        wulczyn2.append(one)
    elif "attack" in one[3]:
        wulczyn3.append(one)

# sorting those
wulczyn1 = sorted(wulczyn1, key=lambda d: str(d[4])) 
wulczyn2 = sorted(wulczyn2, key=lambda d: str(d[4])) 
wulczyn3 = sorted(wulczyn3, key=lambda d: str(d[4])) 
print(len(wulczyn1))
print(len(wulczyn2))
print(len(wulczyn3))

print("removing duplicates and taking labels")
new_wulczyn = []
# here take aggressive and attack as they have the same number of texts
# for index1, one in enumerate(tqdm.tqdm(wulczyn2)):
#     for index2, two in enumerate(wulczyn3): # TODO figure out if I can begin this loop from the index in the outer loop, would make it fast
#         # if text is the same
#         if one[4] == two[4]:
#             # if label same (None)
#             if one[5] == two[5]:
#                 new_wulczyn.append(one)
#                 break
#             # if label not the same
#             else:
#                 wulczyn2[index1][5] = one[5] + two[5]
#                 new_wulczyn.append(wulczyn2[index1])
#                 break

for one, two in zip(wulczyn2, wulczyn3):
    if one[4] == two[4]:
        #if label same (None)
        if one[5] == two[5]:
            new_wulczyn.append(one)
            continue
        # if label not the same
        else:
            nextone = one
            nextone[5] = one[5] + two[5]
            new_wulczyn.append(nextone)
            continue
    else:
        print("fail")
print("success")
print(len(new_wulczyn)) 

temp_wulczyn = [] # if exists in other wulczyns
temp2_wulczyn = [] # if not in other wulczyns

texts = [item[4] for item in new_wulczyn]
# this still ends up taking a while
for one in tqdm.tqdm(wulczyn1):
    if one[4] in texts:
        temp_wulczyn.append(one)
    else:
        # take out the ones that are not in the other lists
        # to use later
        temp2_wulczyn.append(one)

print(len(temp_wulczyn))
print(len(temp2_wulczyn))

for index1, one in enumerate(tqdm.tqdm(temp_wulczyn)):
    for index2, two in enumerate(new_wulczyn):
        if one[4] == two[4]:
            if one[5] == two[5]:
                # if same label no need to do anything
                break
            else:
                new_wulczyn[index2][5]=one[5] + two[5]
                break

new_wulczyn = new_wulczyn + temp2_wulczyn

print(len(new_wulczyn))
print("sorting")
# sort the list of dictionaries by text field to maybe make this faster
new_en = sorted(en_lines, key=lambda d: str(d["text"])) 
new_data = sorted(new_wulczyn, key=lambda d: str(d[4])) 
print(new_en[0])
print(new_data[0])


temp_en = []
texts = [item[4] for item in new_data]
# this still ends up taking a while
for one in tqdm.tqdm(en_lines):
    if one[4] in texts:
        temp_en.append(one)


print("matching texts")
# match text from english to the wulzcyn file and take english id and everything else from wulczyn
temp_list = []     # list of dictionaries
for one in tqdm.tqdm(temp_en):
    for two in new_data:
        if one["text"] == two[4]:
            temp = {}
            temp["file"] = two[1]
            temp["id"] = one["id"]
            temp["lang"] = two[2]
            temp["source"] = two[3]
            res = ast.literal_eval(two[5])
            temp["labels"] = res
            # no need for text yet
            temp_list.append(temp)
            break

print("found texts")
print(len(temp_list))

new_wulczyn=open("data/wulczyn.jsonl","wt")

# get the finnish text by looking at id and then appending to the dictionary 

print("sorting again")
# sort to make them in the same order even though the hashes are nonsense
new_temp = sorted(temp_list, key=lambda d: str(d['id'])) 
new_fi = sorted(fi_lines, key=lambda d: str(d['id'])) 

# here for fi could only take the ones that are in the new_temp, then I could just zip them below
temp_fi = []
ids = [item["id"] for item in new_temp]
# this still ends up taking a while
for one in tqdm.tqdm(new_fi):
    if one[4] in ids:
        temp_fi.append(one)

# TODO HERE COULD ZIP? SHOULD NOW BE SAME AMOUNT AND IN RIGHT ORDER 


print("matching ids")
for one in tqdm.tqdm(new_temp):
    for two in new_fi:
        if one["id"] == two["id"]:
            one["text"] = fi_lines["text"]
            break # break free from inner loop when found
    # save jsonline to file
    line=json.dumps(one,ensure_ascii=False,sort_keys=True)
    print(line,file=new_wulczyn)
