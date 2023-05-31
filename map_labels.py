import json
import sys
import ast

data = sys.argv[1]

with open(data, 'r') as json_file:
    json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list] # list of dictionaries

combined=open("./data/combined-translated-uni.jsonl","wt")

for one in lines:
    if not one["labels"]:
        one["label"] = "non-toxic"
    if one["text"] == "EMPTY":
        continue
    
    for label in one["labels"]:
        # if non-toxic
        if label == "none" or label == "normal" or label == "other" or label == "positive" or label == "appropriate":
            one["label"] = "non-toxic"
            break
        #skip
        if label == "idk/skip":
            break
        #If empty
        if not len(label) or label == "":
            one["label"] = "non-toxic"
            break
        # if toxic
        else:
            one["label"] = "toxic"
            break

    # if label added
    if "label" in one:
        one.pop("labels")
        line=json.dumps(one,ensure_ascii=False,sort_keys=True)
        print(line,file=combined)