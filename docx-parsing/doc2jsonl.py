import docx
import sys
import argparse
import json
import tqdm

def yield_translations(fnames):
    for fname in tqdm.tqdm(fnames):
        d=docx.Document(fname)

        curr_doc=None
        for p in d.paragraphs:
            if len(p.runs) == 0: # check where the error comes from with this
                print("EMPTY")
                if curr_doc:
                    print(curr_doc["id"])
                    curr_doc["text"]= "EMPTY"
                    # do not yield or make None
                    #800897 is gone, only the id remains in file en 33

            elif p.runs[0].bold and p.runs[0].underline:
                if curr_doc:
                    yield curr_doc # if run into id again, change of comment
                curr_doc={"text":None,"id":p.text}
            else:
                if not curr_doc["text"]: #take the text that comes after bolded id
                    curr_doc["text"]=p.text
                else:
                    assert not curr_doc["text"] #if this fails it means a document with several paragraphs
                    curr_doc["text"]=p.text
        else:
            yield curr_doc
                    
                

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-data', help='Jsonl with metadata created by jsonl2docx')
    parser.add_argument('DOCXS', help='All translated docx files in a txt file')

    args = parser.parse_args()

    #let's read the txt file to a list to get the docx in correct order 
    # ORDER DOES NOT MATTER, ENG FILE 21 RUINS IT BECAUSE THERE IS A SPLIT WITH THOSE IDS TO HI AND DE 0 FILES + id 0 and 1
    # problems also later so ughhh
    with open(args.DOCXS) as f:
        docs = f.read().splitlines()

    with open(args.meta_data, 'r') as json_file:
        json_list = list(json_file)
        meta = [json.loads(jline) for jline in json_list] # list of dictionaries
    
    all_d=list(yield_translations(docs)) # list of dictionaries

    print(len(all_d)) # there is one more of these than there should hmmm??? I yielded the faulty one twice
    print(len(meta))
    assert len(all_d)==len(meta)

    combined=open("../data/combined-translated2.jsonl","wt") # keep it jsonl instead of turning back to tsv
    
    # sorting the ids in correct order fixes everything and now this is super fast :)
    # thankfully no ids were missing
    new_meta = sorted(meta, key=lambda d: int(d['idx'])) 
    new_all = sorted(all_d, key=lambda d: int(d['id'])) 


    for m,d in zip(new_meta, new_all):
        if int(d["id"]) == int(m["idx"]):
            # here make the meta and stuff together from the two dictionaries
            # I need the stuff from orig
            labels = m["orig"][4]
            # change string representation of list to actual list
            import ast
            res = ast.literal_eval(labels)
            d["labels"] = res # list of labels from here
            # in the end d has text, id(x), labels ++
            d["file"] = m["file_name"]
            d["lang"] = m["out_lang"]
            d["source"] = m["file_platform"]
        else:
            print("id error, 1st loop")
            print("meta:", m["idx"], "docx:", d["id"])
            raise Exception("oops")
        line=json.dumps(d,ensure_ascii=False,sort_keys=True)
        
        print(line,file=combined)