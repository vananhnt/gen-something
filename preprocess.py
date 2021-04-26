import pandas as pd
import os
import re
import glob
import pickle
from tqdm import tqdm
from transformers import RobertaTokenizer

unstrip_dataset = './ghidra/unstrip'
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

def to_token(s):
    #values = [plain2_tok(t.value, t.kind) for t in tokenizer(s, new_line=False) if t.kind != 'whitespace']
    values = tokenizer.tokenize(s)
    #result = ' '.join([v for v in values if v is not None])
    return values

def stripcomments(text):
    return re.sub('//.*?\n|/\*.*?\*/', '', text, flags=re.S)

def load_dataframe(dataset_dir):
    fileID = 0
    fid = 0
    number_files = len(os.listdir((dataset_dir))) # dir is your directory path
    
    row_list = []
    
    header = ['id', 'funcname', 'signature', 'decompiled', 'disassembly', 'bytes', 'address', 'sampleID']
    for filename in tqdm(glob.iglob('./ghidra/unstrip/*.pkl'), total=number_files):
        funcmap = pickle.load(open(filename, "rb" ))
        for k in funcmap:
            funcname = k
            (signature, dec_func, disasm_result, byte, addr) = funcmap[k]      
            func_dict = {'id': fid, 'funcname': funcname, 
                         'signature': signature, 'decompiled': stripcomments(dec_func), 
                         'disassembly': disasm_result, 'bytes': byte, 
                         'address':addr, 'sampleID':fileID,
                        }
            fid += 1
        fileID +=1
        row_list.append(func_dict)
    df = pd.DataFrame(row_list, columns = header)               
    return df

def extract_decompile(dataset_dir):
    # return dict : idx, source, target
    df = load_dataframe(dataset_dir)
    samples = df.set_index('id')[['decompiled', 'funcname']].T.to_dict('list')
    return samples

