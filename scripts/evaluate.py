from tqdm import tqdm
from model import Seq2Seq
import torch
import glob
import argparse
import pickle
import numpy as np
from preprocess import extract_decompile, args_parser
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from entities import convert_examples_to_features, read_examples
import argparse
import torch
import bleu
import torch.nn as nn

def load_result(output_dir_path):
    y_trues = list()
    y_preds = list()
    bleus = []
    for gold in tqdm(glob.iglob(output_dir_path + '**/*.gold', recursive=True)):
        dev_bleu = 0
        test = gold.replace('gold', 'output') 
        #print(test, gold)
        with open(gold,'r') as f, open(test, 'r') as f1:
            gold_text = str(f.readlines()).split('\\t',1)[1].split('\\n',1)[0]
            predictions = f1.readlines()
            output_text = str(predictions).split('\\t',1)[1].split('\\n',1)[0]
            #print(gold_text, output_text)
            if 'FUN' not in gold_text:
                y_trues.append(gold_text)
                y_preds.append(output_text)
                (goldMap, predictionMap) = bleu.computeMaps(predictions, gold) 
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                bleus.append(dev_bleu)
    precision, recall = calculate_precision_recall(y_trues, y_preds)
    f1 = compute_f1(precision, recall)
    bleu_t = np.float32(sum(bleus) / len(bleus))
    print(precision, recall)
    print(f1)
    print(bleu_t)
        
"""
Precision = #correctly_predicted_tokens / #predicted_tokens
Recall =  #correctly_predicted_tokens / #original_tokens
"""


def calculate_precision_recall(original_names, predicted_names):
    precision_list = list()
    recall_list = list()
    line_number = 0

    if len(original_names) == 0 or len(predicted_names) == 0:
        #print("Fiorella Metrics Error Length 0")
        return np.float32(0.0), np.float32(0.0)

    for original_name, predicted_name in zip(original_names, predicted_names):
        if isinstance(original_name, list):
            original_name_tokens = list(set(original_name))
            predicted_name_tokens = list(set(predicted_name))
        else:
            original_name_tokens = list(set([x for x in original_name.strip("\n").split(" ") if len(x)>0]))
            predicted_name_tokens = list(set([x for x in predicted_name.strip("\n").split(" ") if len(x)>0]))
            
        # num1 = len(set(original_name_tokens).intersection(set(predicted_name_tokens)))      
        num = len(set(original_name_tokens) & set(predicted_name_tokens))

        if len(predicted_name_tokens) and len(original_name_tokens) > 0:
            per_func_precision = num / len(predicted_name_tokens)
            per_func_recall = num / len(original_name_tokens)
        else:
            per_func_precision = 0
            per_func_recall = 0

        precision_list.append(per_func_precision)
        recall_list.append(per_func_recall)
        line_number += 1

    precision = np.float32(sum(precision_list) / len(precision_list))
    recall = np.float32(sum(recall_list) / len(recall_list))
    #return compute_f1(precision, recall)
    return precision, recall

def compute_f1(precision, recall):
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1


def load_function(pkl_file):
    funcmap = pickle.load(open(pkl_file, "rb" ))
    for k in funcmap:
            funcname = k
            (signature, dec_func, disasm_result, byte, addr) = funcmap[k]      
            print(funcname)

def predict(bin_file):
    parser = argparse.ArgumentParser()
    args = args_parser(parser)
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    config = RobertaConfig.from_pretrained("microsoft/codebert-base")
    encoder = RobertaModel.from_pretrained("microsoft/codebert-base")    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    model.to(device)
    sample_dict = extract_decompile(bin_file)
    
    for idx in sample_dict:
        
            eval_examples = read_examples(sample_dict, idx=idx)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
            eval_data = TensorDataset(all_source_ids,all_source_mask)   

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            
            #Predict and calculate bleu
            
            model.load_state_dict(torch.load(args.load_model_path))
            model.eval() 
            p=[]
            for batch in eval_dataloader:
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask= batch                  
                with torch.no_grad():
                    preds = model(source_ids=source_ids,source_mask=source_mask)  
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions=[]
            for ref,gold in zip(p,eval_examples):
                print(ref.replace(' ', ''), '===', gold.target.replace(' ', '')) 
                    
            
#load_result('./model_cross_asm_codebert/')
#load_function('./ghidra/unstrip/dev/f95101721d069c8e1f34d50e58c3892d.pkl')
if __name__ == "__main__":
    load_result('./model_cross_asm_bert')
    #predict('./ghidra/unstrip/test/3b8aff7406b2035735d5321e2dcbf046.pkl')