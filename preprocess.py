import pandas as pd
import os
import re
import glob
import pickle
from tqdm import tqdm
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

def to_token(s):
    #values = [plain2_tok(t.value, t.kind) for t in tokenizer(s, new_line=False) if t.kind != 'whitespace']
    values = tokenizer.tokenize(s)
    #result = ' '.join([v for v in values if v is not None])
    return values

def stripcomments(text):
    return re.sub('//.*?\n|/\*.*?\*/', '', text, flags=re.S)


# TODO: Need to add sampleID when process ghidra, not here
def load_dataframe(dataset_dir):
    fileID = 0
    fid = 0
    number_files = len(os.listdir(dataset_dir)) # dir is your directory path
    
    row_list = []
    
    header = ['id', 'funcname', 'signature', 'decompiled', 'disassembly', 'bytes', 'address', 'sampleID']

    for filename in tqdm(glob.iglob(dataset_dir + '/*.pkl'), total=number_files):
        funcmap = pickle.load(open(filename, "rb" ))
        for k in funcmap:
            funcname = k
            (signature, dec_func, disasm_result, byte, addr) = funcmap[k]      
            func_dict = {'id': fid, 'funcname': funcname, 
                         'signature': signature, 'decompiled': stripcomments(dec_func), 
                         'disassembly': disasm_result, 'bytes': byte, 
                         'address':addr, 'sampleID':fileID,
                        }
            row_list.append(func_dict)
            fid += 1
        fileID +=1
    df = pd.DataFrame(row_list, columns = header)               
    return df

def extract_decompile(dataset_dir):
    # return dict : idx, source, target
    df = load_dataframe(dataset_dir)
    samples = df.set_index('id')[['decompiled', 'funcname']].T.to_dict('list')
    return samples

def args_parser(parser):
    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )       
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    args = parser.parse_args()
    return args