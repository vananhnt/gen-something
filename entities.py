from preprocess import to_token, to_token_asm, extract_decompile, extract_asm
from tqdm import tqdm
from transformers import RobertaTokenizer

class Example(object):
    """A single training/test example."""
    def __init__(self,idx,source,target, vex=None):
        self.idx = idx
        self.source = source
        self.target = target
        self.vex = vex

# for spoc dataset
# def read_examples(pg, f_range=None):
#     """Read examples from program generator."""
#     examples=[]
#     if f_range is not None:
#         for idx, program_dict in tqdm(enumerate(pg), total=len(f_range)):
#             codes, comments = program_dict['program_str'].strip().split('\n'), program_dict['comments']
#             if len(codes) != len(comments):
#                 raise Exception('line of code does not match comments!')
#             for code, comment in zip(codes, comments): # for each lines
#                 train_src_toks, train_tgt_toks = to_token(comment), to_token(code)
#                 src =' '.join(train_src_toks).replace('\n',' ')
#                 tgt =' '.join(train_tgt_toks).replace('\n',' ')
#                 examples.append(Example(idx = idx,source=src,target=tgt))
#     else:
#         program_dict = pg
#         # Read examples from program_dict
#         codes, comments = program_dict['program_str'].strip().split('\n'), program_dict['comments']
#         if len(codes) != len(comments):
#             raise Exception('line of code does not match comments!')
#         for code, comment in zip(codes, comments): # for each lines
#             train_src_toks, train_tgt_toks = to_token(comment), to_token(code)
#             src =' '.join(train_src_toks).replace('\n',' ')
#             tgt =' '.join(train_tgt_toks).replace('\n',' ')
#             examples.append(Example(idx = program_dict['idx'],source=src,target=tgt))
#     return examples

def read_examples(dataset, idx = None):
    """Read examples from dictionary id: [source, target]. or dir"""
    examples=[]
    
    if idx is not None and type(dataset) == dict:
        row = dataset[idx]
        if len(row) > 1:
                source, target = row[0], row[1]
                train_src_toks, train_tgt_toks = to_token(source), to_token(target)
                src =' '.join(train_src_toks).replace('\n',' ')
                tgt =' '.join(train_tgt_toks).replace('\n',' ')
                examples.append(Example(idx = idx,source=src,target=tgt))
    else:
        sample_dict = extract_decompile(dataset)
        for idx in tqdm(sample_dict, total = len(sample_dict)):
            row = sample_dict[idx]
            if len(row) > 1:
                source, target = row[0], row[1]
                train_src_toks, train_tgt_toks = to_token(source), to_token(target)
                src =' '.join(train_src_toks).replace('\n',' ')
                tgt =' '.join(train_tgt_toks).replace('\n',' ')
                examples.append(Example(idx = idx,source=src,target=tgt))
    return examples

def read_examples_asm(dataset, idx = None):
    """Read examples from dictionary id: [source, target]. or dir"""
    examples=[]
    
    if idx is not None and type(dataset) == dict:
        row = dataset[idx]
        if len(row) > 1:
                source, target, vex = row[0], row[1], row[2]
                train_src_toks, train_tgt_toks, train_vex_toks = to_token(source), to_token(target), to_token(vex)
                src =' '.join(train_src_toks).replace('\n',' ')
                tgt =' '.join(train_tgt_toks).replace('\n',' ')
                vx = ' '.join(train_vex_toks).replace('\n',' ')
                examples.append(Example(idx = idx,source=src,target=tgt,vex=vx))
    else:
        sample_dict = extract_asm(dataset)
        for idx in tqdm(sample_dict, total = len(sample_dict)):
            row = sample_dict[idx]
            if len(row) > 1:
                source, target, vex = row[0], row[1], row[2]
                train_src_toks, train_tgt_toks, train_vex_toks = to_token(source), to_token(target), to_token(vex)
                src =' '.join(train_src_toks).replace('\n',' ')
                tgt =' '.join(train_tgt_toks).replace('\n',' ')
                vx = ' '.join(train_vex_toks).replace('\n',' ')
                examples.append(Example(idx = idx,source=src,target=tgt,vex=vx))
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        

def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    max_source_length=256
    max_target_length =20
    if args.with_vex:
        for example_index, example in enumerate(examples):
            #source
            source_tokens = tokenizer.tokenize(example.source)[:int(max_source_length/2)-2]
            vex_tokens = source_tokens+[tokenizer.sep_token]+tokenizer.tokenize(example.vex)
            concat_tokens = vex_tokens[:max_source_length-2]
            source_tokens =[tokenizer.cls_token]+concat_tokens+[tokenizer.sep_token]
            source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
            
            source_mask = [1] * (len(source_tokens))
            padding_length = max_source_length - len(source_ids)
            source_ids+=[tokenizer.pad_token_id]*padding_length
            source_mask+=[0]*padding_length

            #target
            if stage=="test":
                target_tokens = tokenizer.tokenize("None")
            else:
                target_tokens = tokenizer.tokenize(example.target)[:max_target_length-2]
            target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            target_mask = [1] *len(target_ids)
            padding_length = max_target_length - len(target_ids)
            target_ids+=[tokenizer.pad_token_id]*padding_length
            target_mask+=[0]*padding_length          

            features.append(InputFeatures(example_index,source_ids,target_ids,source_mask,target_mask))
    else:
        for example_index, example in enumerate(examples):
            #source
            source_tokens = tokenizer.tokenize(example.source)[:max_source_length-2]
            source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
            source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
            source_mask = [1] * (len(source_tokens))
            padding_length = max_source_length - len(source_ids)
            source_ids+=[tokenizer.pad_token_id]*padding_length
            source_mask+=[0]*padding_length

            #target
            if stage=="test":
                target_tokens = tokenizer.tokenize("None")
            else:
                target_tokens = tokenizer.tokenize(example.target)[:max_target_length-2]
            target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            target_mask = [1] *len(target_ids)
            padding_length = max_target_length - len(target_ids)
            target_ids+=[tokenizer.pad_token_id]*padding_length
            target_mask+=[0]*padding_length          

            features.append(InputFeatures(example_index,source_ids,target_ids,source_mask,target_mask))
    return features

if __name__ == '__main__':
    exm = read_examples('./ghidra/unstrip/')
    #for example in exm:
        #print(example.source)
        