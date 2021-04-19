from model import Seq2Seq
from tqdm import tqdm, trange
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig,AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from preprocess import build_program
from entities import convert_examples_to_features, read_examples
import argparse
from itertools import cycle
import torch
import torch.nn as nn
import logging
import numpy as np
import os
import bleu
import pickle
import random


# learning_rate=5e-5
# weight_decay=0.0
# adam_epsilon=1e-8
# train_batch_size=8
# eval_batch_size=64
# batch_size=64
# beam_size=10
# source_length=36
# target_length=36
# eval_steps=1000
# train_steps=1000 #50000 
# n_gpu=1
# gradient_accumulation_steps=1
# do_eval=True
# dev_file='dev.out'
# eval_file= 'eval.out'


default_output_dir = './model/'
train_dataset='./spoc/spoc-train.frange'
dev_dataset = './spoc/spoc-testp.frange'
test_dataset = './spoc/spoc-testw.frange'

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, eval_loss, best_loss):
    #save last checkpoint
    last_output_dir = os.path.join(default_output_dir, 'checkpoint-last')
    if not os.path.exists(last_output_dir):
        os.makedirs(last_output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)                    
    if eval_loss<best_loss:
        logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
        logger.info("  "+"*"*20)
        best_loss=eval_loss
        # Save best checkpoint for best ppl
        output_dir = os.path.join(default_output_dir, 'checkpoint-best-ppl')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model 
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)  

def train_batch(model, batch, tr_loss, step, dev_dataset, optimizer, bar):
    source_ids,source_mask,target_ids,target_mask = batch
    loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)
    loss = loss.mean()
    tr_loss += loss.item()
    train_loss=round(tr_loss*10/(step+1),4)
    bar.set_description("loss {}".format(train_loss/(step+1)))
    loss.backward()
         
    if (step + 1) % 10 == 0: 
        #Update parameters
        optimizer.step()
        optimizer.zero_grad()
    
        #Eval model
        tr_loss = 0
        train_steps = 0
        best_loss=1e6
        if 'dev_loss' in dev_dataset:
            eval_examples,eval_data=dev_dataset['dev_loss']
        else:
            pg,f_range = build_program('eval')
            eval_examples = read_examples(pg, f_range)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, stage='dev')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)      
            eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)   
            dev_dataset['dev_loss']=eval_examples,eval_data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size= batch_size)
            
            # Start Evaluating
            model.eval()
            eval_loss,tokens_num = 0,0
            for batch in eval_dataloader:
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,target_ids,target_mask = batch                  

                with torch.no_grad():
                    _,loss,num = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)     
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_loss': round(np.exp(eval_loss),5), 'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   
                
                # Save checkpoint
                save_checkpoint(model, eval_loss, best_loss)
    return train_loss

def evaluate(model, eval_examples, eval_dataloader, idx): 
        model.eval()
        p=[]
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask= batch                  
            with torch.no_grad():
                preds = model(source_ids=source_ids,source_mask=source_mask)  
                for pred in preds:
                    t=pred[0].cpu().numpy()
                    t=list(t)
                    if 0 in t:
                        t=t[:t.index(0)]
                    text = tokenizer.decode(t,clean_up_tokenization_spaces=True)
                    p.append(text)
        model.train()
        predictions=[]
        with open(os.path.join(default_output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(default_output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
            for ref,gold in zip(p,eval_examples):
                predictions.append(str(gold.idx)+'\t'+ref)
                f.write(str(gold.idx)+'\t'+ref+'\n')
                f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

        (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(default_output_dir, "test_{}.gold".format(idx))) 
        dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
        logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
        logger.info("  "+"*"*20)    

def train_epoches(model, train_examples, train_dataloader, optimizer):
    num_train_optimization_steps = train_steps
    #Start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num epoch = %d", num_train_optimization_steps*train_batch_size//len(train_examples))
        
    model.train()
    dev_dataset={}
    nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0,0,0,0,0,1e6 
    bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
    train_dataloader=cycle(train_dataloader)
    eval_flag = True
    # Train epoches
    for step in bar:
        batch = next(train_dataloader)
        batch = tuple(t.to(device) for t in batch)
        train_loss = train_batch(model, batch, tr_loss, step, dev_dataset, optimizer, bar)

def main():
    parser = argparse.ArgumentParser()
    
    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
#     parser.add_argument("--train_filename", default=None, type=str, 
#                         help="The train filename. Should contain the .jsonl files for this task.")
#     parser.add_argument("--dev_filename", default=None, type=str, 
#                         help="The dev filename. Should contain the .jsonl files for this task.")
#     parser.add_argument("--test_filename", default=None, type=str, 
#                         help="The test filename. Should contain the .jsonl files for this task.")  
    
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
    
#     config = RobertaConfig.from_pretrained("microsoft/codebert-base")
#     #build model
#     encoder = RobertaModel.from_pretrained("microsoft/codebert-base")    
#     decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
#     decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
#     model=Seq2Seq(encoder=encoder,decoder=decoder,config=config, beam_size=beam_size,max_length=256, sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
#     if args.load_model_path is not None:
#         logger.info("reload model from {}".format(args.load_model_path))
#         model.load_state_dict(torch.load(args.load_model_path))
    
#     model.to(device)
    
#     pg,f_range = build_program('train')
#     train_examples = read_examples(pg, f_range)
#     train_features = convert_examples_to_features(train_examples, tokenizer,stage='train')
#     all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
#     all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
#     all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
#     all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
#     train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
#     train_sampler = RandomSampler(train_data)
#     train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
   
#     if args.do_train:
#         no_decay = ['bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]
#         optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
#         train_epoches(model, train_examples, train_dataloader,  optimizer)
    
#     if args.do_test:
#         pg, f_range = build_program('test')
#         for idx, program_dict in tqdm(enumerate(pg), total=len(f_range)):
#             eval_examples = read_examples(program_dict)
#             eval_features = convert_examples_to_features(eval_examples, tokenizer,stage='test')
#             all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
#             all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
#             eval_data = TensorDataset(all_source_ids,all_source_mask)   

#             # Calculate bleu
#             eval_sampler = SequentialSampler(eval_data)
#             eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
#             evaluate(model, eval_examples, eval_dataloader, idx)
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    
    args.output_dir = default_output_dir
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = RobertaConfig.from_pretrained("microsoft/codebert-base")
    
    #build model
    encoder = RobertaModel.from_pretrained("microsoft/codebert-base")    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        pg,f_range = build_program('train')
        train_examples = read_examples(pg, f_range)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
        train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
    
        
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps*args.train_batch_size//len(train_examples))
        

        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader)
        eval_flag = True
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch
            loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)
            
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                #Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True
                
            if args.do_eval and ((global_step + 1) %args.eval_steps == 0) and eval_flag:
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    pg,f_range = build_program('eval')
                    eval_examples = read_examples(pg, f_range)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)      
                    eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)   
                    dev_dataset['dev_loss']=eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                
                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,target_ids,target_mask = batch                  

                    with torch.no_grad():
                        _,loss,num = model(source_ids=source_ids,source_mask=source_mask,
                                           target_ids=target_ids,target_mask=target_mask)     
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                #Pring loss of dev dataset    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   
                
                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)                    
                if eval_loss<best_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    best_loss=eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)  
                            
                            
                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    pg,f_range = build_program('eval')
                    eval_examples = read_examples(pg, f_range)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
                    eval_data = TensorDataset(all_source_ids,all_source_mask)   
                    dev_dataset['dev_bleu']=eval_examples,eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

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
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)    
                if dev_bleu>best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
               
    if args.do_test:
        pg, f_range = build_program('test')
        for idx, program_dict in tqdm(enumerate(pg), total=len(f_range)):
            logger.info("Test file: {}".format(idx))
            eval_examples = read_examples(program_dict)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
            eval_data = TensorDataset(all_source_ids,all_source_mask)   

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval() 
            p=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
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
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(str(gold.idx)+'\t'+ref)
                    f.write(str(gold.idx)+'\t'+ref+'\n')
                    f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

            (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test_{}.gold".format(idx))) 
            dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  "+"*"*20)    
    
    
if __name__ == "__main__":
    main()