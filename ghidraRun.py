import os
import random
import shutil
import csv
import pickle
import glob
import subprocess
from tqdm import tqdm
import math

root_dir = "/home/va/Documents/arm_full"
ghidra_dir = "/home/va/ghidra_9.0.1"
ghidra_out = './ghidra/unstrip'
#add remove comments

def save_decompile():
    for filename in tqdm(glob.iglob(root_dir + '/*', recursive=True)):
        print(filename.split('/')[-1])
        cmd =  '/home/va/ghidra_9.0.1/support/analyzeHeadless /home/va run_1 -import {} -scriptPath /home/va -postScript ./ghidra/decompiler.py ./ghidra/out/{}_decompiled.pkl'.format(filename, filename.split('/')[-1])
        #cmd =  '/home/va/ghidra_9.0.1/support/analyzeHeadless /home/va test_1 -process {} -scriptPath /home/va -postScript ./ghidra/decompiler.py ./ghidra/out/{}_decompiler.pkl'.format(filename.split('/')[-1], filename.split('/')[-1])
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        print(str(p).replace('\\n', '\n'))

def read_funcmap():
    fileID = 0
    fid = 0
    number_files = len(os.listdir(('./ghidra/unstrip'))) # dir is your directory path
    
    with open('./test.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'funcname', 'signature', 'decompiled', 'disassembly', 'bytes', 'address', 'sampleID'])
        for filename in tqdm(glob.iglob('./ghidra/unstrip/*.pkl'), total=number_files):
            funcmap = pickle.load(open(filename, "rb" ))
            for k in funcmap:
                funcname = k
                (signature, dec_func, disasm_result, byte, addr) = funcmap[k]      
                csvwriter.writerow([fid, funcname, signature, dec_func, disasm_result, byte, addr, fileID])
                fid += 1
            fileID +=1

def split_train_test():
    source=ghidra_out
    dev='./ghidra/dev'
    test='./ghidra/test'
    total = len(os.listdir(('./ghidra/unstrip')))
    
    no_train, no_dev =  math.floor(0.7*total), math.floor(0.15*total), 
    no_test = total - no_train - no_dev    
    
    print("[" + str(total) +" files total ]")

    for i in range(no_dev):
        #Variable random_file stores the name of the random file chosen
        random_file=random.choice(os.listdir(source))
        source_file="%s/%s"%(source,random_file)
        dest_file=dev
        shutil.move(source_file,dest_file)
    print("[" + str(no_dev) + " files moved ]")
    
    for i in range(no_test):
        #Variable random_file stores the name of the random file chosen
        random_file=random.choice(os.listdir(source))
        source_file="%s/%s"%(source,random_file)
        dest_file=test
        shutil.move(source_file,dest_file)
    print("[" + str(no_test) +" files moved ]")
          
if __name__ == '__main__':
    #save_decompile()
    #read_funcmap()
    split_train_test()