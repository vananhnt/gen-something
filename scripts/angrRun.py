import angr
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, to_agraph
from angrutils import *
from graphviz import Digraph, Source
from collections import deque
import pyvex
import claripy
import os
import time
import pyvex
import re
from tqdm import tqdm
import glob
import pickle

import networkx as nx
import matplotlib.pyplot as plt

root_dir = "/home/va/Downloads/cross-compile-dataset-master/bin/"

ARM_REGS_LIST = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'fp', 'ip', 'sp', 'lr', 'pc']
stack = deque()

def graph_rendering(filepath, _file_out):
    proj = angr.Project(filepath, load_options={'auto_load_libs':False})

    state = proj.factory.entry_state()
    
    state.options |= {angr.sim_options.CONSTRAINT_TRACKING_IN_SOLVER}
    state.options -= {angr.sim_options.COMPOSITE_SOLVER}
    #state.options |= {angr.sim_options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS}
    state.options |= {angr.sim_options.SYMBOL_FILL_UNCONSTRAINED_MEMORY}
    # simgr = proj.factory.simgr(start_state)

    main = proj.loader.main_object.get_symbol("main")
    start_state = proj.factory.blank_state(addr=main.rebased_addr)
        #cfg = proj.analyses.CFGEmulated(keep_state=True)
    cfg = proj.analyses.CFGFast()
    print("It has %d nodes and %d edges" % (len(cfg.graph.nodes()), len(cfg.graph.edges())))
    
def fill_symbolic_regs(state):
    state.regs.r0 = claripy.BVS('r0', 32)
    state.regs.r1 = claripy.BVS('r1', 32)
    state.regs.r2 = claripy.BVS('r2', 32)
    state.regs.r3 = claripy.BVS('r3', 32)
    state.regs.r4 = claripy.BVS('r4', 32)
    state.regs.r5 = claripy.BVS('r5', 32)
    state.regs.r6 = claripy.BVS('r6', 32)
    state.regs.r7 = claripy.BVS('r7', 32)
    state.regs.r8 = claripy.BVS('r8', 32)
    state.regs.r9 = claripy.BVS('r9', 32)
    state.regs.r10 = claripy.BVS('r10', 32)
    # state.regs.r11 = claripy.BVS('fp', 32)
    # state.regs.r12 = claripy.BVS('ip', 32)
    # state.regs.r13 = claripy.BVS('sp', 32)
    # state.regs.r14 = claripy.BVS('lr', 32)
    #state.regs.r15 = claripy.BVS('pc', 32)

def sym_exe(filepath):
    func_map = dict()
    p = angr.Project(filepath, load_options={'auto_load_libs':False})
    # main = proj.loader.main_object.get_symbol("main")
    # start = 0x400ee0 
    state = p.factory.blank_state(addr=p.entry)

    state.options |= {angr.sim_options.CONSTRAINT_TRACKING_IN_SOLVER}
    state.options -= {angr.sim_options.COMPOSITE_SOLVER}
    #state.options |= {angr.sim_options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS}
    state.options |= {angr.sim_options.SYMBOL_FILL_UNCONSTRAINED_MEMORY}
    
    cfg = p.analyses.CFGFast()
    
    #entry_func = cfg.kb.functions[p.entry]
    
    #print(entry_func.transition_graph.nodes())
    count = 0
    for addr, func in p.kb.functions.items():
        # print("===============")
        # print("func:", func.name)
        subcfg = cfg.functions.get(addr).transition_graph
        # subcfg
        # nx.draw(sub)
        # p.show()
        func_text = ''
        for node in subcfg.nodes():
            #if node.addr >= start_address and node.addr <= end_address:
                #target_blocks.add(node)
                #pass
                for stmt in p.factory.block(node.addr).vex.statements:
                    func_text += normalise(str(stmt))
                    #func_text += str(stmt)
                # stmt_list = [str(stmt.tag).replace('Ist_', '') for stmt in p.factory.block(node.addr).vex.statements]
                # print(stmt_list)    
                    #if isinstance(stmt, pyvex.IRStmt.Exit):
                        # print("Condition:",)
                        # stmt.guard.pp()
                        # print("")
                        # print("Target:",)
                        # stmt.dst.pp()
                        # print("")
                        # print(stmt)
        func_map[func.name] = func_text.replace('  ', ' ')
    return func_map
    
filepath = root_dir + 'static/gcc/o2/chroot'

def normalise(IRStmt):
    result = ''
    if 'if' in IRStmt:
        result = 'if_'+ re.search('(?<=\{\s)(.*?)\(', IRStmt).group(1)
    elif bool(re.match('t(.*?)\)', IRStmt)):
        result = re.search('=\s(.*?)\(', IRStmt).group(1)
    elif bool(re.match('[a-z|A-Z]*\(.*\)\s=', IRStmt)):
        result = re.search('(?<=)(.*?)\(', IRStmt).group(1)
    
    hx = re.search('0[xX][0-9a-fA-F]+', IRStmt)
    if hx and result != '':
        concat = str(int(hx.group(0), 16)) if int(hx.group(0), 16) < 5000 else 'MEM' 
        result += ' ' + concat
    return result + '\n' if result != '' else ''

count = 0
for filename in tqdm(glob.iglob(root_dir + '**/**', recursive=True)):
    if os.path.isfile(filename) and '.' not in filename:
        f = filename.split('/')[-1]
        prefix = 'bin'
        isdynamic = 'dynamic' if 'dynamic' in filename else 'static' 
        newfile = f+'_'+prefix+'_'+isdynamic
        
        angr_map = sym_exe(filename)

        pkl_file = './ghidra/cross/'+newfile+'_decompiled.pkl'
        
        funcmap = pickle.load(open(pkl_file, "rb" ))
        for k in funcmap:
            funcname = k
            (signature, dec_func, disasm_result, byte, addr) = funcmap[k]
            if funcname.replace('FUN', 'sub') in angr_map.keys():  
                #print(angr_map[funcname.replace('FUN','sub')])
                pass
            else:
                print(funcname)
        start_time = time.time()
        print("--- %s seconds ---" % (time.time() - start_time))
        if count < 2: count+=1
        else: break