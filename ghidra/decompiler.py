from ghidra.app.decompiler import DecompInterface
import pickle

# `currentProgram` or `getScriptArgs` function is contained in `__main__`
# actually you don't need to import by yourself, but it makes much "explicit"
import __main__ as ghidra_app

# distance from address to byte
# |<---------->|
# 00401000      89 c8      MOV EAX, this
DISTANCE_FROM_ADDRESS_TO_BYTE = 15

# distance from byte to instruction
#               |<------->|
# 00401000      89 c8      MOV EAX, this
DISTANCE_FROM_BYTE_TO_INST = 15

# output format of instructions
MSG_FORMAT = ' {{addr:<{0}}} {{byte:<{1}}} {{inst}}\n'.format(
    DISTANCE_FROM_ADDRESS_TO_BYTE,
    DISTANCE_FROM_ADDRESS_TO_BYTE+DISTANCE_FROM_BYTE_TO_INST 
)

COL_FORMAT = '{}\n'

def unoverflow(x):
    return (abs(x) ^ 0xff) + 1


def to_hex(integer):
    return '{:02x}'.format(integer)


def _get_function_signature(func):
    # get function signature
    calling_conv = func.getDefaultCallingConventionName()
    params = func.getParameters()

    return '\n{calling_conv} {func_name}({params})\n'.format(
        calling_conv=calling_conv,
        func_name=func.getName(),
        params=', '.join([str(param).replace('[', '').replace(']', '').split('@')[0] for param in params]))

def _get_instructions(func):
    instructions = ''

    # get instructions in function
    func_addr = func.getEntryPoint()
    insts = ghidra_app.currentProgram.getListing().getInstructions(func_addr, True)

    # process each instruction
    for inst in insts:
        if ghidra_app.getFunctionContaining(inst.getAddress()) != func:
            break

        instructions += MSG_FORMAT.format(
            addr=inst.getAddressString(True, True),
            byte=' '.join([to_hex(b) if b >= 0 else to_hex(unoverflow(b)) for b in inst.getBytes()]),
            inst=inst
        )
    return instructions

def get_inst_content(func):
    instructions = ''

    # get instructions in function
    func_addr = func.getEntryPoint()
    insts = ghidra_app.currentProgram.getListing().getInstructions(func_addr, True)

    # process each instruction
    for inst in insts:
        if ghidra_app.getFunctionContaining(inst.getAddress()) != func:
            break

        instructions += COL_FORMAT.format(inst)
    return instructions

def get_byte(func):
    instructions = ''
    # get instructions in function
    func_addr = func.getEntryPoint()
    insts = ghidra_app.currentProgram.getListing().getInstructions(func_addr, True)

    # process each instruction
    for inst in insts:
        if ghidra_app.getFunctionContaining(inst.getAddress()) != func:
            break
        instructions += COL_FORMAT.format(' '.join([to_hex(b) if b >= 0 else to_hex(unoverflow(b)) for b in inst.getBytes()]))
    return instructions

def get_address(func):
    instructions = ''
    func_addr = func.getEntryPoint()
    insts = ghidra_app.currentProgram.getListing().getInstructions(func_addr, True)

    # process each instruction
    for inst in insts:
        if ghidra_app.getFunctionContaining(inst.getAddress()) != func:
            break
        instructions += COL_FORMAT.format(inst.getAddressString(True, True))
    return instructions

def disassemble_func(func):
    '''disassemble given function, and returns as string.
    Args:
        func (ghidra.program.model.listing.Function): function to be disassembled
    Returns:
        string: disassembled function with function signature and instructions
    '''
    return _get_instructions(func)
    #return  _get_function_signature(func) + _get_instructions(func)


def disassemble(program):
    '''disassemble given program.
    Args:
        program (ghidra.program.model.listing.Program): program to be disassembled
    Returns:
        string: all disassembled functions 
    '''

    disasm_result = ''

    # enum functions and disassemble
    funcs = program.getListing().getFunctions(True)
    for func in funcs:
        disasm_result += disassemble_func(func)

    return disasm_result


class Decompiler:
    '''decompile binary into psuedo c using Ghidra API.
    Usage:
        >>> decompiler = Decompiler()
        >>> psuedo_c = decompiler.decompile()
        >>> # then write to file
    '''

    def __init__(self, program=None, timeout=None):
        '''init Decompiler class.
        Args:
            program (ghidra.program.model.listing.Program): target program to decompile, 
                default is `currentProgram`.
            timeout (ghidra.util.task.TaskMonitor): timeout for DecompInterface::decompileFunction
        '''

        # initialize decompiler with current program
        self._decompiler = DecompInterface()
        self._decompiler.openProgram(program or ghidra_app.currentProgram)

        self._timeout = timeout
    
    def decompile_func(self, func):
        '''decompile one function.
        Args:
            func (ghidra.program.model.listing.Function): function to be decompiled
        Returns:
            string: decompiled psuedo C code
        '''

        # decompile
        dec_status = self._decompiler.decompileFunction(func, 0, self._timeout)
        # check if it's successfully decompiled
        if dec_status and dec_status.decompileCompleted():
            # get psuedo c code
            dec_ret = dec_status.getDecompiledFunction()
            if dec_ret:
                return dec_ret.getC()

    def decompile(self):
        '''decompile all function recognized by Ghidra.
        Returns:
            string: decompiled all function as psuedo C
        '''

        dec_map = dict()

        # all decompiled result will be joined
        psuedo_c = ''

        # enum all functions and decompile each function
        funcs = ghidra_app.currentProgram.getListing().getFunctions(True)
        for func in funcs:
            dec_func = self.decompile_func(func)
            #disasm_result = disassemble_func(func)
            signature = _get_function_signature(func)
            disasm_result = get_inst_content(func)
            addr = get_address(func)
            byte = get_byte(func)

            if dec_func and disasm_result:
                dec_map[str(func)] = (signature, dec_func, disasm_result, byte, addr)
                #psuedo_c += dec_func

        return dec_map


def run():

    # getScriptArgs gets argument for this python script using `analyzeHeadless`
    args = ghidra_app.getScriptArgs()
    if len(args) > 1:
        print('[!] wrong parameters, see following\n\
Usage: ./analyzeHeadless <PATH_TO_GHIDRA_PROJECT> <PROJECT_NAME> \
-process|-import <TARGET_FILE> [-scriptPath <PATH_TO_SCRIPT_DIR>] \
-postScript|-preScript decompile.py <PATH_TO_OUTPUT_FILE>')
        return
    
    # if no output path given, 
    # <CURRENT_PROGRAM>_decompiled.c will be saved in current dir
    if len(args) == 0:
        cur_program_name = ghidra_app.currentProgram.getName()
        output = '{}_decompiled.pkl'.format(''.join(cur_program_name.split('.')[:-1]))
    else:
        output = args[0]

    # do decompile
    decompiler = Decompiler()
    psuedo_map = decompiler.decompile()

    # save to file
    # with open(output, 'w') as fw:
    #     fw.write(psuedo_c)
    #     print('[*] success. save to -> {}'.format(output))
    pickle.dump(psuedo_map, open(output, "wb" ) )

if __name__ == '__main__':
    run()
