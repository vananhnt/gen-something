from tqdm import tqdm
from transformers import RobertaTokenizer
import os

citation_symbols = {'start-quote': '\"', 'start-char':'\'',
                    'end-quote': '\"', 'end-char': '\''}
WHITE_SPACE = '!WHITE_SPACE!'
NEXT_LINE = '!Next_LINE!'
debug = False
data_dir = './spoc/data/'
train_dir = './spoc/onmt/'
program_dir = data_dir + 'programs/'
comment_dir = data_dir + 'comments/'
comment_revealed_dir = data_dir + 'comments_revealed/'
indent_dir = data_dir + 'indent/'
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# mapping from a token to a token for onmt
def plain2_tok(val, kind):
    if '\n' in val:
        return NEXT_LINE
    if kind in citation_symbols:
        return kind
    if kind == 'in-quote' or kind == 'in-char':
        if ' ' in val:
            return WHITE_SPACE
        else:
            return val
    if kind != 'whitespace':
        return val
    return None


def to_token(s):
    #values = [plain2_tok(t.value, t.kind) for t in tokenizer(s, new_line=False) if t.kind != 'whitespace']
    values = tokenizer.tokenize(s)
    #result = ' '.join([v for v in values if v is not None])
    return values

class Program_generator:
    def __init__(self, program_dir):
        self.program_dir = program_dir
        self.f_names = sorted([f[:-3] for f in os.listdir(self.program_dir)])
        self.f_name2idx = {f_name:idx for idx, f_name in enumerate(self.f_names)}
        self.num_programs = len(self.f_names)

    def indexed_program(self, idx, all_info):
        result = {'idx': idx}

        f_name = self.f_names[idx]
        result['f_name'] = f_name

        with open(self.program_dir + f_name + '.cc') as in_file:
            program_str = in_file.read()
        result['program_str'] = program_str

        program_by_line = program_str.strip().split('\n')
        result['program_by_line'] = program_by_line

        #p = Program(program_str)
        if not all_info:
            return p

        with open(comment_dir + f_name + '.txt') as in_file:
            comments = in_file.read().split('\n')
        result['comments'] = comments

        with open(comment_revealed_dir + f_name + '.txt') as in_file:
            nan_revealed = in_file.read().split('\n')
        result['nan_revealed'] = nan_revealed

        with open(indent_dir + f_name + '.txt') as in_file:
            indent = in_file.read().split('\n')
        result['indent'] = indent
    
        return result
    
    def program_generator(self, all_info=False, file_range=None, shuffle=False, seed=None):
        if file_range is None:
            file_idx_range = range(self.num_programs)
        elif type(file_range) == str:
            file_idx_range = [self.f_name2idx[file_range]]
        else:
            file_idx_range = [self.f_name2idx[f] for f in file_range]
        file_idx_range = sorted(list(file_idx_range))
        if shuffle or seed is not None:
            if seed is not None:
                random.seed(seed)
            random.shuffle(file_idx_range)
        def gen():
            for idx in file_idx_range:
                yield self.indexed_program(idx, all_info)
        return gen()

def build_program(typef):
    if typef == 'train':
        f = './spoc/spoc-train.frange'
        with open(f, 'r') as in_file: # train_dir
            s = in_file.read().strip()
        f_range = s.split('\n')
    elif typef == 'eval':
        f = './spoc/spoc-testw.frange'
        with open(f, 'r') as in_file: # eval_dir
            s = in_file.read().strip()
        f_range = s.split('\n')
    else:
        with open('./spoc/spoc-testw.frange', 'r') as in_file_1: 
            s_test = in_file_1.read().strip()
        f_range = s_test.split('\n')
       
        with open('./spoc/spoc-testp.frange', 'r') as in_file_2:
            s_eval = in_file_2.read().strip()
        f_range.extend(s_eval.split('\n'))
    
    comment_key = 'comments'
    assert comment_key in ('comments', 'nan_revealed')
    prefix = 'programs'
    program_dir = data_dir + prefix
    
    pg = Program_generator(program_dir=program_dir + '/').program_generator(all_info=True, file_range=f_range)
    return pg, f_range
#     for idx, program_dict in tqdm(enumerate(pg), total=len(f_range)):
#         codes, comments = program_dict['program_str'].strip().split('\n'), program_dict[comment_key]
#         if len(codes) != len(comments):
#             raise Exception('line of code does not match comments!')
#         for code, comment in zip(codes, comments):
#             train_src_toks, train_tgt_toks = to_token(comment), to_token(code)

if __name__ == '__main__':
    build_program()