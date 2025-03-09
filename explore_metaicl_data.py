import json
import os
import matplotlib.pyplot as plt
from functools import partial
from tokenizers.processors import TemplateProcessing
from multiprocessing import Pool
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import AutoTokenizer
import pprint
from collections import Counter, defaultdict
from datasets import load_dataset


hr_lr_split = json.load(open('metaicl_config/hr_to_lr.json'))
HR_TASKS = set(hr_lr_split['train'])
LR_TASKS = set(hr_lr_split['test'])
HR_TASKS.remove('gigaword')

# load dataset
dataset = load_dataset("bigheiniuJ/EvalMetaICLAll")
train_split = dataset['meta_train']
test_split = dataset['meta_eval_100shot']
train_tasks = set(list(train_split['task']))
test_tasks = set(list(test_split['task']))
assert HR_TASKS.issubset(train_tasks)
assert LR_TASKS.issubset(test_tasks)

# get classification tasks
task_to_is_classification = {}
for task in test_tasks:
    path = os.path.join(f'metaicl_config/tasks/{task}.json')
    is_classification = json.load(open(path, 'r'))['task_type'] == 'classification'
    task_to_is_classification[task] = is_classification

# check if classification tasks have options
task_to_indices = defaultdict(list)
for idx, task in enumerate(test_split['task']):
    task_to_indices[task].append(idx)

def check_data_i(i):
    data_i = test_split[i]
    assert len(data_i['options']) > 1

for task in test_tasks:
    if task_to_is_classification[task]:
        with Pool(32) as p:
            task_lens = p.map(check_data_i, task_to_indices[task])

# finally, filter train and test split to HR and LR tasks
train_split = train_split.filter(lambda example: example["task"] in HR_TASKS)
test_split = test_split.filter(lambda example: example["task"] in LR_TASKS)








# only keep test split's one of five seeds
test_split = test_split.filter(lambda example: example['seed'] == '100')
# unused columns
train_split = train_split.remove_columns("seed")
train_split = train_split.remove_columns("split")
test_split = test_split.remove_columns("seed")
test_split = test_split.remove_columns("split")
print('num train rows', len(train_split))
print('num test rows', len(test_split))
print()

print('train task sample count')
pprint.pprint(Counter(train_split['task']))
print()
print('test task sample count')
pprint.pprint(Counter(test_split['task']))
print()

# number of hr tasks 61
# number of lr tasks 26
# missing hr gigaword
# num train rows 940246
# num test rows 18190

# train task sample count
# Counter({'squad-no_context': 16384,
#          'wiqa': 16384,
#          'kilt_fever': 16384,
#          'yelp_polarity': 16384,
#          'wiki_qa': 16384,
#          'wikisql': 16384,
#          'art': 16384,
#          'race-middle': 16384,
#          'hotpot_qa': 16384,
#          'search_qa': 16384,
#          'kilt_zsre': 16384,
#          'tweet_eval-emoji': 16384,
#          'race-high': 16384,
#          'glue-qqp': 16384,
#          'anli': 16384,
#          'kilt_trex': 16384,
#          'xsum': 16384,
#          'wino_grande': 16384,
#          'cosmos_qa': 16384,
#          'kilt_nq': 16384,
#          'yelp_review_full': 16384,
#          'swag': 16384,
#          'glue-sst2': 16384,
#          'lama-conceptnet': 16384,
#          'kilt_hotpotqa': 16384,
#          'squad-with_context': 16384,
#          'biomrc': 16384,
#          'dbpedia_14': 16384,
#          'superglue-record': 16384,
#          'discovery': 16384,
#          'hate_speech_offensive': 16384,
#          'social_i_qa': 16384,
#          'scitail': 16384,
#          'yahoo_answers_topics': 16384,
#          'tab_fact': 16384,
#          'ag_news': 16384,
#          'kilt_ay2': 16384,
#          'quoref': 16384,
#          'imdb': 16384,
#          'tweet_eval-sentiment': 16384,
#          'ade_corpus_v2-classification': 16384,
#          'emo': 16384,
#          'hellaswag': 16384,
#          'glue-mnli': 16384,
#          'lama-trex': 16384,
#          'paws': 16384,
#          'glue-qnli': 16384,
#          'freebase_qa': 16384,
#          'circa': 16384,
#          'piqa': 16113,
#          'emotion': 16000,
#          'google_wellformed_query': 15406,
#          'hatexplain': 15383,
#          'tweet_eval-offensive': 11916,
#          'ropes': 10924,
#          'tweet_qa': 10692,
#          'sciq': 10481,
#          'liar': 10269,
#          'quail': 10246,
#          'amazon_polarity': 10000})

# test task sample count
# Counter({'hate_speech18': 2341,
#          'dream': 2240,
#          'commonsense_qa': 1421,
#          'tweet_eval-hate': 1199,
#          'qasc': 1126,
#          'medical_questions_pairs': 810,
#          'codah': 756,
#          'openbookqa': 700,
#          'sick': 695,
#          'financial_phrasebank': 653,
#          'glue-mrpc': 608,
#          'quartz-no_knowledge': 584,
#          'quartz-with_knowledge': 584,
#          'climate_fever': 507,
#          'ai2_arc': 499,
#          'quarel': 478,
#          'glue-rte': 477,
#          'poem_sentiment': 305,
#          'superglue-copa': 300,
#          'ethos-race': 287,
#          'ethos-religion': 287,
#          'ethos-national_origin': 287,
#          'glue-wnli': 271,
#          'tweet_eval-stance_feminist': 267,
#          'superglue-cb': 256,
#          'tweet_eval-stance_atheism': 252})







# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct', cache_dir='./encoder_decoder_cache')
assert isinstance(tokenizer, PreTrainedTokenizerFast)
assert tokenizer.pad_token is None
assert isinstance(tokenizer.bos_token, str)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

bos = tokenizer.bos_token
eos = tokenizer.eos_token
tokenizer._tokenizer.post_processor = TemplateProcessing(
    single=f"{bos}:0 $A:0 {eos}:0",
    pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
    special_tokens=[
        (f"{bos}", tokenizer.bos_token_id),
        (f"{eos}", tokenizer.eos_token_id)
    ],
)
# tokenizer(['hello hihi', 'hello world to you'], return_tensors='pt', padding=True)
# {'input_ids': tensor([[128000,  15339,  15960,   6151, 128009, 128009],
#                       [128000,  15339,   1917,    311,    499, 128009]]),
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 0],
#                           [1, 1, 1, 1, 1, 1]])}

class ComputeTaskLength:
    def __init__(self, task_indices_tuple, data):
        task, indices = task_indices_tuple

        len = []
        for i in indices:
            data_i = data[i]
            io = ' '.join([data_i["input"], data_i['output']])
            enc = tokenizer(io, return_tensors="pt")
            assert enc['input_ids'].shape[0] == 1
            assert enc['attention_mask'].numel() == enc['attention_mask'].sum()
            len.append(enc['input_ids'].shape[1])

        self.task = task
        self.len = len


def count_seq_len(data, split):

    # task to indices
    task_to_indices = defaultdict(list)
    for idx, task in enumerate(data['task']):
        task_to_indices[task].append(idx)
    assert all(len(indices) >= 16 for indices in task_to_indices.values())

    # print info
    # for task, indices in task_to_indices.items():
    #     print(f"{task} has {len(indices)} indices")

    task_indices_tuples = [(task, indices) for task, indices in task_to_indices.items()]
    compute_task_length_maker = partial(ComputeTaskLength, data=data)
    with Pool(32) as p:
        task_lens = p.map(compute_task_length_maker, task_indices_tuples)
    # task_lens = [ComputeTaskLength(x) for x in task_indices_tuples]

    for x in task_lens:
        plt.figure()
        plt.title(f"{x.task} {split}")
        plt.hist(x.len, bins=25)
        plt.savefig(f'{out_dir}/{x.task}_{split}.jpg')
        plt.close()

    # aggregate
    all_lens = []
    for x in task_lens:
        for l in x.len:
            all_lens.append(l)
    plt.figure()
    plt.title(f"all {split}")
    plt.hist(all_lens, bins=25)
    plt.savefig(f'{out_dir}/all_{split}.jpg')
    plt.close()

    print(f'avg len of {split}: {sum(all_lens)/len(all_lens)}')
    print(f'max len of {split}: {max(all_lens)}')


out_dir = 'metaicl_data_viz'
os.makedirs(out_dir, exist_ok=True)
count_seq_len(train_split, 'train')
count_seq_len(test_split, 'test')

# avg len of train: 117.5443107442095
# max len of train: 6937
# avg len of test: 50.29329301814184
# max len of test: 1285
