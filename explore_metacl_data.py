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


# load dataset
dataset = load_dataset("bigheiniuJ/EvalMetaICLAll")
train_split = dataset['meta_train']
test_split = dataset['meta_eval_100shot']
train_tasks = set(list(train_split['task']))
test_tasks = set(list(test_split['task']))
assert train_tasks.issubset(test_tasks)




# check classification tasks
classification_tasks = ["superglue-rte", "tweet_eval-sentiment", "discovery", "glue-rte", "superglue-wsc", "glue-mrpc", "tweet_eval-stance_hillary", "tweet_eval-offensive", "emotion", "hatexplain", "glue-cola", "sick", "paws", "ethos-sexual_orientation", "glue-qqp", "tweet_eval-emotion", "sms_spam", "health_fact", "glue-mnli", "imdb", "ethos-disability", "glue-wnli", "scitail", "trec", "yahoo_answers_topics", "liar", "glue-sst2", "tweet_eval-stance_abortion", "circa", "tweet_eval-stance_climate", "glue-qnli", "tweet_eval-emoji", "ethos-directed_vs_generalized", "ade_corpus_v2-classification", "hate_speech_offensive", "superglue-wic", "google_wellformed_query", "tweet_eval-irony", "ethos-gender", "onestop_english", "trec", "rotten_tomatoes", "kilt_fever"]
assert set(classification_tasks).issubset(test_tasks)

task_to_indices = defaultdict(list)
for idx, task in enumerate(test_split['task']):
    task_to_indices[task].append(idx)

def check_data_i(i):
    data_i = test_split[i]
    if data_i['task'] in classification_tasks:
        assert len(data_i['options']) > 1, data_i['task']
    # else:
    #     assert len(data_i['options']) == 0, (data_i['task'], data_i['task'] in classification_tasks, data_i['options'])

for task in test_tasks:
    print('making sure classification task', task, 'has options and otherwise not')
    with Pool(32) as p:
        task_lens = p.map(check_data_i, task_to_indices[task])



# hr to lr
hr_tasks = set(["piqa", "hate_speech_offensive", "google_wellformed_query", "social_i_qa", "circa", "quoref", "glue-sst2", "scitail", "emo", "cosmos_qa", "freebase_qa", "ag_news", "art", "paws", "kilt_ay2", "glue-qnli", "quail", "ade_corpus_v2-classification", "sciq", "hatexplain", "emotion", "glue-qqp", "kilt_fever", "kilt_nq", "dbpedia_14", "kilt_zsre", "hellaswag", "squad-with_context", "hotpot_qa", "glue-mnli", "ropes", "squad-no_context", "kilt_hotpotqa", "discovery", "superglue-record", "race-middle", "race-high", "lama-trex", "swag", "gigaword", "amazon_polarity", "biomrc", "tab_fact", "tweet_eval-emoji", "tweet_eval-offensive", "tweet_eval-sentiment", "tweet_qa", "imdb", "lama-conceptnet", "liar", "anli", "wiki_qa", "kilt_trex", "wikisql", "wino_grande", "wiqa", "search_qa", "xsum", "yahoo_answers_topics", "yelp_polarity", "yelp_review_full"])
lr_tasks = set(["quarel", "financial_phrasebank", "openbookqa", "codah", "qasc", "glue-mrpc", "dream", "sick", "commonsense_qa", "medical_questions_pairs", "quartz-with_knowledge", "poem_sentiment", "quartz-no_knowledge", "glue-wnli", "climate_fever", "ethos-national_origin", "ethos-race", "ethos-religion", "ai2_arc", "hate_speech18", "glue-rte", "superglue-cb", "superglue-copa", "tweet_eval-hate", "tweet_eval-stance_atheism", "tweet_eval-stance_feminist"])
assert not set(hr_tasks).intersection(set(lr_tasks))
print('number of hr tasks', len(hr_tasks))
print('number of lr tasks', len(lr_tasks))
for t in hr_tasks:
    if t not in train_tasks:
        print('missing hr', t)
for t in lr_tasks:
    if t not in test_tasks:
        print('missing lr', t)

# partition by task
train_split = train_split.filter(lambda example: example["task"] in hr_tasks)
test_split = test_split.filter(lambda example: example["task"] in lr_tasks)

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
