from datasets import load_dataset


# Load the dataset
dataset = load_dataset("cais/mmlu", 'all')['test']
subjects = sorted(set(dataset['subject']))
for s in subjects:
    sub = dataset.filter(lambda x: x['subject'] == s)
    print(len(sub))
    breakpoint()