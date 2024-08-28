import pandas as pd
from transformers import AutoTokenizer
import itertools
import torch
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
def gen_point(path, sensitive_attribute, toxic_threshold):
    dfg = pd.read_csv(path, chunksize=100000) 
    for df in dfg:
        df = df[(df['split'] == 'train') & (~df[sensitive_attribute].isna())]
        df['sensitive'] = df[sensitive_attribute] > 0.0
        df['toxic'] = df["toxicity"] > toxic_threshold
        op = df[['comment_text', 'toxic', 'sensitive']]
        for _, row in op.iterrows():
            yield(row)
def gen_batch(path, sensitive_attribute, toxic_threshold=0.15, batch=200):
    g = gen_point(path, sensitive_attribute, toxic_threshold)
    pf = 'is "'
    sf = '" toxic? please respond with Yes or No.'
    while True:
        bg = itertools.islice(g, batch)
        df = pd.concat(r.to_frame().T for r in bg)
        toxic = torch.tensor(df.toxic.to_numpy(dtype=bool))
        sensitive = torch.tensor(df.sensitive.to_numpy(dtype=bool))
        lists = [tokenizer.encode(pf + ro + sf) for ro in df.comment_text]
        pad_to = max(len(i) for i in lists) + 1
        features = torch.tensor([(pad_to - len(i)) * [50256] + i for i in lists]) #front pad with <|endoftext|>
        yield (features, toxic, sensitive)

