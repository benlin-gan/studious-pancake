import pandas as pd

def gen_dataset(path, sensitive_attribute, toxic_threshold=0.15):
    dfg = pd.read_csv(path, chunksize=100000) 
    for df in dfg:
        df = df[(df['split'] == 'train') & (~df[sensitive_attribute].isna())]
        df['sensitive'] = df[sensitive_attribute] > 0.0
        df['toxic'] = df["toxicity"] > toxic_threshold
        yield df[['id', 'comment_text', 'toxic', 'sensitive']]
    

