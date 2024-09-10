import pandas as pd
import itertools
import torch
import numpy as np
def gen(path, sensitive_attribute):
    df = pd.read_csv(path) 
    df['sensitive'] = df[sensitive_attribute] > 0.5
    df['toxic'] = df.toxicity > 0.5
    return df[['comment_text', 'sensitive', 'toxic']]
