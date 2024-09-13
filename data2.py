import pandas as pd
import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
def gen(path, sensitive_attribute):
    df = pd.read_csv(path) 
    df['sensitive'] = df[sensitive_attribute] > 0.5
    df['toxic'] = df.toxicity > 0.5
    df['score'] = df.yes - df.no
    df = df.sort_values('score')[::-1]
    return df[['comment_text', 'sensitive', 'toxic', 'score']]
def sensitive_only(df):
    return df[df.sensitive]
def bnsp(df):
    return df[(~df.sensitive & ~df.toxic) | (df.sensitive & df.toxic)]
def bpsn(df):
    return df[(df.sensitive & ~df.toxic) | (~df.sensitive & df.toxic)]
def roc(df, name):
    ct = df.toxic.cumsum()
    df['tpr'] = ct/ct.iloc[-1]
    cf = (~df.toxic).cumsum()
    df['fpr'] = cf/cf.iloc[-1]
    plt.clf()
    plt.plot(df.fpr, df.tpr)
    plt.plot(df.fpr, df.fpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(name)
    os.makedirs('tmp', exist_ok=True)
    plt.savefig(f"tmp/{'_'.join(name.split(' '))}.png")

def main():
    ids = ['male','female','homosexual_gay_or_lesbian', 'christian','jewish', 'muslim','black','white', 'psychiatric_or_mental_illness']
    for i in ids:
        df = gen('zs.csv', i);
        print(f'generated {i}')
        roc(sensitive_only(df), f"{i} only roc curve")
        roc(bnsp(df), f"{i} bnsp roc curve")
        roc(bpsn(df), f"{i} bpsn roc curve")
        print(f'graphed {i}')

if __name__ == '__main__':
    main()
