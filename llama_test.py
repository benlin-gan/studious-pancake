import transformers 
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import data2
import tqdm
import pandas as pd

def main():
    generator = work()
    pd.DataFrame(generator).to_csv('output.csv', index=False)
def work():
    #Tokens: Yes - 9642, No - 2822
    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    llama = transformers.pipeline("text-generation", model='meta-llama/Meta-Llama-3.1-8B-Instruct', model_kwargs={"torch_dtype": torch.bfloat16}).model
    llama.requires_grad_(False)
    llama.to('cuda')
    df = data2.gen('test_set.csv', 'female') #attribute choice is arbitrary, as it only matters for evaluation, not inference
    batch_size = 15
    try:
        with open("output.csv", 'w') as h:
            for i in tqdm.tqdm(range(42000, df.shape[0], 15)):
                b = [f'is "{text}" toxic? please respond with Yes or No.\n' for text in df.iloc[i:i+15].comment_text]
                lgits = batch(llama, tokenizer, b)
                pd.DataFrame(lgits).to_csv(h, header=False, index=False)
                h.flush()
    finally:
        h.flush()
        h.close()
        
def batch(llama, tokenizer, b):
    tokens = tokenizer(b, padding=True, return_tensors='pt')
    N = tokens['input_ids'].shape[1] #sequence length
    B = len(b)
    #extract the index of the last active position of the attention mask
    extract = tokens['attention_mask'].cumsum(-1)[:, -1] - 1
    logits = llama(**tokens.to('cuda'))[0].to('cpu') 
    return logits[torch.arange(B), extract, :][:, torch.tensor([9642, 2822])]
    
if __name__=='__main__':
    main()
