import pandas as pd

def gen_dataset(path, sensitive_attribute):
    df = pd.read_csv(path, chunk_size=100000)


