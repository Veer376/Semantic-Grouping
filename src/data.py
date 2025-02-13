"""
df.parquet is the preprocessed data file that contains the
clinical trails features and cirteria fro eligibilities.txt
Got it from the notebook colab file.

"""

import pandas as pd
import os
from config import data_dir
print('Loading... data')
df = pd.read_parquet(os.path.join(data_dir, 'df.parquet'), engine='pyarrow')
print('Loading... embeddings')
embedding_df=pd.read_parquet(os.path.join(data_dir, "embedding_df.parquet"), engine="pyarrow")
