import pandas as pd
import numpy as np

# Memory Optimization for DataFrames
## Downcasts data types (int, float)

def reduce_df_memory(df):
    tmp = df.copy()

    float_cols = tmp.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')
    tmp[float_cols.columns] = float_cols

    int_cols = tmp.select_dtypes(include=['int']).apply(pd.to_numeric, downcast='int')
    tmp[int_cols.columns] = int_cols

    return tmp