import os
import pandas as pd
filename = 'Atalante-1-2-USD-NoLimitHoldem-PokerStarsPA-1-16-2022.txt'
file_dir = '../data/'  # os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(file_dir, f"train_data/{filename}.txt")

df = pd.read_csv(file_path)
print(df.head())