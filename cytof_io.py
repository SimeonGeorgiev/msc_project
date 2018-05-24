from pandas import read_csv, concat
import numpy as np

def shuffle(df, n):
	for _ in range(n):
		df = df.reindex(np.random.permutation(df.index))
	return df

def load_df(file="Levine_32dimtransform.csv", s_e_markers=(4, 36), seed=42, n=10, non_neg=False):
	np.random.seed(seed)
	df = read_csv(file)
	s, e = s_e_markers
	markers = list(df)[s:e]
	if non_neg == True:
		df[df[markers] < 0] = 0
	df.fillna(value=-1, inplace=True)
	return shuffle(df, n), markers

if __name__ == "__main__":
	import pandas as pd
	df = pd.DataFrame({'A':range(10), 'B':range(10)})
	df = shuffle(df)
	print(df)
