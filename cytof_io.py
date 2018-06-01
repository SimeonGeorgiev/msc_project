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

def rename(fcs, all_=False):
    old_names = []
    new_names = []
    map_names = {}
    for PS in filter(lambda x: "$P" in x and "S" in x, fcs.meta.keys()):
        new_name = fcs.meta[PS]
        new_names.append(new_name)
        or_name = PS[2:-1]
        old_name = fcs.meta['_channel_names_'][int(or_name)-1]
        old_names.append(old_name)
        map_names.update({old_name: new_name})

    fcs.data = fcs.data.rename(map_names, axis='columns')
    if all_:
        return fcs, old_names, new_names, map_names
    else:
        return fcs

def load_fcs(file, markers=None, n=2, scale=5, ID=None, seed=42, trans=True):
    np.random.seed(seed)
    if ID is None:
        ID = file
    return rename(FCM(ID=ID, datafile=file))

if __name__ == "__main__":
	import pandas as pd
	df = pd.DataFrame({'A':range(10), 'B':range(10)})
	df = shuffle(df)
	print(df)
