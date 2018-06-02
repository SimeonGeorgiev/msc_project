from pandas import read_csv, concat
import numpy as np
from FlowCytometryTools import FCMeasurement as FCM

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

def arcsinh_trans(scale=5):
    def wrapped(X):
        return np.arcsinh(X/scale)
    return wrapped

def get_C1_CA_metals(metals):
    not_cd45 = list(filter(lambda x: 'CD45' not in x, metals))
    not_labelled = list(filter(lambda x: '_' not in x, not_cd45))
    labelled = list(filter(lambda x: '_' in x, not_cd45))
    labels_for_live = list(filter(lambda x: ('DEAD' not in x) and ('DNA' not in x), labelled))
    return not_cd45, not_labelled, labelled, labels_for_live

def load_sample(ID, metals=(2, -4)):
    stim = load_fcs(ID="Stim", file="{} Stim Cleaned.fcs".format(ID)).data
    unstim = load_fcs(ID="Unstim", file="{} Unstim Cleaned.fcs".format(ID)).data
    metal_names = list(stim)[metals[0]: metals[1]]
    trans = arcsinh_trans()
    IDstim = trans(stim.copy()[metal_names])
    IDunstim = trans(unstim.copy()[metal_names])

    IDstim['S'] = True
    IDstim[ID] = True
    IDunstim['S'] = False
    IDunstim[ID] = True
    ID_sample_df = IDstim.append(IDunstim, ignore_index=True)
    return ID_sample_df

                      
if __name__ == "__main__":
	import test_cytof_io
