import os
from cytof_io import *
from pandas import DataFrame

os.chdir("PBMC_150518")
CA = load_sample('CA')
C1 = load_sample('C1')

def test_loaded_sample(ID):
	assert type(ID) == DataFrame, "Not a DataFrame"
	assert ID.shape[1] < ID.shape[0], str(ID.shape) + ". Not of right shape."
	assert len(ID.index.unique()) == len(ID), "Duplicate indices."

test_loaded_sample(CA)
test_loaded_sample(C1)
