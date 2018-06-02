import os
from cytof_io import *
import pandas as pd
from pandas import DataFrame
import numpy as np
os.chdir("PBMC_150518")
CA = load_sample('CA')
C1 = load_sample('C1')

def test_loaded_sample(ID):
	assert type(ID) == DataFrame, "Not a DataFrame"
	assert ID.shape[1] < ID.shape[0], str(ID.shape) + ". Not of right shape."
	assert len(ID.index.unique()) == len(ID), "Duplicate indices."
	assert ID.isnull().values.sum() == 0, "NaNs in the data."

test_loaded_sample(CA)
test_loaded_sample(C1)
prin('All assertions are True')
