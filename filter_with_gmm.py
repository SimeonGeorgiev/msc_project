import numpy as np
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture
import pandas as pd

def filter_with_gmm(df, viability_predictors, rs=42, P=90, nc=1):
  
  """
  function filter_with_gmm(df, viability_predictors, rs=42, P=90, nc=1)

  df = dataframe containing mass cytometry data to be filtered
  viability_predictors = list of marker names AND other parameters to be filtered:
          DNA, ForwardScatter, cell length, markers etc...
  rs = random_state; P = percentile to cut off, default is P=90
  nc = components to use, recommended 1.
  
  Filters the dataset using the log likelihood of multivariate gaussian.
  One component is sufficient as it seems produces the same results.
  When this function is ran on the same dataset with different random_state values:
  numbers = []
  for i in range(1, 200):
    df, gmm = filter_with_gmm(df, viability_predictors, rs=i, P=90)
    numbers.append(sum(~df['inlier']))
  
  numbers = pd.Series(numbers)
  print(numbers.describe())
  count      199.0
  mean     26564.0 
  std          0.0
  min      26564.0
  25%      26564.0
  50%      26564.0
  75%      26564.0
  max      26564.0
  dtype: float64
  """
  GMM_filter = GaussianMixture(n_components=nc, random_state=rs)
  GMM_filter.fit(df[viability_predictors].values)
  scores = GMM_filter.score_samples(df[viability_predictors].values)
  cut_off = np.percentile(scores, P)
  df['scores'] = scores; del scores
  df['inlier'] = cut_off > df['scores']
  return df, GMM_filter
#  print(sum(~(cut_off > scores)))


if __name__ == "__main__":

  
  df = pd.read_csv("Levine_32dimtransform.csv")
  df['specified'] = df['label'].isnull()
  viability_predictors = list(df)[1:37]
  
  df, gmm = filter_with_gmm(df, viability_predictors)
  print(sum(df['inlier']), sum(~df['inlier']))

