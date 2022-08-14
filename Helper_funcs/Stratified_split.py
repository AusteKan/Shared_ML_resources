#A helper function for a stratified split to balance the data 

from sklearn.model_selection import StratifiedShuffleSplit
def StratifiedSplit(X,y,n_splits=1,test_size=0.2,random_state=0):
  """
  A stratified split for datasets that might require balancing
  #Arguments:
    X; array(int), data frame (int) - data to train on
    y; array(int), data frame (int) - data labels
  #Returns
    train_set, test_set,train_y,test_y; array (int) - stratified split of data
  """
  #Initiate split
  split=StratifiedShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=random_state)
  #Subset data
  for train_index, test_index in split.split(X,y):
    train_set=X[train_index]
    train_y=y[train_index]
    test_set=X[test_index]
    test_y=y[test_index]
  
  return train_set, test_set,train_y,test_y
