#A helper function to validate your model
def model_validation(X_test,y_test,model,lb,batch_size=128):

  """
  NOTE this is a helper function for a classifier

  The model validation function uses the test set where the training was done on a multi-label single class set
  Model labels are one-hot encoded for training where categorical crossentropy is tracked

  #Arguments:
    X_test; array(int), data frame (int) - data to train on
    y_test; array(int), data frame (int) - data labels
    lb; LabelBinarizer (obj) - binarised labels
    batch_size; int - a batch size with a default value of 128
  #Returns
    std out - prints loss and accuracy of the classifier
    std out - prints a classification report
    plt - a confusion matrix
  """
  
  #Evaluate the model with model.evaluate
  results=model.evaluate(X_test,y_test,batch_size=batch_size)
  print("{} loss {} and accuracy {}".format(str(model),results[0],results[1]))

  #Predict labels for the test set
  y_pred=model.predict(X_test)

  #Get classification report
  print(classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1),target_names=list(lb.classes_)))
  
  #Plot a confusion matrix
  fig,ax=plt.subplots(figsize=(15,15))
  sns.heatmap(confusion_matrix(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1)),annot=True, fmt='g',cmap='Blues')
  ax.set_xticklabels(list(lb.classes_))
  ax.set_yticklabels(list(lb.classes_))
  ax.tick_params(axis='x',rotation=90)
  ax.tick_params(axis='y',rotation=0)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.show()
