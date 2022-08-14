def train_validation_plots(history,epoch_num,model_name):

  """
  This is a helper function to plot validation and training outputs for a multi-label classifier
  Parameters plotted include accuracy and loss (categorical_crossentropy)

  #Arguments:
    history; keras(obj)- model training history
    epochnum; int - the number of epochs
    model_name; str - model name
  #Returns
    plt1; std out plot- an accuracy plot
    plt2; std out plot- a loss plot
  """


  plt.plot(range(1,epoch_num+1),history['accuracy'],color='blue',label="Train accuracy")
  plt.plot(range(1,epoch_num+1),history['val_accuracy'],color='red',label="Validation accuracy")
  plt.legend()
  plt.title(model_name)
  plt.show()

  plt.plot(range(1,epoch_num+1),history['loss'],color='blue',label="Train loss")
  plt.plot(range(1,epoch_num+1),history['val_loss'],color='red',label="Validation loss")
  plt.legend()
  plt.title(model_name)
  plt.show()

  return
