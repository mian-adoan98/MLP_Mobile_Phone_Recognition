# Model Evaluation Tookit: Overview 
import matplotlib.pyplot as plt 
s
# define function --> visualise the accuracy and validation of the model
def model_performance_vis(epochs, train, validation,target ,ax): 
  # plt.figure(figsize=(14, 7))
  # plot graph for determining accuracy and loss
  ax.plot(epochs, train, "o-"  ,label="Training", color="black")
  ax.plot(epochs, validation, "o-",label="Validation")

  # add graph details
  ax.set_xlabel("Epochs")
  ax.set_ylabel(target)

  ax.set_title(f"{target} Diagram")
  plt.legend()

  return ax
