import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt

def main():
  X=[[3,5],[4,3],[7,10],[6,7],[7,6],[20,3],[3,30]]
  Y=[0,1,0,0,1,1,0]

  X_test=[[0,1],[1,2],[3,4],[5,6],[1,0],[2,1],[4,3],[6,5]]
  Y_test=[0,0,0,0,1,1,1,1]
# fix random seed for reproducibility
  seed = 0
  numpy.random.seed(seed)
  model = Sequential()
  model.add(Dense(1, input_dim=len(X[0]), init='uniform', activation='relu'))
  #model.add(Dense(8, init='uniform', activation='relu'))
  model.add(Dense(1, init='uniform', activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # Fit the model
  model.fit(X, Y, nb_epoch=628, batch_size=10)
  # evaluate the model
  scores = model.evaluate(X, Y)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print "Running predictions for support"

  Y_pred = model.predict(X_test)
  Y_pred = [int(round(i)) for i in Y_pred]
  print Y_pred
  print Y_test

  X = numpy.array(X)

  plt.scatter(X[:,0],X[:,1], c=Y, linewidth=1)

  #Plot the decision boundary
  x1_min=-10
  x1_max=35
  x1_step_size = 0.4
  x2_min=-10
  x2_max=35
  x2_step_size = 0.4
  x_test=[]
  for x1 in numpy.arange(x1_min, x1_max, x1_step_size):
    for x2 in numpy.arange(x2_min, x2_max, x2_step_size):
      x_test.append([x1,x2])

  y_test = model.predict(x_test)
  colors=['red','green']
  y_colors = [colors[int(round(i))] for i in list(y_test)]
  x_test = numpy.array(x_test)
  plt.scatter(x_test[:,0],x_test[:,1], c=y_colors,cmap=plt.cm.Paired, alpha=0.8)
  plt.show()
if __name__ == "__main__":
  main() 
