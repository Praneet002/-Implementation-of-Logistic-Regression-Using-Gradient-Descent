# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM :

To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required :

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

### STEP 1 :

Use the standard libraries in python for finding linear regression.

### STEP 2 :

Set variables for assigning dataset values.

### STEP 3 :

Import linear regression from sklearn.

### STEP 4:

Predict the values of array.

### STEP 5:

Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

### STEP 6 :

Obtain the graph.

## Program :

### Program to implement the the Logistic Regression Using Gradient Descent.
### Developed by : ABRIN NISHA A 
### RegisterNumber : 212222230005

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

df=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=df[:,[0,1]]
y=df[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y) / X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),
                        method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min() - 1,X[:,0].max()+1
  y_min,y_max=X[:,1].min() - 1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))

  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label='admitted')
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label='NOT admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output :

### Array Value of x :

![M1](https://github.com/Abrinnisha6/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118889454/869c1b98-55fc-4e6f-b346-a19a5b74c94b)

### Array Value of y :

![M2](https://github.com/Abrinnisha6/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118889454/52206aa2-d8d5-4762-af0b-97019b1632f9)

### Exam 1 - score graph :

![M3](https://github.com/Abrinnisha6/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118889454/12f2b3de-7dce-4ab1-8a97-660ff043b1d1)

### Sigmoid function graph :

![M4](https://github.com/Abrinnisha6/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118889454/5e7eb4e9-ec57-4441-a81d-19d5dd978837)

### X_train_grad value :

![M5](https://github.com/Abrinnisha6/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118889454/d350fd8e-e71a-4955-97fd-dd116d0534c4)

 ### Y_train_grad value :

 ![M6](https://github.com/Abrinnisha6/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118889454/a8b36830-1846-453c-a0f5-7aaeb973634f)

### Print res.x :

![M7](https://github.com/Abrinnisha6/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118889454/62c350d7-e57d-4d20-8a26-30f1a9268957)

### Decision boundary - graph for exam score :

![M8](https://github.com/Abrinnisha6/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118889454/08a6967e-19c7-4943-9d81-f0145ea9d6f7)

### Proability value :

![M9](https://github.com/Abrinnisha6/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118889454/fb37ffbb-54d5-49c9-a917-c7783a6f595b)

### Prediction value of mean :

![M10](https://github.com/Abrinnisha6/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118889454/6be9d456-e14c-4cd5-945f-e2824136fa7f)


## Result :

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.









