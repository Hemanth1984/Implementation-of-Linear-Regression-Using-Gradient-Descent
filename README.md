# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required library and read the dataframe

2. Write a function computeCost to generate the cost function.

3. Perform iterations og gradient steps with learning rate.

4. Plot the cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Hemachandran E
RegisterNumber: 212224230093
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X=np.c_[np.ones(len(X1)), X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)
print()
print(X)
print()
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:, -1].values).reshape(-1,1)
print(y)
print()
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print()
print('Name: Hemachandran E')
print("Register No: 212224230093")
print()
print(X1_Scaled)
print()
print(Y1_Scaled)
print()
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled =scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:
THETA:

<img width="720" height="150" alt="484175574-a4dea689-c670-4846-af9a-29b0565e3fe9" src="https://github.com/user-attachments/assets/977bfb98-a72a-4865-9947-61502fc76d31" />

X:

<img width="633" height="781" alt="484175708-8528e280-f839-4600-8f35-def8a3e96043" src="https://github.com/user-attachments/assets/84a63a7e-b712-4ab6-a696-698c81e8283a" />

Y:

<img width="431" height="748" alt="484175799-451534f3-ccf2-4bd9-8ce3-32b3864e72ac" src="https://github.com/user-attachments/assets/ac7bf262-8c5f-40d5-89a4-7a18b57863ae" />

X1_Scaled:

<img width="812" height="780" alt="image" src="https://github.com/user-attachments/assets/008771fa-d555-4ea0-8462-84f52d70ceb1" />

Y1_Scaled:

<img width="565" height="775" alt="484176025-5c53a8c2-49fd-410d-8d52-00f2e2100ef3" src="https://github.com/user-attachments/assets/92dc1b69-5670-43c4-9a6c-7245358576a1" />

Predicted value:

<img width="375" height="47" alt="484176124-025c0ccb-3a10-4a83-b5b7-f674ba7f6a93" src="https://github.com/user-attachments/assets/444d0a14-7982-43d8-87e0-be83acf733d7" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
