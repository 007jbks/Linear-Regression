# Predicting Student's marks based on class attendance
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = {
	'Attendance':[70,92,67,82,80,72,85,75,80,70,85,65,70,75,80,85,75,68,65,95,90,78,78,80,85,95,78,79,65,98,98,68,88,75,88,88,88,90,75,98,95,98,78,95,75,77,85,70,95,95,75,90,90,96,94,80,75,78,82,80,80,92,85,68,88,80,62,82],
	'Marks':[42,39,32,50,44,55,43,37,43,41,48,61,44,45,58,52,45,46,46,40,57,42,35,52,64,47,42,52,55,50,53,50,51,49,39,39,49,52,49,45,36,35,50,43,49,57,38,41,46,50,51,38,49,44,41,57,45,49,49,50,44,40,50,43,36,50,42,51]
}
df = pd.DataFrame(data)
X = df[['Attendance']]
y = df['Marks']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=41)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(f"Intercept:{model.intercept_}")
print(f"Coefficient:{model.coef_}")
results = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})
print(results)
mean_deviation = np.mean(np.abs(y_test - y_pred))
print(f"Mean Deviation{mean_deviation:.2f}")
mse = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error:{mse:.2f}")

plt.scatter(X,y,color="red",label="Actual Data")
plt.plot(X,model.predict(X),color="blue",label="Regression Line")
plt.xlabel("Attendance")
plt.ylabel("Marks in End Semester Exams(out of 70)")
plt.legend()
plt.show()