import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Raw Data of Amazon Stocks
data = {
	'Year':[1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
	'Stock Price':[0.7814,3.4450,2.3219,0.6086,0.8221,1.8823,2.1730,1.9900,1.7912,3.3532,3.4856,4.3536,6.9403,9.8100,10.9887,14.8658,16.5876,23.8494,34.8921,48.2920,81.8891,89.2447,133.7207,166.7916,125.7990,121.3728,184.0118]
}
pf = pd.DataFrame(data)
X = pf[['Year']]
Y = pf['Stock Price']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.05,random_state=42)
model = LinearRegression()
model.fit(X_train,Y_train)
print(f"Intercept:{model.intercept_}")
print(f"Coefficeint:{model.coef_}")
Y_pred = model.predict(X_test)
results = pd.DataFrame({'Actual':Y_test,'Predicted':Y_pred})
print(results)
mse = mean_squared_error(Y_test,Y_pred)
print(f"Mean Squared Error:{mse:.2f}")
plt.scatter(X,Y,label="Actual Data",color="blue")
plt.plot(X,model.predict(X),color="red",label="Regression Line")	
plt.show()