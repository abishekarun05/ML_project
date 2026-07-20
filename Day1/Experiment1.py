from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
x=[[1], [2], [3], [4]]
y=[50, 60, 70, 80]
model=LinearRegression()
model.fit(x, y)
prediction=model.predict([[5]])
print("Prediction for the 5th hour:", prediction[0])
data=[[1], [2], [2], [8], [9], [8]]
model2=KMeans(n_clusters=2, n_init=10)
model2.fit(data)
print("Groups found:", model2.labels_)