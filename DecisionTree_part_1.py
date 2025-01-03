import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('./Resource/music.csv')
X = df.drop(columns=['genre'])
Y = df['genre']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
predictions = model.predict(X_test)

joblib.dump(model, 'music-recommender.joblib')

score = accuracy_score(Y_test,predictions)
score
