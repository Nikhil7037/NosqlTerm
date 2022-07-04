import pandas as pd
import sklearn
from sklearn import preprocessing

df = pd.read_csv("train.csv")

print(df.head())

df = df.fillna(value = -1)

# Dropping columns
del df["Track Name"]

# Typecasting
df["Artist Name"] = df["Artist Name"].astype(str)


# Initializing Encoder
number = sklearn.preprocessing.LabelEncoder()

# Encoding
df["Artist Name"] = number.fit_transform(df["Artist Name"])

print(df.head())

print(df.shape)

from sklearn.model_selection import train_test_split

# Columns used as predictors
X = df.drop(["Class"], axis = 1).values

y = df["Class"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

print("this is testing data result")
print(y_test)
print("this is expected data result")
y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
print(cm)
print(ac)

