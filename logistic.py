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

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier

model = OneVsOneClassifier(LogisticRegression())
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
