import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
import pickle


filename = "../../data/processed/final-df.csv"
df = pd.read_csv(filename)


df.drop(columns=["Unnamed: 0"], inplace=True)
df.head()

# ----------------------------------------------------------------
# Creating the Model
# ----------------------------------------------------------------

X = df.drop(columns=["Price"])

y = np.log(df["Price"])


X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.15, random_state=2
)

# ----------------------------------------------------------------
# Applying all Different types to algorithms to find the ideal output
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# Linear Regression
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)

step2 = LinearRegression()

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))


# ----------------------------------------------------------------
# Ridge Regression
# ----------------------------------------------------------------
step1 = ColumnTransformer(
    transformers=[
        ("col_tnf", OneHotEncoder(sparse=False, drop="first"), [0, 1, 7, 10, 11])
    ],
    remainder="passthrough",
)

step2 = Ridge(alpha=10)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))

# ----------------------------------------------------------------
# Lasso Regression
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)

step2 = Lasso(alpha=0.001)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))

# ----------------------------------------------------------------
# KNN (nearest neighbors)
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))


# ----------------------------------------------------------------
# Decision Tree
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))

# ----------------------------------------------------------------
# SVM (Support vector machine)
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)

step2 = SVR(kernel="rbf", C=10000, epsilon=0.1)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))

# ----------------------------------------------------------------
# Random Forest Model
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)

step2 = RandomForestRegressor(
    n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15
)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))

# ----------------------------------------------------------------
# Extra Trees
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)

step2 = ExtraTreesRegressor(
    n_estimators=100,
    random_state=3,
    max_samples=0.5,
    bootstrap=True,
    max_features=0.75,
    max_depth=15,
)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))


# ----------------------------------------------------------------
# Ada Boost
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)

step2 = AdaBoostRegressor(n_estimators=15, learning_rate=1.0)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))

# ----------------------------------------------------------------
# Gradient Boost
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))

# ----------------------------------------------------------------
# XgBoost
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)

step2 = XGBRegressor(n_estimators=45, max_depth=5, learning_rate=0.5)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))

# ----------------------------------------------------------------
# Voting Regression
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)


rf = RandomForestRegressor(
    n_estimators=350, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15
)
gbdt = GradientBoostingRegressor(n_estimators=100, max_features=0.5)
xgb = XGBRegressor(n_estimators=25, learning_rate=0.3, max_depth=5)
et = ExtraTreesRegressor(
    n_estimators=100,
    random_state=3,
    bootstrap=True,
    max_samples=0.5,
    max_features=0.75,
    max_depth=10,
)

step2 = VotingRegressor(
    [("rf", rf), ("gbdt", gbdt), ("xgb", xgb), ("et", et)], weights=[5, 1, 1, 1]
)

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))

# ----------------------------------------------------------------
# Stacking
# ----------------------------------------------------------------

step1 = ColumnTransformer(
    transformers=[
        (
            "col_tnf",
            OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"),
            [0, 1, 7, 10, 11],
        )
    ],
    remainder="passthrough",
)


estimators = [
    (
        "rf",
        RandomForestRegressor(
            n_estimators=350,
            random_state=3,
            max_samples=0.5,
            max_features=0.75,
            max_depth=15,
        ),
    ),
    ("gbdt", GradientBoostingRegressor(n_estimators=100, max_features=0.5)),
    ("xgb", XGBRegressor(n_estimators=25, learning_rate=0.3, max_depth=5)),
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

pipe = Pipeline([("step1", step1), ("step2", step2)])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print("R2 score", r2_score(Y_test, Y_pred))
print("MAE", mean_absolute_error(Y_test, Y_pred))

# ----------------------------------------------------------------
# Exporting the model
# ----------------------------------------------------------------

pickle.dump(df, open("../../models/df.pkl", "wb"))
pickle.dump(pipe, open("../../models/pipe.pkl", "wb"))
