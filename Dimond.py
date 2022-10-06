import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn. linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
train = pd.read_csv('/content/drive/MyDrive/SHAI/train.csv')
test = pd.read_csv('/content/drive/MyDrive/SHAI/test.csv')
train.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
test.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
test_index= test.index
print("train shape: "+ str(train.shape))
print("test shape: "+ str(test.shape))
train.describe(include='all')
train.info()
train.isnull().sum()
numerical_df = train[['carat', 'depth', 'table', 'x', 'y', 'z', 'price']]
train.hist(['carat', 'depth', 'table', 'x','y','z'], figsize=(18,10))
def PlotBarCharts(inpData, colsToPlot):
    fig, subPlot=plt.subplots(nrows=1, ncols=len(colsToPlot), figsize=(20,5))
    fig.suptitle('Bar charts of: '+ str(colsToPlot))

    for colName, plotNumber in zip(colsToPlot, range(len(colsToPlot))):
        inpData.groupby(colName).size().plot(kind='bar',ax=subPlot[plotNumber])
PlotBarCharts(inpData=train, colsToPlot=['cut', 'color', 'clarity'])
ContinuousCols=['carat','depth','table', 'x','y','z']
for predictor in ContinuousCols:
    train.plot.scatter(x=predictor, y='price', figsize=(10,5), title=predictor+" VS "+ 'price')
corr_matrix = train.corr()
corr_matrix
fig, ax = plt.subplots(figsize=(12,7))
dataplot = sns.heatmap(corr_matrix, cmap="YlGnBu_r", annot=True)
scatter_matrix(train[ContinuousCols],figsize=(12,12))
plt.show()
train = train.drop(train[train["x"]==0].index)
train = train.drop(train[train["y"]==0].index)
train = train.drop(train[train["z"]==0].index)
train = train[(train["depth"] < 75) & (train["depth"] > 45)]
train = train[(train["table"] < 80) & (train["table"] > 40)]
train = train[(train["carat"] <10)]
train = train[(train["x"]<40)]
train = train[(train["y"]<40)]
train = train[(train["z"]<40)&(train["z"]>2)]
ContinuousCols=['carat', 'depth', 'table', 'x','y','z']
for predictor in ContinuousCols:
    train.plot.scatter(x=predictor, y='price', figsize=(10,5), title=predictor+" VS "+ 'price')
categorical = train[['clarity', 'color', 'cut']]
encoded_train = train.copy()
encoded_test = test.copy()
le = LabelEncoder()
for col in categorical.columns:
    encoded_train[col] = le.fit_transform(encoded_train[col])
    encoded_test[col] = le.fit_transform(encoded_test[col])
encoded_train.drop('index', axis=1, inplace=True)
encoded_test.drop('index', axis=1, inplace=True)
encoded_train.head()
X_train = encoded_train.drop('price', axis=1)
y_train = encoded_train.price
X_test = encoded_test
poly = PolynomialFeatures(2)
poly_train = poly.fit_transform(X_train)
poly_test = poly.fit_transform(X_test)
poly_train
pipeline_lr = Pipeline([("scalar1", StandardScaler()),
                     ("lr", LinearRegression())])

pipeline_lasso = Pipeline([("scalar2", StandardScaler()),
                      ("lasso", Lasso())])

pipeline_dt = Pipeline([("scalar3", StandardScaler()),
                     ("dt", DecisionTreeRegressor())])

pipeline_rf=Pipeline([("scalar4", StandardScaler()),
                     ("rf", RandomForestRegressor())])

pipeline_kn=Pipeline([("scalar5", StandardScaler()),
                     ("kn", KNeighborsRegressor())])

pipeline_xgb=Pipeline([("scalar6", StandardScaler()),
                     ("xgb", XGBRegressor())])

pipeline_gbr=Pipeline([("scalar7", StandardScaler()),
                     ("gbr", GradientBoostingRegressor())])

pipelines = [pipeline_lr, pipeline_lasso, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb, pipeline_gbr]
pipeline_dict = {0: "LinearRegression", 1: "Lasso", 2: "DecisionTree", 3: "RandomForest", 4: "KNeighbors", 5: "XGBRegressor", 6: "GradientBoostingRegressor"}
for i, pipe in enumerate(pipelines):
    pipe.fit(X_train, y_train)
    model_name = pipeline_dict[i]
    Score=pipe.score(X_train, y_train)
    MSE=mean_squared_error(y_train,pipe.predict(X_train))
    RMSE=abs(np.sqrt(MSE))
    print("{} -> score: {}".format(model_name,Score))
    print("{} -> MSE: {}".format(model_name,MSE))
    print("{} -> RMSE: {}".format(model_name,RMSE))
predictions = pipeline_dt.predict(X_test)
print(predictions)
submission = pd.DataFrame({"Unnamed: 0": test_index, "price": predictions})
submission.to_csv("/content/drive/MyDrive/SHAI/Diamond_Price_Prediction.csv", index=False)
