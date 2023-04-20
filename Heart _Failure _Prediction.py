import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import LocalOutlierFactor  
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)   
pd.set_option('display.max_rows', None)    
pd.set_option('display.float_format', lambda x: '%.3f' % x)   
pd.set_option('display.width', 500) 

df = pd.read_csv("/kaggle/input/heart-failure-prediction/heart.csv") 


def check_df(dataframe, head=5):
    print("#################### Head ####################")
    print(dataframe.head(head))
    print("################### Shape ####################")
    print(dataframe.shape)
    print("#################### Info #####################")
    print(dataframe.info())
    print("################### Nunique ###################")
    print(dataframe.nunique())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles #################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("################# Duplicated ###################")
    print(dataframe.duplicated().sum())

check_df(df)


# We need to identify the numerical and categorical variables in the data.

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    return cat_cols, num_cols, cat_but_car 

cat_cols, num_cols, cat_but_car = grab_col_names(df)

print(num_cols)
print(cat_cols)


def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        fig = plt.subplots(figsize=(6, 4))
        sns.distplot(df[col_name],
             kde=False,
             kde_kws={"color": "g", "alpha": 0.3, "linewidth": 5, "shade": True})
        plt.show(block=True)
        
        
# We are analyzing the numeric variables.

for col in num_cols:
   num_summary(df, col, plot = True)
    
    
# We are analyzing the categorical variables.

for col in cat_cols:
    print(df[col].value_counts())
    fig = plt.subplots(figsize=(6, 4))
    sns.countplot(x = df[col], data = df)
    sns.set(rc={'figure.dpi':90})
    plt.show(block = True)
    
    
# We are analyzing the target variable.

for col in num_cols:
    print(df.groupby('HeartDisease').agg({col: 'mean'}))
    fig = plt.subplots(figsize=(6, 4))
    sns.violinplot(x=df["HeartDisease"], y=df[col])
    plt.show(block=True)
    
    
# We are analyzing the outliers.

# To detect outliers, we need to set threshold values.
def outlier_thresholds(dataframe, col_name, q1=0.04, q3=0.96):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# We are checking the variables that have outliers.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    

for col in num_cols:
    print(col, check_outlier(df, col))
    
    
# We replace the outliers with the threshold values we determined.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
for col in num_cols:
    replace_with_thresholds(df, col)
    
check_outlier(df, num_cols)


# We generate our scores with LOF.

df_num_cols = df[num_cols]

clf= LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df_num_cols)
df_scores = clf.negative_outlier_factor_

# We are examining the scores through a graph.
scores = pd.DataFrame(np.sort(df_scores))   
scores.plot(stacked=True, xlim=[0, 70], style='.-')
plt.show()

# We set the 7th point as the threshold.
th= np.sort(df_scores)[7]

# We are looking at the outlier observation units that fall below this threshold.
df[df_scores < th]

# We remove these outlier observations from the dataset.
df = df[~(df_scores < th)]


# We are examining our correlation analysis.

corr = df[df.columns].corr()

sns.set(rc={"figure.figsize" : (11,11)})
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax= 1 , center=0, annot=True,linewidth=.5,square=False)

plt.show()


# We are analyzing the missing values.

df.isnull().any()


df.describe().T


# It seems no missing values. But at first we saw that there are meaningless values.

# For example: Variable such as Cholesterol cannot be 0. We will treat these variables as NaN.


# We set 0 values to NaN

df["Cholesterol"] = np.where(df["Cholesterol"] == 0, np.NaN, df["Cholesterol"])
    
df.isnull().sum()


# We have identified missing values and we need to fill them in.
# We are implementing this using K-Nearest Neighbor (KNN) algorithm.

dff = pd.get_dummies(df, drop_first=True)

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

dff.head()


# We are filling in the variable and reversing the data back to its original state.

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["Cholesterol"] = dff["Cholesterol"]

df.isnull().sum()


df.dropna(inplace=True)

df.describe().T


# Now we are creating new variables.

df.loc[(df['Oldpeak'] <= 0) & (df['FastingBS'] == 0), 'NEW_PEAK_FAST'] = 'peak<bs0'

df.loc[(df['Oldpeak'] <= 0) & (df['FastingBS'] == 1), 'NEW_PEAK_FAST'] = 'peak<bs1'

df.loc[(df['Oldpeak'] > 0) & (df['Oldpeak'] < 2) & (df['FastingBS'] == 0), 'NEW_PEAK_FAST'] = 'peak--bs0'

df.loc[(df['Oldpeak'] > 0) & (df['Oldpeak'] < 2) & (df['FastingBS'] == 1), 'NEW_PEAK_FAST'] = 'peak--bs1'

df.loc[(df['Oldpeak'] >= 2) & (df['FastingBS'] == 0), 'NEW_PEAK_FAST'] = 'peak>bs0'

df.loc[(df['Oldpeak'] >= 2) & (df['FastingBS'] == 1), 'NEW_PEAK_FAST'] = 'peak>bs1'


# Now we are creating new variables.

def calculate_peak_bshr(row):
    if row['Oldpeak'] < 0:
        return ((row['FastingBS'] + 1) * 100 / row['MaxHR']) * 1.5
    elif row['Oldpeak'] < 2:
        return (row['FastingBS'] + 1) * 100 / row['MaxHR']
    else:
        return ((row['FastingBS'] + 1) * 100 / row['MaxHR']) * 2
    
df['peak_bshr'] = df.apply(calculate_peak_bshr, axis=1)


# Now we are creating new variables.

df["AGE_CHOL"] = df["Age"] / df["Cholesterol"]


# Now we are creating new variables.

df.loc[(df['ST_Slope'] == "Up") & (df['ExerciseAngina'] == "N"), 'NEW_ST_EXAN'] = 'stupexan-'

df.loc[(df['ST_Slope'] == "Up") & (df['ExerciseAngina'] == "Y"), 'NEW_ST_EXAN'] = 'stupexan+'

df.loc[(df['ST_Slope'] == "Down") & (df['ExerciseAngina'] == "N"), 'NEW_ST_EXAN'] = 'stdownexan-'

df.loc[(df['ST_Slope'] == "Down") & (df['ExerciseAngina'] == "Y"), 'NEW_ST_EXAN'] = 'stdownexan+'

df.loc[(df['ST_Slope'] == "Flat") & (df['ExerciseAngina'] == "N"), 'NEW_ST_EXAN'] = 'stflatexan-'

df.loc[(df['ST_Slope'] == "Flat") & (df['ExerciseAngina'] == "Y"), 'NEW_ST_EXAN'] = 'stflatexan+'


# Now we are creating new variables.

def calculate_pain_age_hr(row):
    if row['ChestPainType'] == "ATA":
        return row['Age'] * 12 / row['MaxHR'] 
    elif row['ChestPainType'] == "NAP":
        return row['Age'] * 14 / row['MaxHR']
    elif row['ChestPainType'] == "TA":
        return row['Age'] * 16 / row['MaxHR']
    else:
        return row['Age'] * 18 / row['MaxHR']
    
df['pain_age_hrr'] = df.apply(calculate_pain_age_hr, axis=1)

df.head()


# We are performing the encoding process.

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() >= 2]

ohe_cols = [col for col in ohe_cols if col not in "HeartDisease"]

df= one_hot_encoder(df, ohe_cols)

df.head()


# We are performing standardization processes.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

mms = MinMaxScaler()    
df[num_cols] = mms.fit_transform(df[num_cols])

df.head()


# We are applying our machine learning model.( Logistic Regression)

y = df["HeartDisease"]   

X = df.drop(["HeartDisease"], axis=1)   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state=25)

log_model = LogisticRegression().fit(X_train, y_train) 

y_pred = log_model.predict(X_test)   
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))


plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob)


# K-Fold Cross Validation

y = df["HeartDisease"]   

X = df.drop(["HeartDisease"], axis=1)   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

log_model = LogisticRegression().fit(X_train, y_train)

cv_results = cross_validate(log_model,    
                            X, y,         
                            cv=17,        
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


print(cv_results['test_accuracy'].mean(),
      cv_results['test_precision'].mean(),
      cv_results['test_recall'].mean(),
      cv_results['test_f1'].mean(),
      cv_results['test_roc_auc'].mean())








