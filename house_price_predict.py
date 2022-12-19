import warnings

warnings.filterwarnings("ignore")  # 忽略警告信息
import numpy as np
import pandas as pd
from scipy.stats import norm, skew  # 获取统计信息
import seaborn as sns  # 绘图包
from sklearn.preprocessing import LabelEncoder

color = sns.color_palette()
sns.set_style('darkgrid')


import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # 限制浮点输出到小数点后3位



# 加载数据
trainning_set = pd.read_csv('kagglehouse/train.csv')
testing_set = pd.read_csv('kagglehouse/test.csv')

# 检查样本和特征的数量
print("训练集初始大小: {} ".format(trainning_set.shape))
print("测试集初始大小: {} ".format(testing_set.shape))

#查看前5行数据
print(trainning_set.head())
print(testing_set.head())

# 删除原数据集的Id列
trainning_set.drop("Id", axis=1, inplace=True)
testing_set.drop("Id", axis=1, inplace=True)

#绘制目标值分布
sns.distplot(trainning_set['SalePrice'])
plt.show()

# 计算房价的峰度和偏度
top = trainning_set['SalePrice'].skew()
bottom = trainning_set['SalePrice'].kurt()
print('峰度：',top)
print('偏度：',bottom)

#查看目标值统计信息
print(trainning_set['SalePrice'].describe())

numberfeature = []
cate = []
for colum in testing_set.columns:
    if testing_set[colum].dtype == 'object':
        cate.append(colum)
    else:
        numberfeature.append(colum)
print('数值型特征：', len(numberfeature))
print('类别型特征：', len(cate))


# 数据预处理
y = trainning_set['SalePrice']
y = np.log1p(y)

sns.distplot(y, fit=norm)

print('处理后峰度：', y.skew())
print('处理后偏度：', y.kurtosis())
plt.show()

#处理异常值
sns.scatterplot(x='GrLivArea', y='SalePrice', data=trainning_set)
plt.show()

trainning_set = trainning_set.drop(trainning_set[(trainning_set['GrLivArea']>4000) & (trainning_set['SalePrice']<200000)].index)
sns.scatterplot(x='GrLivArea', y='SalePrice', data=trainning_set)
plt.show()

trainning_set_nan = trainning_set.isnull().sum()
trainning_set_nan = trainning_set_nan.drop(trainning_set_nan[trainning_set_nan==0].index).sort_values(ascending=False)
print(trainning_set_nan)


none_lst = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType1',
              'BsmtFinType2', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'MasVnrType']
for coloo in none_lst:
    trainning_set[coloo] = trainning_set[coloo].fillna('None')
    testing_set[coloo] = testing_set[coloo].fillna('None')

#Functional填充Typ
trainning_set['Functional'] = trainning_set['Functional'].fillna('Typ')
testing_set['Functional'] = testing_set['Functional'].fillna('Typ')

zero_listsss = ['GarageYrBlt', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageCars', 'GarageArea',
              'TotalBsmtSF']
for colmm in zero_listsss:
    trainning_set[colmm] = trainning_set[colmm].fillna(0)
    testing_set[colmm] = testing_set[colmm].fillna(0)


trainning_set['LotFrontage'] = trainning_set.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
for ind in testing_set['LotFrontage'][testing_set['LotFrontage'].isnull().values==True].index:
    x = testing_set['Neighborhood'].iloc[ind]
    testing_set['LotFrontage'].iloc[ind] = trainning_set.groupby('Neighborhood')['LotFrontage'].median()[x]


trainning_set = trainning_set.drop(['Utilities'], axis=1)
testing_set = testing_set.drop(['Utilities'], axis=1)


trainning_set['MSSubClass'] = trainning_set['MSSubClass'].apply(str)  # apply()函数默认对列进行操作
trainning_set['YrSold'] = trainning_set['YrSold'].astype(str)
trainning_set['MoSold'] = trainning_set['MoSold'].astype(str)
testing_set['MSSubClass'] = testing_set['MSSubClass'].apply(str)
testing_set['YrSold'] = testing_set['YrSold'].astype(str)
testing_set['MoSold'] = testing_set['MoSold'].astype(str)

#LabelEncoder编码
for colnm in cate:
    trainning_set[colnm] = trainning_set[colnm].astype(str)
    testing_set[colnm] = testing_set[colnm].astype(str)
cols = ['Street', 'Alley', 'LotShape', 'LandContour', 'LandSlope', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual',
               'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir',
               'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
for cl in cols:
    encoder = LabelEncoder()
    value_trainning_set = set(trainning_set[cl].unique())
    value_testing_set = set(testing_set[cl].unique())
    value_list = list(value_trainning_set | value_testing_set)
    encoder.fit(value_list)
    trainning_set[cl] = encoder.transform(trainning_set[cl])
    testing_set[cl] = encoder.transform(testing_set[cl])

# 计算特征的偏度
numeric = trainning_set[numberfeature]
feats = numeric.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
nessness = pd.DataFrame({'Skew':feats})
print(nessness.head(10))

new_nessness = nessness[nessness.abs() > 0.5]
print("有{}个高偏度特征被Box-Cox变换".format(new_nessness.shape[0]))

from scipy.special import boxcox1p

skewed_features = new_nessness.index
ngg = 0.15
for feat in skewed_features:
    trainning_set[feat] = boxcox1p(trainning_set[feat], ngg)
    testing_set[feat] = boxcox1p(testing_set[feat], ngg)


#把所有数据连上
totaldataa = pd.concat((trainning_set.drop('SalePrice', axis=1), testing_set)).reset_index(drop=True)
#构造新特征
totaldataa['TotalSF'] = totaldataa['TotalBsmtSF'] + totaldataa['1stFlrSF'] + totaldataa['2ndFlrSF'] # 房屋总面积
totaldataa['OverallQual_TotalSF'] = totaldataa['OverallQual'] * totaldataa['TotalSF']  # 整体质量与房屋总面积交互项
totaldataa['OverallQual_GrLivArea'] = totaldataa['OverallQual'] * totaldataa['GrLivArea'] # 整体质量与地上总房间数交互项
totaldataa['OverallQual_TotRmsAbvGrd'] = totaldataa['OverallQual'] * totaldataa['TotRmsAbvGrd'] # 整体质量与地上生活面积交互项
totaldataa['GarageArea_YearBuilt'] = totaldataa['GarageArea'] * totaldataa['YearBuilt'] # 车库面积与建造时间交互项
totaldataa['IsRemod'] = 1
totaldataa['IsRemod'].loc[totaldataa['YearBuilt']==totaldataa['YearRemodAdd']] = 0  #是否翻新(翻新：1， 未翻新：0)
totaldataa['BltRemodDiff'] = totaldataa['YearRemodAdd'] - totaldataa['YearBuilt']  #翻新与建造的时间差（年）

dummy_features = list(set(cate).difference(set(cols)))
totaldataa = pd.get_dummies(totaldataa, drop_first=True)


import numpy as np
import pandas as pd
import time

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # 限制浮点输出到小数点后3位

y_trainning_set = np.log1p(trainning_set.pop('SalePrice'))

# 岭回归
alphabets = np.logspace(-3, 2, 50)
testing_set_scores = []
for alphabet in alphabets:
    clf = Ridge(alphabet)
    testing_set_score = np.sqrt(-cross_val_score(clf, trainning_set.values, y_trainning_set, cv=10, scoring='neg_mean_squared_error'))
    testing_set_scores.append(np.mean(testing_set_score))
plt.plot(alphabets, testing_set_scores)
plt.show()

alphabets =np.logspace(0,1,50)


lasso_alphabet = np.logspace(-4,1,50)
testing_set_scores = []
for alphabet in lasso_alphabet:
    lasso =Lasso(alphabet)
    testing_set_score = np.sqrt(-cross_val_score(lasso, trainning_set.values, y_trainning_set, scoring='neg_mean_squared_error', cv=10))
    testing_set_scores.append(np.mean(testing_set_score))
    print(alphabet,np.mean(testing_set_score))
plt.plot(lasso_alphabet,testing_set_scores)
plt.show()

lasso_alphabet = np.logspace(-5,-2,50)

from sklearn.ensemble import RandomForestRegressor

interval = [50, 100, 150, 200, 250, 300, 350, 400]
testing_set_scores = []
for ind in interval:
    clf = RandomForestRegressor(n_estimators=ind, max_features=0.3)
    testing_set_score = np.sqrt(-cross_val_score(clf, trainning_set.values, y_trainning_set, cv=10, scoring='neg_mean_squared_error'))
    testing_set_scores.append(np.mean(testing_set_score))
    print(f"N:{ind},平均testing_set_score:{sum(testing_set_score)/10}")

plt.plot(interval, testing_set_scores)
plt.show()


lasso = Lasso(alpha=0.0001)
lasso.fit(trainning_set.values, y_trainning_set)
lasso_predict = lasso.predict(testing_set.values)
y_lasso = np.expm1(lasso_predict)

testing_set0 = pd.read_csv('kagglehouse/testing_set.csv')
testing_set_id = testing_set0['Id']
submission = pd.DataFrame()
submission['Id'] = testing_set_id
submission['SalePrice'] = y_lasso

submission.to_csv('kagglehouse/submission.csv', index=False)