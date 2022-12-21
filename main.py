#%%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential
from keras.layers import Dense

import xgboost
import lightgbm

# %%
salaryDf = pd.read_csv(r"data\NBA_season1718_salary.csv")
dataDf = pd.read_csv(r"data\player_data.csv")
playersDf = pd.read_csv(r"data\Players.csv")
statsDf = pd.read_csv(r"data\Seasons_Stats.csv")

# %%
salaryDf.columns = salaryDf.columns.str.lower()
dataDf.columns = dataDf.columns.str.lower()
playersDf.columns = playersDf.columns.str.lower()
statsDf.columns = statsDf.columns.str.lower()

#%%
dataDf = dataDf.rename(columns={'name':'player'})
#%%
salaryDf.info()
#%%
dataDf.info()
#%%
playersDf.info()
#%%
statsDf.info()

#%%
# salary data와 stats data를 merge를 해보았으나 stats data에는 
# 한 선수의 여러 해의 값들이 들어가 있어서 하나의 값으로 통일하기 위해
# 선수별로 groupby를 통해 평균값으로 대체함.

pd.merge(salaryDf, statsDf, how='inner', on='player')

# %%
statsDfAvg = statsDf.groupby('player').mean()
mergeDf = pd.merge(salaryDf, statsDfAvg, how='inner', on='player')
mergeDf = mergeDf.drop(['year'], axis=1)

# %%
midDf = mergeDf.drop(['unnamed: 0_x', 'unnamed: 0_y'], axis=1)
print("merge result data frame's shape :", midDf.shape)

#%%
playersDf

# %%
# 포지션과 몸무게는 player data에서, 키는 players에서 가져옴..
dataDf1 = dataDf[['player', 'position', 'weight']]
playersDf1 = playersDf[['player', 'height']]

midDf1 = pd.merge(midDf, dataDf1, how='inner', on='player')
resultDf = pd.merge(midDf1, playersDf1, how='inner', on='player')


print("모든 데이터에서 필요한 feature들을 가져온 결과의 shape :", resultDf.shape)

# %%
# 총 487개의 결측치가 있는 컬럼 두 개와
# 결측치가 있으면서 2점슛과 3점슛 컬럼과 상관관계가 있어 보이는 2점슛 성공률, 3점슛 성공률 컬럼 제거.

print("<<<<결측치 확인>>>>\n", resultDf.isna().sum())

resultDf = resultDf.drop(['blanl', 'blank2', '3p%', 'ft%'], axis=1)
print("\n\n<<<<결측치 제거 확인>>>>\n", resultDf.isna().sum())
print("최종 data frame shape :", resultDf.shape)

#%%
# 범주형 변수 features labeling 진행
from sklearn.preprocessing import LabelEncoder

resultDfc = resultDf.copy()
le = LabelEncoder()
resultDfc['tm'] = le.fit_transform(resultDf['tm'])
resultDfc['position'] = le.fit_transform(resultDf['position'])
resultDfc

# # dict1 = {'leTm': le.fit_transform(resultDf['tm']), 'lePos' : le.fit_transform(resultDf['position'])}
# # leDf = pd.DataFrame(dict1)
# resultDfc = pd.concat([resultDfc, leDf], axis=1).drop(['tm', 'position'], axis=1)

#%%
# target value인 salary값을 맨 마지막에 위치하게 변환
resultDfc.rename(columns={'season17_18':'salary'}, inplace=True)
resultDf1 = pd.concat([resultDfc.drop(['salary'],axis=1), resultDfc['salary']], axis=1)
resultDf1.head()

# %%
# target과의 상관관계 파악하기
mask = np.zeros_like(resultDf1.corr(), dtype=bool)
mask[np.triu_indices_from(mask)]=True

plt.figure(figsize=(20,20))
sns.heatmap(resultDf1.corr(), mask=mask, cmap='coolwarm_r', linewidths=1)

# %%
# 상관관계 수치 0.5 이상인 feature들만 가져오기
corrSe = resultDf1.corr()['salary'].sort_values(ascending=True)
highCorrCols = corrSe[corrSe >= 0.5].index.to_list()

resultDf2 = resultDf1[highCorrCols]
print("상관관계 0.5이상인 feature들의 dataframe's shape :", resultDf2.shape)
resultDf2

# %%
mm_sc = MinMaxScaler()
mm_array = mm_sc.fit_transform(resultDf2)
mm_df = pd.DataFrame(mm_array, columns=resultDf2.columns.tolist())
mm_df

# %%
# 모델 복잡도를 떨어뜨리기 위해서 PCA(차원 축소)를 진행
# 누적 설명 분산량(각 주성분의 설명 분산량의 합)이 0.9(90%) 이상일 때 가장 이상적으로 차원축소를 했다고 봄..
# 여기서는 n_components가 3개일 때 가장 이상적.(salary제외 18개의 feature들을 4개로 축소)

pca = PCA(n_components=2)
priComponents = pca.fit_transform(mm_df.iloc[:,:-1])
pd.DataFrame(priComponents)
print(sum(pca.explained_variance_ratio_))

pca = PCA(n_components=3)
priComponents = pca.fit_transform(mm_df.iloc[:,:-1])
pd.DataFrame(priComponents)
print(sum(pca.explained_variance_ratio_))

pca = PCA(n_components=4)
priComponents = pca.fit_transform(mm_df.iloc[:,:-1])
pd.DataFrame(priComponents)
print(sum(pca.explained_variance_ratio_))

pca = PCA(n_components=18)
priComponents = pca.fit_transform(mm_df.iloc[:,:-1])
pd.DataFrame(priComponents)
print(sum(pca.explained_variance_ratio_))

# %%
pca = PCA(n_components=3)
priComponents = pca.fit_transform(mm_df.iloc[:,:-1])
pcaDf = pd.DataFrame(priComponents, columns=['pc1', 'pc2', 'pc3'])
pcaDf

# %%
X_set = pcaDf
y_set = mm_df['salary']

X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size = 0.2)

#%%
def mlLearn(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("="*10, f" {model.__class__.__name__} learn!! ", "="*10)
    print(f"{model.__class__.__name__}'s mean_squared_error :", mean_squared_error(y_test, pred))
    print(f"{model.__class__.__name__}'s r2_Score :", r2_score(y_test, pred))
    print("\n")


# %%
lnRg = LinearRegression()
xgb = xgboost.XGBRegressor()
rfRg = RandomForestRegressor()
lgb = lightgbm.LGBMRegressor()

models = [lnRg, rfRg, xgb, lgb]

for model in models:
    mlLearn(model, X_train, X_test, y_train, y_test)

# %%
