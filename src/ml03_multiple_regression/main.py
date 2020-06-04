import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

print('-'*10 + 'データフレーム' + '-'*10)
print(df.head())

X = df.drop('MEDV', axis=1)
y = df['MEDV']
print('-'*10 + '特徴量とターゲットに分割' + '-'*10)
print('X:', X.shape)
print('y:', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lm = LinearRegression()
lm.fit(X_train, y_train)

print('-'*10 + '学習結果' + '-'*10)
print('バイアス:', lm.intercept_)
print('重み:', lm.coef_)
print('-'*10 + 'モデルの評価' + '-'*10)
print('Train Score:', lm.score(X_train, y_train))
print('Test Score:', lm.score(X_test, y_test))