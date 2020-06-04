from typing import Tuple

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def get_boston_data() -> pd.DataFrame:
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)\
        .assign(MEDV=boston.target)

    print('-'*10 + 'データフレーム' + '-'*10)
    print(df.head())

    return df


def split_training_and_testing(df: pd.DataFrame) -> Tuple:
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    print('-'*10 + '特徴量とターゲットに分割' + '-'*10)
    print('X:', X.shape)
    print('y:', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0
                                                        )
    return X_train, X_test, y_train, y_test


def create_multiple_model(
        X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series) -> None:
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    print('-'*10 + '学習結果' + '-'*10)
    print('バイアス:', lm.intercept_)
    print('重み:', lm.coef_)
    print('-'*10 + 'モデルの評価' + '-'*10)
    print('Train Score:', lm.score(X_train, y_train))
    print('Test Score:', lm.score(X_test, y_test))

if __name__ == "__main__":
    df = get_boston_data()
    X_train, X_test, y_train, y_test = split_training_and_testing(df)
    create_multiple_model(X_train, X_test, y_train, y_test)

