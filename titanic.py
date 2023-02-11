import pandas as pd
import numpy as np
from sklearn import tree
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0:'欠損数', 1: '%'})
    return kesson_table_ren_columns
# kesson_table(train)
# kesson_table(test)

# データ処理--------------------------------------------------------------------------------------------------------
# 欠損を補修
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna("S")
test.Fare[152] = test.Fare.median()

# 文字列を数字へ変更
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# 決定木：予測
# 「train」の目的変数と説明変数の値を取得
# 目的
target = train["Survived"].values

# 4つの変数ver.------------------------------------------------------------------------------------------------------------
# 説明
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# 「test」の説明変数の値を取得
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)

print(my_prediction.shape)
print(my_prediction)

# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)

# my_prediction(予想データ)とPassengerIdをデータフレームへ落とし込みcsv化
my_solution = pd.DataFrame(my_prediction,PassengerId,columns = ["Survived"])
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])


# 7つの変数ver.------------------------------------------------------------------------------------------------------------
# 追加となった項目も含めて予測モデルその2で使う値を取り出す
features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# 決定木の作成とアーギュメントの設定と学習（前回との違い：過学習(Overfitting)を考慮）
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split=min_samples_split, random_state=1)
# 学習
my_tree_two = my_tree_two.fit(features_two,target)

# testから7項目の値を取り出す
test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# 決定木で予測をしてCSVへ書き出す
my_prediction_tree_two = my_tree_two.predict(test_features_2)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns=["Survived"])
my_solution_tree_two.to_csv("my_tree_two.csv", index_label = ["PassengerId"])
