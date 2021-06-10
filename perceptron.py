import os
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn import datasets, model_selection

class Perceptron:
    def __init__(self, x_dim, rho=0.0001):
        np.random.seed(seed=1)
        self.w = np.array([0.5,0.5, 0.5]) # np.random.randn(x_dim + 1)
        self.rho = rho

    def train(self, data, label, count=1):
        while True:
            perm = np.random.permutation(len(data))
            data, label = data[perm], label[perm]

            classified = True

            print("カウント", count, "巡目", "w=", self.w)
            count += 1
            # if count % 2 == 0:
            #     self.rho = self.rho * 1.9

            for x, y in zip(list(data), list(label)):
                pred, value = self.predict(x)
                if pred != y:
                    classified = False

                    x = np.array(list(x) + [1])
                    """
                    元のコード
                    self.w = self.w - pred * self.rho * x
                    以下は閾値からの誤差をpredの値に追加してずれた数値を重みとして扱っている。
                    """
                    self.w = self.w - (pred + value) * self.rho * x
            if classified:
                break
    
    # 重みと入力の積が正であれば1, 負である場合は-1
    def predict(self, x):
        x = np.array(list(x) + [1])
        # print(np.dot(self.w, x))
        if np.dot(self.w, x) > 0:
            # print(np.dot(self.w, x))
            return 1, np.dot(self.w, x)
        else:
            return -1, np.dot(self.w, x)

if __name__ == '__main__':
    dataset = datasets.load_iris()

    x_train, y_train = dataset.data, dataset.target

    perceptron = Perceptron(x_dim=2)

    # Setosa, Versicolourのデータだけ抽出
    # 0,1のものはTrueにする
    mask = np.bitwise_or(y_train == 0, y_train == 1)
    # 花弁の大きさ
    x_train = x_train[mask][:,2:]
    # >>> y_train.size
    # 150
    # >>> y_train[mask].size
    # 100
    # 01のクラスラベルのみに絞る
    y_train = y_train[mask]
    y_train = np.array([-1 if y == 0 else 1 for y in y_train])

    # train
    perceptron.train(x_train, y_train)
