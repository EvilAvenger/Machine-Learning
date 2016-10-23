
#!/usr/bin/python3

import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

__author__ = "Vitaly Bibikov"
__version__ = "0.1"


def __main__():
    np.set_printoptions(suppress=True)
    data = pd.read_csv('https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%203/data/data-logistic.csv', names=None, header=None).as_matrix()

    X = data[:, 1:]
    y = data[:, 0]

    lr = LogicalRegression(C=10, step=0.1, itertations=10000, l2_regularize=True) 
    lr.fit(X,y)

    result = lr.get_probabilities()
    p = roc_auc_score(y, result)

    print("{0:.3f}".format(p))
    return

class LogicalRegression(object):

    def __init__(self, C, step, itertations, l2_regularize):
        self.C = C
        self.k = step
        self.itertations = itertations
        self.l2_regularize = l2_regularize
        return

    def fit(self, x, y):

        self.x = x
        self.y = y

        self.length = x.shape[0]
        self.weights = [0.1] * x.shape[1]
        is_ready = False
        
        for i in range(0, self.itertations + 1):

            if is_ready:
                break

            result = self.__process(i, self.l2_regularize)

            for j in range(0, len(self.weights)):
                is_ready = not self.__reweight(j)
                print("Value: {2}, iteration: {0}, weight: #{1}".format(i, j, self.weights[j]))
              
    def get_probabilities(self):
        result = []

        for i in range(0, self.length):
            weighted_sum = -self.weights[0] * self.x[i][0] - self.weights[1] * self.x[i][1]
            probability = 1 / (1 + math.exp(weighted_sum))
            result.append(probability)

        return result

    def __get_loss_value(self, x_vec, y_vec):
        error = self.__count_error(x_vec, y_vec)
        result = math.log(error)

        return result

    def __count_error(self, x_vec, y_vec):
        elements = sum([x * w for x, w in zip(list(x_vec), self.weights)])
        margin = elements * -y_vec

        return math.exp(margin) + 1

    def __reweight(self, index):
        result = 0
        
        for i in range(0, self.length):
            error = (1 - (1 / (self.__count_error(self.x[i], self.y[i]))))
            error = error * self.y[i] * self.x[i][index]
            result = result + error

        formula = self.weights[index] + (self.k / self.length) * result
        coef = self.k * self.C * self.weights[index]
            
        new_weight = formula - coef
        weight_delta = self.weights[index] - new_weight
        self.weights[index] = new_weight

        print("Delta: {delta}, #{index} ".format(index=index, delta=weight_delta))
        return weight_delta != 0

    def __process(self, index, l2_regularize):
        q = self.__risk_minification()
        result = q / self.length

        if l2_regularize:
            result = self.__make_L2_regularization(result)

        print("Result: {value}, #{index}".format(value=result, index=index))
        return result

    def __make_L2_regularization(self, result):
        result = result + (0.5 * self.C * sum([w*w for w in self.weights]))
        return result

    def __risk_minification(self):
        result = 0
        for i in range(0, self.length):
            result = result + self.__get_loss_value(self.x[i], self.y[i])
        return result


if __name__ == '__main__':
    __main__()
