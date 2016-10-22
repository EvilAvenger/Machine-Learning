
#!/usr/bin/python3

import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

__author__ = "Vitaly Bibikov"
__version__ = "0.1"


def __main__():
    np.set_printoptions(suppress=True)
    data = pd.read_csv('D:\\Dropbox\\Study\\Programming\\Coursera\\Machine-Learning\\Yandex\\Week 3\\data\\data-logistic.csv', names=None, header=None)

    data = data.as_matrix().astype(float)

    X = data[:, 1:]
    y = data[:, 0]

    lr = LogicalRegression(C = 5, step = 0.0001, itertations = 10000) 
    lr.fit(X,y)
    result = lr.get_probabilities() # -  0.936285714286 
    p = roc_auc_score(y, result)
    print(p)

    return

class LogicalRegression(object):

    def __init__(self, C, step, itertations):
        self.C = C
        self.k = step
        self.weights = []
        self.length = 0
        self.itertations = itertations
        return

    def fit(self, x, y):

        self.x = x
        self.y = y

        self.length = x.shape[0]
        self.weights = [0] * x.shape[1]
        is_ready = False
        
        for i in range(0, self.itertations + 1):

            if is_ready:
                break

            result = self.__process()
            #print("Result: {value}, #{index}".format(value=result, index=i))
            for j in range(0, len(self.weights)):
                is_ready = not self.__reweight(j)
                #print("Value: {value}, iteration: {index}, weight: #{weight}".format(index=i, weight=j, value=self.weights[j]))

    def get_probabilities(self):
        result = []

        for i in range(0, self.length):
            weighted_sum = -self.weights[0] * self.x[i][0] - self.weights[1] * self.x[i][1]
            probability = 1 / (1 + math.exp(weighted_sum))
            result.append(probability)

        return result

    def __get_loss_function_value(self, x_vec, y_vec):
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
            error = 1 - (1 / (self.__count_error(self.x[i], self.y[i])))
            error = error * self.y[i] * self.x[i][index]
            result = result + error

        formula = self.weights[index] + (self.k / self.length) * result
        coef = self.k * self.C * self.weights[index]
        new_weight = formula - coef
        weight_delta = self.weights[index] - new_weight

        print("Delta: {delta}, #{index} ".format(index=index, delta=weight_delta))
        self.weights[index] = new_weight
        return weight_delta != 0

    def __process(self):
        q = self.__minification_func()
        result = q / self.length
        #result = self.__make_L2_regularization(result)
        return result

    def __make_L2_regularization(self, result):
        result = result + (0.5 * self.C * sum([w*w for w in self.weights]))
        return result

    def __minification_func(self):
        result = 0
        for i in range(0, self.length):
            result = result + self.__get_loss_function_value(self.x[i], self.y[i])
        return result


if __name__ == '__main__':
    __main__()
