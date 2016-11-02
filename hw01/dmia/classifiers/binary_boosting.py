#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import ClassifierMixin, BaseEstimator


class BinaryBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators, lr=0.1, max_depth=3):
        self.base_regressor = DecisionTreeRegressor(criterion='friedman_mse',
                                                    splitter='best',
                                                    max_depth=max_depth)
        self.lr = lr
        self.n_estimators = n_estimators
        self.feature_importances_ = None
        self.estimators_ = []

    def loss_grad(self, original_y, pred_y):
        # Вычислите градиент на кажом объекте
        ### YOUR CODE ###
        grad = -original_y*(1/(1+np.exp(original_y*pred_y)))

        return grad

    def fit(self, X, original_y):
        # Храните базовые алгоритмы тут
        self.estimators_ = []
        self.coef_=[]
        for i in range(self.n_estimators):
            grad = self.loss_grad(original_y, self._predict(X))
            # Настройте базовый алгоритм на градиент, это классификация или регрессия?
            ### YOUR CODE ###
            estimator=deepcopy(self.base_regressor)
            estimator.fit(X,grad)
            
            ### END OF YOUR CODE
            self.estimators_.append(estimator)

        self.out_ = self._outliers(grad)
        #self.feature_importances_ = self.calc_feature_imp()

        return self

    def _predict(self, X):
        # Получите ответ композиции до применения решающего правила
        ### YOUR CODE ###
        y_pred = np.array([self.lr*est.predict(X) for est in self.estimators_]).sum(axis=0)
        return y_pred
    
    def calc_feature_imp(self):
        # Посчитайте self.feature_importances_ с помощью аналогичных полей у базовых алгоритмов
        
        ### YOUR CODE ###
        f_imps = np.array([cl.feature_importances_ for cl in self.estimators_])
        return f_imps/len(self.estimators_)
    
    def predict(self, X):
        # Примените к self._predict решающее правило
        ### YOUR CODE ###
        y_pred = -np.sign(self._predict(X))

        return y_pred

    def _outliers(self, grad):
        # Топ-10 объектов с большим отступом
        ### YOUR CODE ###
        _outliers = grad.argsort()[-10:]

        return _outliers

    
