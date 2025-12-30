import numpy as np

class Perception:
    '''
    Perception Clasifier(感知机）

    Parameters:
    -------------

    eta : float
        Learning rate(0-1)
    n_iter : int
        Passes over the training dataset
    random_state : int
        Random number generator seed for random weight

    Attributes
    -------------
    w_ : 1-d array
        Weights after fitting
    b_ : scalar
        Bias after fitting

    errors : list
        Number of misclassifications (updates) in each epoch.

    '''

    def __init__(self,eta = 0.1,n_iter = 50,random_state = 824):
        self.eta = eta
        self.random_state = random_state
        self.n_iter = n_iter

    def fit(self,X,y):
        '''
        Fitting Train Data
        Parameters
        -----------
        :param X: {array-like},shape = [n_examples,n_features]
                Training vectors, where n_examples is the number of examples
                and n_features is the number of features
        :param y: array-like,shape = [n_examples,1]
                Target values(label)
        Returns:
        -----------
        :self : object
        '''

        regn = np.random.RandomState(self.random_state)
        self.w_ = regn.normal(loc = 0.0,scale = 0.01,size = X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update*xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self,X):
        '''
        calculate net input
        '''
        return np.dot(X,self.w_)+ self.b_

    def predict(self,X):
        '''
        return class label after unit step
        '''
        return np.where(self.net_input(X) >= 0.0, 1,0)

### 感知机最大问题是其收敛性，举个例子，若不存在线性超平面将两类别数据完全分开，除非设定最大n_iter，那么perception将永远进行权值更新一直迭代下去