import numpy as np

class LogisticRegressionGD:
    '''
    Grdient Descent-based logistic regression classifier

    Parameters
    -----------
    eta : float
        Learning rate(0-1)
    n_iter : int
        Passes over the training dataset
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes:
    -----------
    w_ : 1-d array
        Weights after fitting
    b_ : scalar
        Bias after fitting

    losses_: list
       Mean Squared error loss function values in each epoch
    '''

    def __init__(self,eta = 0.01,n_iter = 50,random_state = 824):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self,X,y):
         '''
        Fitting Training Data
        Parameters:
        -----------
        :param X: {array-like},shape = [n_examples,n_features]
                Training vectors,where n_examples is the number
                of examples and n_features is the number of features.
        :param y: array-like,shape = [n_examples,1]
                Target values
        Returns:
        ---------
        :self : Instance of LogisticRegressionGD
        '''

         rgen = np.random.RandomState(self.random_state)
         self.w_ = rgen.normal(loc = 0.0,scale = 0.01,size = X.shape[1])
         self.b_ = np.float_(0)
         self.losses_ = []

         for i in range(self.n_iter):
             net_input = self.net_input(X)
             output = self.activation(net_input)
             errors = y - output
             self.w_ += self.eta *2.0*X.T.dot(errors)/X.shape[0]
             self.b_ += self.eta*2.0*np.mean(errors)
             loss = (-y.T.dot(np.log(output)) - (1-y).T.dot(np.log(1-output)))/X.shape[0]
             self.losses_.append(loss)
         return self

    def net_input(self,X):
        return np.dot(X,self.w_) + self.b_

    def activation(self,z):
        return 1.0/(1.0 + np.exp(-np.clip(z,-250,250)))

    def predict(self, X):
        '''
        Return class label after one unit step
        '''
        return np.where(self.activation(self.net_input(X)) >= 0.5,1,0)


class LogisticRegressionSGD:
    '''
    Logistic Regression with Stochastic Gradient Descent
    Parameters:
    -----------
    eta : float
        Learning rate(0-1)
    n_iter : int
        Passes over the training dataset
    shuffle : bool(default = True)
        Shuffles training data every epoch if True to prevent cycles
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes:
    -----------
    w_ : 1-d array
        Weights after fitting
    b_ : scalar
        Bias after fitting
    losses_ : list
       Mean Squared error loss function values averaged over all training examples in each epoch
    '''

    def __init__(self,eta = 0.1,n_iter = 50,shuffle = True,random_state = 824):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        '''
        Fit Training Data
        Parameters:
        ------------
        :param X: {array-like},shape = [n_examples,n_features]
                Training vectors,where n_examples is the number
                of examples and n_features is the number of features.
        :param y: array-like,shape = [n_examples,1]
                Target values

        Returns:
        ---------
        :self : object,Instance of LogisticRegressionSGD
        '''

        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self,X,y):
        '''
        Fitting Training Data without reinitializing the weights
        主要目的是为了实现在线学习
        '''

        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
        # 检查是为了区分单条样本（X 为 1‑D）和批量样本（X 为 2‑D），
        # 因为对 1‑D 的 X 用 for xi,target in zip(X,y) 会把特征当作样本遍历
            for xi,target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)

        return self

    def _shuffle(self,X,y):
        '''
        Shuffle Training Data
        '''
        r = self.rgen.permutation(len(y))
        return X[r],y[r]

    def _initialize_weights(self,m):
        '''
        Initialize weights into small random numbers
        '''
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0,scale = 0.01,size = m)
        self.b_ = np.float_(0.)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        '''
        Apply LogisticRegression learning rule to update weights
        '''
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.b_ += self.eta * 2.0 * error
        self.w_ += self.eta * 2.0 * error*xi
        loss = -target*(np.log(output)) - (1-target)*(np.log(1-output))

        return loss

    def net_input(self, X):
        '''
        calculate net input
        '''
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        '''
        Compute Sigmoid activation
        '''
        return 1.0/(1.0 + np.exp(-np.clip(z,-250,250)))

    def predict(self, X):
        '''
        return class label after unit step
        '''
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)