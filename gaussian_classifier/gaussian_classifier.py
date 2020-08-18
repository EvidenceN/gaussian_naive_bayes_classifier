import pandas as pd
import numpy as np
class gnb:
  def __init__(self, prior=None, n_class=None, 
               mean=None, variance = None, classes=None):
    # prior assumption of probability
    self.prior = prior
    # how many unique classes
    self.n_class = n_class
    # mean of x values
    self.mean = mean
    # variance of x values
    self.variance = variance
    # the unique classes present
    self.classes = classes


  # get the mean and variance of the x values
  def fit(self, x, y):
    # get the mean and variance of the x values
    self.x = x
    self.y = y
    self.mean = np.array(x.groupby(by=y).mean())
    self.variance = np.array(x.groupby(by=y).var())
    self.n_class = len(np.unique(y))
    self.classes = np.unique(y)
    self.prior = 1/self.n_class
    return self

  def mean_var(self):
    # mean and variance from the trainig data
    m = np.array(self.mean)
    v = np.array(self.variance)

    # pull and combine the corresponding mean and variance
    self.mean_var = []
    for i in range(len(m)):
      m_row = m[i]
      v_row = v[i]
      for a, b in enumerate(m_row):
        mean = b
        var = v_row[a]
        self.mean_var.append([mean, var])

    return self.mean_var

  def split(self):
    spt = np.vsplit(np.array(self.mean_var()), self.n_class)
    return spt

  def gnb_base(self, x_val, x_mean, x_var):
    # define the base formula for prediction probabilities
    # Variance of the x value in question
    self.x_val = x_val
    # x mean value
    self.x_mean = x_mean
    # the x value that is being used for computation
    self.x_var = x_var

    # natural log
    e = np.e
    # pi
    pi = np.pi
    # first part of the equation
    # 1 divided by the sqrt of 2 * pi * y_variance
    equation_1 = 1/(np.sqrt(2 * pi * x_var))
    
    # second part of equation implementation
    # denominator of equation
    denom = 2 * x_var

    # numerator calculation

    numerator = (x_val - x_mean) ** 2
    # the exponent
    expo = np.exp(-(numerator/denom))
    prob = equation_1 * expo

    return prob

  def predict(self, X):
    self.X = X
    # calculate the probabilities using base formula above

    # defining the mean and variance that has being split into
    # various classes.

    split_class = self.split()
    prob = []
    for i in range(self.n_class):
      # first class
      class_one = split_class[i]
      for i in range(len(class_one)):
        # first value in class one
        class_one_x_mean = class_one[i][0]
        class_one_x_var = class_one[i][1]
        x_value = X[i]
        # now calculate the probabilities of each class. 
        prob.append([self.gnb_base(x_value, class_one_x_mean, 
                                   class_one_x_var)])

    # turn prob into an array

    prob_array = np.array(prob)

    # split the probability into various classes again

    prob_split = np.vsplit(prob_array, self.n_class)

    # calculate the final probabilities

    final_probabilities = []

    for i in prob_split:
      class_prob = np.prod(i) * self.prior
      final_probabilities.append(class_prob)

    # determining the maximum probability 
    maximum_prob = max(final_probabilities)

    # getting the index that corresponds to maximum probability
    prob_index = final_probabilities.index(maximum_prob)

    # using the index of the maximum probability to get
    # the class that corresponds to the maximum probability
    prediction = self.classes[prob_index]

    return prediction



