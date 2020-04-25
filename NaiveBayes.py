import os
import sys
import numpy as np
from scipy.special import logsumexp
from collections import Counter
import random
import pdb
import math
import itertools
from itertools import product
# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry
# helpers to learn and traverse the tree over attributes
import copy


def renormalize(cnt):
  '''
  renormalize a Counter()
  '''
  tot = 1. * sum(cnt.values())
  for a_i in cnt:
    cnt[a_i] /= tot
  return cnt

#--------------------------------------------------------------------------
# Naive bayes CPT and classifier
#--------------------------------------------------------------------------


class NBCPT(object):
  '''
  NB Conditional Probability Table (CPT) for a child attribute.  Each child
  has only the class variable as a parent
  '''

  def __init__(self, A_i):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
        - A_i: the index of the child variable
    DO NOT forget to include the Beta(0.1,0.1) prior
    '''
    self.index = A_i
    self.N1N2_class0 = [0.1,0.1]
    self.N1N2_class1 = [0.1,0.1]
    self.total_class0 = 0.2
    self.total_class1 = 0.2

  def learn(self, A, C):
    '''
    TODO
    populate any instance variables specified in __init__ to learn
    the parameters for this CPT
        - A: a 2-d numpy array where each row is a sample of assignments
        - C: a 1-d n-element numpy where the elements correspond to the
          class labels of the rows in A
    '''
    for i in range(len(C)):
        if (C[i]==0):
            self.total_class0 +=1
            if(A[i,self.index]==0):
                self.N1N2_class0[0] +=1
            else:
                self.N1N2_class0[1] +=1
        if (C[i]==1):
            self.total_class1 +=1
            if(A[i,self.index]==0):
                self.N1N2_class1[0] +=1
            else:
                self.N1N2_class1[1] +=1


  def get_cond_prob(self, entry, c):
    '''
    TODO
    return the conditional probability P(X|Pa(X)) for the values
    specified in the example entry and class label c
        - entry: full assignment of variables
            e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
        - c: the class
    '''
    if(c==0):
        if(entry[self.index]==0):
            return self.N1N2_class0[0]/self.total_class0
        elif(entry[self.index]==1):
            return self.N1N2_class0[1]/self.total_class0
    if(c==1):
        if(entry[self.index]==0):
            return self.N1N2_class1[0]/self.total_class1
        elif(entry[self.index]==1):
            return self.N1N2_class1[1]/self.total_class1



class NBClassifier(object):
  '''
  NB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    Suggestions for the attributes in the classifier:
        - P_c: the probabilities for the class variable C
        - cpts: a list of NBCPT objects
    '''
    self.cpts=[]
    self.P_c0 =0
    self.P_c1 =0
    for i in range(np.shape(A_train)[1]):
        self.cpts.append(NBCPT(i))
    self._train(A_train,C_train)

  def _train(self, A_train, C_train):
    '''
    TODO
    train your NB classifier with the specified data and class labels
    hint: learn the parameters for the required CPTs
        - A_train: a 2-d numpy array where each row is a sample of assignments
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A
    '''
    for i in range(len(self.cpts)):
        self.cpts[i].learn(A_train,C_train)
    # I suppose that there is no incomplete entry
    self.P_c0 = 0.0
    self.P_c1 = 0.0
    for i in range(len(C_train)):
        if (C_train[i]==0):
            self.P_c0 +=1
        if (C_train[i]==1):
            self.P_c1 +=1
        
    

  def classify(self, entry):
    '''
    TODO
    return the log probabilites for class == 0 and class == 1 as a
    tuple for the given entry
    - entry: full assignment of variables
    e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1
    NOTE this must return both the predicated label {0,1} for the class
    variable and also the log of the conditional probability of this
    assignment in a tuple, e.g. return (c_pred, logP_c_pred)
    '''
    incomplete_indexes = np.where(entry == -1)[0]
    states = list(itertools.product([0, 1], repeat=len(incomplete_indexes)))
    p_total0=0.0
    p_total1=0.0
    #print(len(states))
    #print(entry)
    #print(states)
    if(states != [()]):
        for i,state in enumerate(states):
            entry[incomplete_indexes]=states[i]
            p0=self.P_c0
            p1=self.P_c1
            for j in range(len(entry)):
                p0 = p0* self.cpts[j].get_cond_prob(entry,0)
                p1 = p1* self.cpts[j].get_cond_prob(entry,1)
            p_total0 += p0
            p_total1 += p1
            #print("ajdar")
    else:
        p0 = self.P_c0
        p1 = self.P_c1
        for j in range(len(entry)):
            p0 = p0 * self.cpts[j].get_cond_prob(entry, 0)
            p1 = p1 * self.cpts[j].get_cond_prob(entry, 1)
        p_total0 += p0
        p_total1 += p1
        #print("akbar")
    if(p_total0 >= p_total1):
        return (0,np.log(p_total0/(p_total0+p_total1)))
    else:
        return (1,np.log(p_total1/(p_total0+p_total1)))
    #return (1,np.log(p_total1/(p_total0+p_total1)))
  def predict_unobserved(self, entry, index):
    '''
    TODO
    Predicts P(A_index = 1 \mid entry)
    '''
    entry_temp= copy.deepcopy(entry)
    p_Aeq0andother=0.0
    p_Aeq1andother=0.0
    for possible_A in range(2):
        entry = copy.deepcopy(entry_temp)
        entry[index]=possible_A
        incomplete_indexes = np.where(entry == -1)[0]
        states = list(itertools.product([0, 1], repeat=len(incomplete_indexes)))
        p_total0=0.0
        p_total1=0.0
        if (states != [()]):
            for i,state in enumerate(states):
                entry[incomplete_indexes]=states[i]
                p0=self.P_c0
                p1=self.P_c1
                for z in range(len(entry)):
                    p0 = p0* self.cpts[z].get_cond_prob(entry,0)
                    p1 = p1* self.cpts[z].get_cond_prob(entry,1)
                p_total0 += p0
                p_total1 += p1
        else:
            p0 = self.P_c0
            p1 = self.P_c1
            for z in range(len(entry)):
                p0 = p0 * self.cpts[z].get_cond_prob(entry, i)
                p1 = p1 * self.cpts[z].get_cond_prob(entry, i)
            p_total0 += p0
            p_total1 += p1
        if (possible_A == 0):
            p_Aeq0andother =p_total0+p_total1
        else:
            p_Aeq1andother =p_total0+p_total1
    return (p_Aeq0andother/(p_Aeq0andother + p_Aeq1andother),p_Aeq1andother/(p_Aeq0andother + p_Aeq1andother))



# load all data
A_base, C_base = load_vote_data()


def evaluate(classifier_cls, train_subset=False):
  '''
  evaluate the classifier specified by classifier_cls using 10-fold cross
  validation
  - classifier_cls: either NBClassifier or other classifiers
  - train_subset: train the classifier on a smaller subset of the training
    data
  NOTE you do *not* need to modify this function
  '''
  global A_base, C_base

  A, C = A_base, C_base

  # score classifier on specified attributes, A, against provided labels,
  # C
  def get_classification_results(classifier, A, C):
    results = []
    pp = []
    for entry, c in zip(A, C):
      c_pred, unused = classifier.classify(entry)
      results.append((c_pred == c))
      pp.append(unused)
    # print('logprobs', np.array(pp))
    return results
  # partition train and test set for 10 rounds
  M, N = A.shape
  tot_correct = 0
  tot_test = 0
  step = int(M / 10 + 1)
  for holdout_round, i in enumerate(range(0, M, step)):
    # print("Holdout round: %s." % (holdout_round + 1))
    A_train = np.vstack([A[0:i, :], A[i+step:, :]])
    C_train = np.hstack([C[0:i], C[i+step:]])
    A_test = A[i: i+step, :]
    C_test = C[i: i+step]
    if train_subset:
      A_train = A_train[: 16, :]
      C_train = C_train[: 16]


    # train the classifiers
    classifier = classifier_cls(A_train, C_train)

    train_results = get_classification_results(classifier, A_train, C_train)
    # print(
    #    '  train correct {}/{}'.format(np.sum(train_results), A_train.shape[0]))
    test_results = get_classification_results(classifier, A_test, C_test)
    tot_correct += sum(test_results)
    tot_test += len(test_results)

  return 1.*tot_correct/tot_test, tot_test


def evaluate_incomplete_entry(classifier_cls):

  global A_base, C_base

  # train a classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  c_pred, logP_c_pred = classifier.classify(entry)
  print("  P(C={}|A_observed) = {:2.4f}".format(c_pred, np.exp(logP_c_pred)))

  return


def predict_unobserved(classifier_cls, index=11):
  global A_base, C_base

  # train a classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  a_pred = classifier.predict_unobserved(entry, index)
  print("  P(A{}=1|A_observed) = {:2.4f}".format(index+1, a_pred[1]))

  return


def main():
  '''
  TODO modify or add calls to evaluate() to evaluate your implemented
  classifiers
  Suggestions on how to use:
  ##For Q1
  print('Naive Bayes')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
  print('  10-fold cross validation total test error {:2.4f} on {} '
        'examples'.format(1 - accuracy, num_examples))
  ##For Q2
  print('Naive Bayes Classifier on missing data')
  evaluate_incomplete_entry(NBClassifier)

  index = 11
  print('Prediting vote of A%s using NBClassifier on missing data' % (
      index + 1))
  predict_unobserved(NBClassifier, index)
  ##For Q3
  print('Naive Bayes (Small Data)')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=True)
  print('  10-fold cross validation total test error {:2.4f} on {} '
        'examples'.format(1 - accuracy, num_examples))

  '''
  print('Naive Bayes')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
  print('  10-fold cross validation total test error {:2.4f} on {} '
        'examples'.format(1 - accuracy, num_examples))
  print('  10-fold cross validation total test accuracy {:2.4f} on {} '
        'examples'.format(accuracy, num_examples))
  
  print('Naive Bayes Classifier on missing data')
  evaluate_incomplete_entry(NBClassifier)
  
  
  index = 11
  print('Prediting vote of A%s using NBClassifier on missing data' % (
      index + 1))
  predict_unobserved(NBClassifier, index)
  
  print('Naive Bayes (Small Data)')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=True)
  print('  10-fold cross validation total test error {:2.4f} on {} '
        'examples'.format(1 - accuracy, num_examples))
  print('  10-fold cross validation total test accuracy {:2.4f} on {} '
        'examples'.format(accuracy, num_examples))
  
if __name__ == '__main__':
  main()
