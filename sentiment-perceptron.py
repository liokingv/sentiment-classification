#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: liokingv
"""


from __future__ import division

import sys
import time
from svector import svector
from collections import defaultdict
import numpy as np



def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    v['<bias>'] = 1.0 # add bias
    return v
    
def test(devfile, model):
    predictions = []
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        pred = model.dot(make_vector(words))
        err += label * pred <= 0
        predictions.append(pred)
    return err/i, predictions  # i is |D| now

def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1. 
    model = svector() # w
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent 
        dev_err, predictions = test(devfile, model)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

       
def predict(test_data, model):
    return ["+" if model.dot(vecx) > 0 else "-" for vecx in test_data ]

        
# calculates the highest and lowest sentences 
def extreme_val(pred_list, k):
    most_pos = np.argpartition(pred_list,-k)[-k:]
    most_neg = np.argpartition(-1*pred_list,-k)[-k:]
    return most_pos, most_neg

# finds the most positive and negative words
def word_val(model):
    values = np.asarray(list(model.values()))
    values.sort()
    top_words = values[-20:]
    bottom_words = values[0:20]
    
    top_words = list(dict.fromkeys(top_words))
    bottom_words = list(dict.fromkeys(bottom_words))
    print()
    print("The top 20 most positive features are:")
    for val in top_words:
        top = [k for k, v in model.items() if v == val]
        print(str(val) + " - " + str(top))
        
    print()    
    print("The bottom 20 most negative features are:")
    for val in bottom_words:
        bottom = [k for k, v in model.items() if v == val]
        print(bottom, val)

#finds the most positive and negative sentences according to 
# averaged model
def sentence_val(predictions, sentences):
     sentence_dict = dict(zip(predictions, sentences))
     predictions.sort()
     top = predictions[-5:]
     bottom = predictions[0:5]
     print()
     print("Top 5 positive sentences:")
     print()
     for i in top:
         sent = sentence_dict[i]
         print(' '.join(str(x) for x in sent))
         print()
     print()    
     print()
     print("Top 5 negative sentences:")
     print()
     for i in bottom:
         sent = sentence_dict[i]
         print(' '.join(str(x) for x in sent))    
         print()

     
# Smart Averaged Perceptron
def train_avg(trainfile, devfile, testfile, epochs):
    w = svector()
    wa = svector()
    avg_model = svector()
    predictions = []
    sentences = []
    best_err = 1.
    c = 0
    t = time.time()
    
    for it in range(1, epochs + 1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            sentences.append(words)
            sent = make_vector(words)
            predict = w.dot(sent)
            if predict*label <=0:
                updates += 1
                w += label*sent
                wa += c*label*sent  
            c += 1
        avg_model = c*w - wa
        dev_err, predictions = test(devfile, avg_model)
        best_err = min(dev_err, best_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best avg_dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(w), time.time() - t))
    print(len(avg_model))
    word_val(w)
    sentence_val(predictions, sentences)
    
    if testfile != '':
        test_sent = []
        ratings = []
        for i, (label, words) in enumerate(read_from(testfile), 1):
            test_sent.append(words)
            vecx = make_vector(words)
            if avg_model.dot(vecx) > 0:
                ratings.append('+')       
            else:
                ratings.append('-')        
    
        print(ratings.count('+'))
        new_sent = []
        for i in range(0, 1000):
            x = test_sent[i]
            newx = ' '.join([str(item) for item in x])
            new_sent.append(newx)

    with open("test.txt.predicted", "w") as wf: #write to file
        for cid, ratings in zip(ratings, new_sent):
            print(f"{cid} {ratings}", file=wf)
            


        

if __name__ == "__main__":
   train('train.txt', 'dev.txt', 10)
   print()
   train_avg('train.txt', 'dev.txt', 'test.txt',10)
   print()

