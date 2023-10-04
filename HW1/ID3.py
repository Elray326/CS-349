from node import Node
from parse import parse
import math

# finds the most common class, and if the common class equals the total
def mostCommonClass(examples):
  neg = 0
  pos = 0
  total = 0
  for d in examples:
    total += 1
    if d["Class"] == '0':
      neg += 1
    else:
      pos += 1
  # if empty, return empty node t
  if total == 0:
    return [None, True]
  if pos - neg >= 0:
    return ['1', pos == total]
  return ['0', neg == total]

def informationGain(examples):
  pass

def ID3(examples, default):
  # create initial node
  t = Node()
  # find common class
  t.label, result = mostCommonClass(examples)
  if result is True:  # if all examples in D are positive or negative
    return t
  # continue with algorithm
  else:
    Astar = informationGain(examples)
  
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

print(parse("HW1/candy.data"))
print(mostCommonClass(parse("HW1/tennis.data")))