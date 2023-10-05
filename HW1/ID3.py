from node import Node
from parse import parse
#idk if im allowed to use this
from collections import defaultdict
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

#goes through every attr in examples and returns lowest attr/ent combo
def informationGain(examples, attrs):
  minEnt = 1
  bestAttr = ""
  for a in attrs:
    ent = attributeEntropy(examples, a)
    if ent < minEnt:
      minEnt = ent
      bestAttr = a
      

  return bestAttr, minEnt


#verified this worked by hand w tennis example
def attributeEntropy(examples, attr):
  #can prob optimize this for space but whatever. These count how many + and - for each attr types.
  posCounts = defaultdict(int)
  negCounts = defaultdict(int)
  #set containing possible attribute types (for example attribute restaraunte type has:thai,bbq,italian...)
  attrTypes = set()

  tot = 0
  for e in examples:
    attrRes = e[attr]
    if attrRes == '?':
      continue
    if e["Class"] == '1':
      posCounts[attrRes] += 1
    else:
      negCounts[attrRes] += 1
    attrTypes.add(attrRes)
    tot += 1
  #actually doing the entropy calculation here
  ent = 0
  for t in attrTypes:
    n = posCounts[t] + negCounts[t]
    frac = n / tot

    #all are in the same class so 0 entropy
    if not posCounts[t] or not negCounts[t]:
      continue
    ent += frac * (-1 * posCounts[t]/n * math.log2(posCounts[t]/n) + -1 * negCounts[t]/n * math.log2(negCounts[t]/n))
  return ent




  

def getAttributes(examples):
  if not examples:
    return []
  atts = []
  for k in examples[0].keys():
    if k != 'Class':
      atts.append(k)
  return atts


def ID3(examples, default):
  # create initial node
  t = Node()
  # find common class
  t.label, result = mostCommonClass(examples)
  if result is True:  # if all examples in D are positive or negative
    return t
  # continue with algorithm
  else:
    Astar = informationGain(examples,getAttributes(examples))
  
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
testAttr, testEnt = informationGain(parse("HW1/tennis.data"),getAttributes(parse("HW1/tennis.data")))
print(testAttr, testEnt)
#print(parse("HW1/candy.data"))
print(mostCommonClass(parse("HW1/tennis.data")))
print(getAttributes(parse("HW1/candy.data")))