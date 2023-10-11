from node import Node
from parse import parse
#idk if im allowed to use this
from collections import defaultdict
import math


# finds the most common class, and if the common class equals the total
def mostCommonClass(examples, default):
  currCount = -1 
  currKey = -1
  total = 0 # total number of examples (used to tell if all examples are positive, negative, or a combination of both)
  countDict = dict()

  for d in examples:
    total += 1
    if d["Class"] in countDict:
      countDict[d["Class"]] += 1
    else:
      countDict[d["Class"]] = 1

  if total == 0: 
    return [None, True]

  for key in countDict.keys():
    if currCount < countDict[key]:
      currCount = countDict[key]
      currKey = key

  return [currKey, len(countDict) == 1]   # most common class was default


#goes through every attr in examples and returns lowest attr/ent combo
def informationGain(examples, attrs):
  minEnt = 1
  bestAttr = ""
  aTypes = None
  for a in attrs:
    ent, attrTypes = attributeEntropy(examples, a)
    if ent < minEnt:
      minEnt = ent
      bestAttr = a
      aTypes = attrTypes
      

  return bestAttr, minEnt, aTypes



#verified this worked by hand w tennis example
def attributeEntropy(examples, attr):
  #can prob optimize this for space but whatever. These count how many + and - for each attr types.
  #posCounts = defaultdict(int)
  #negCounts = defaultdict(int)
  valueClassCounts = dict()
  #set containing possible attribute types (for example attribute restaraunte type has:thai,bbq,italian...)
  attrTypes = set()

  # detects if y/n (or other type of classification) besides 1 and 0
  #cType = ""
  #if examples[0]["Class"] != '1' and examples[0]["Class"] != '0':
  #  cType = examples[0]["Class"]

  #tot = 0
  for e in examples:
    attrRes = e[attr]
    if attrRes == '?':
      continue
    #if e["Class"] == '1' or e["Class"] == cType:
    #  posCounts[attrRes] += 1
    #else:
    #  negCounts[attrRes] += 1

    if attrRes not in valueClassCounts:
      valueClassCounts[attrRes] = dict()

    if e["Class"] not in valueClassCounts[attrRes]:
      valueClassCounts[attrRes][e["Class"]] = 0

    valueClassCounts[attrRes][e["Class"]] += 1
    attrTypes.add(attrRes)
    #tot += 1
  
  tot = len(examples)
  #actually doing the entropy calculation here
  ent = 0
  for t in attrTypes:
    #n = posCounts[t] + negCounts[t]
    n = sum(valueClassCounts[t].values())
    frac = n / tot

    temp_ent = 0
    #all are in the same class so 0 entropy
    #if not posCounts[t] or not negCounts[t]:
    #  continue
    #ent += frac * (-1 * posCounts[t]/n * math.log2(posCounts[t]/n) + -1 * negCounts[t]/n * math.log2(negCounts[t]/n))

    for c in valueClassCounts[t].values():
      if c:
        temp_ent += (-c/n * math.log2(c/n))
    ent += frac * temp_ent

  return ent, attrTypes
  


# gets all atributes
def getAttributes(examples):
  if not examples:
    return []
  atts = []
  for k in examples[0].keys():
    if k != 'Class':
      atts.append(k)
  return atts





def ID3(examples, default, attributes = None):
  # create initial node
  t = Node()

  # find common class
  t.label, result = mostCommonClass(examples, default)
  if result is True:  # if all examples in D are positive or negative, or if attributes is empty
    return t
  
  # checks if attributes list has already been created (this is only true if this is the first call of ID3())
  if attributes == None:
      attributes = getAttributes(examples) # only runs on initial function call
      # populate attributes list on first call
  
  if attributes == []:
    return t
    
  Astar, ent, aTypes = informationGain(examples,attributes)
  attributes.remove(Astar)
  t.label = Astar
  t.ent = ent
  # iterate over attribute types
  for a in aTypes:
    t.children[a] = None

    d_a = []
    for e in examples:
      if e[Astar] == a:
        d_a.append(e)

    # if empty, return a leaf node
    if len(d_a) == 0:
      leaf = Node()
      leaf.label  = mostCommonClass(examples,default)[0]
      t.children[a] = leaf
    else: 
      # if more exist, recursively develop tree branches
      t.children[a] = ID3(d_a, default, attributes)

  return t


  
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
  #next level node
  while node.children:  # while the current node is not a leaf
    splitVal = example[node.label]
    node = node.children[splitVal]

  return int(node.label)


#testAttr, testEnt = informationGain(parse("HW1/tennis.data"),getAttributes(parse("HW1/tennis.data")))
#print(parse("HW1/candy.data"))


