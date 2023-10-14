from node import Node
from parse import parse
#idk if im allowed to use this
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
    return [None, True, 0]

  for key in countDict.keys():
    if currCount < countDict[key]:
      currCount = countDict[key]
      currKey = key
    elif currCount == countDict[key]:
      currKey = default
      
  return [currKey, len(countDict) == 1, currCount]   # most common class was default


# finds the most common attribute value for a given attribute
def mostCommonAttributeValue(examples, default, attribute):
  currCount = -1 
  currKey = -1
  total = 0 # total number of examples (used to tell if all examples are positive, negative, or a combination of both)
  countDict = dict()

  for d in examples:
    if not d[attribute] == '?':
      if d[attribute] in countDict:
        countDict[d[attribute]] += 1
      else:
        countDict[d[attribute]] = 1

#  if total == 0: 
#    return [None, True]

  for key in countDict.keys():
    if currCount < countDict[key]:
      currCount = countDict[key]
      currKey = key

  return currKey


def postOrderPrune(tree, validation):
  for n in tree.children.values():
    postOrderPrune(n)
  #non leaf node
  if tree.children:
    #currentRes is non pruned
    currentRes = test(tree,validation)
    tempChildren = tree.children.copy()
    tempLabel = tree.label
    #prune it and test
    tree.children = {}
    #stored a new node variable that stores most common class at tree level (make node a leaf node)
    tree.label = tree.commonClass
    newRes = test(tree,validation)

    #if we dont improve, revert to original
    if newRes < currentRes:
      tree.children = tempChildren
      tree.label = tempLabel

  print(tree.label)


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

# calculates the count of each class for entropy and pruning functions
def calculateClassCounts(examples, attr):
  valueClassCounts = dict()
  attrTypes = set()
  tot = 0
  # Case for handling empty attributes
  for e in examples:
    attrRes = e[attr]
    if attrRes == '?':
      continue

    if attrRes not in valueClassCounts:
      valueClassCounts[attrRes] = dict()

    if e["Class"] not in valueClassCounts[attrRes]:
      valueClassCounts[attrRes][e["Class"]] = 0

    valueClassCounts[attrRes][e["Class"]] += 1
    attrTypes.add(attrRes)
    tot += 1
  return valueClassCounts, attrTypes, tot

#verified this worked by hand w tennis example
def attributeEntropy(examples, attr):

  valueClassCounts, attrTypes, tot = calculateClassCounts(examples, attr)

  #actually doing the entropy calculation here
  ent = 0
  for t in attrTypes:
    n = sum(valueClassCounts[t].values())
    try:
      frac = n / tot
    except:
      Exception()

    temp_ent = 0
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

# Implementation of the ID3 algorithm
def ID3(examples, default, attributes = None):
  # create initial node
  t = Node()
  # find common class
  t.label, result, count = mostCommonClass(examples, default)
  t.commonClass = t.label
  if result is True:  # if all examples in D are positive or negative, or if attributes is empty
    return t
  
  # checks if attributes list has already been created (this is only true if this is the first call of ID3())
  if attributes == None:
      attributes = getAttributes(examples) # only runs on initial function call
      # populate attributes list on first call
  
  if attributes == []:
    return t
    
  Astar, ent, aTypes = informationGain(examples,attributes)
  newAttributes = attributes[:]
  newAttributes.remove(Astar)
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
      t.children[a] = ID3(d_a, default, newAttributes)
    mostCommonAtype = mostCommonAttributeValue(examples, default, Astar)
  t.children['?'] = t.children[mostCommonAtype]

  return t


  
'''
Takes in an array of examples, and returns a tree (an instance of Node) 
trained on the examples.  Each example is a dictionary of attribute:value pairs,
and the target class variable is a special attribute with the name "Class".
Any missing attributes are denoted with a value of "?"
  '''

def prune(node, examples, attributes = None):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  # Critical Value Pruning
  if not node.children:  # If it's a leaf node
      return node
  
  if attributes == None:
    attributes = getAttributes(examples)

  Astar, _, aTypes = informationGain(examples, attributes)

  observed = observedFrequencies(examples, Astar, aTypes)
  expected = expectedFrequencies(examples, observed)

  # For each primary key in observed make sure its in expected as well
  for primaryKey in observed:
      if primaryKey not in expected:
          expected[primaryKey] = {}  # Initialize primary key in expected if it doesnt exist

      for secondaryKey in observed[primaryKey]:
          if secondaryKey not in expected[primaryKey]:
              expected[primaryKey][secondaryKey] = 0  # Initialize with default value of 0 (check for divide by 0 error in chi square calc)

  chiSquare = chiSquareHelper(observed, expected)
  criticalValue = 3.841  # Chi Square Value for alpha=0.05 and df=1

  if chiSquare < criticalValue:
      node.children = {}  # prune all children
      node.label = mostCommonClass(examples, 0)[0]
      return node
  
  # If not pruned, recursively check the child nodes
  for value in node.children.keys():
      subset = [x for x in examples if x[Astar] == value]
      prune(node.children[value], subset, attributes)
  
  return node


# calculates chi square
def chiSquareHelper(observed, expected):
  chiSquare = 0
  # calculate chi square using all observed and expected frequencies
  for value in observed.keys():
        for k in observed[value].keys():
          if expected[value][k] == 0:
            if observed[value][k] == 0:
              continue  # Both observed and expected are 0; contribution to chi-squared is 0
            else:
              chiSquare += float('inf')  # Max significance
          else:
            chiSquare += ((observed[value][k] - expected[value][k]) ** 2 ) / expected[value][k]
  return chiSquare

# calculates the observed frequencies for each attribute
def observedFrequencies(examples, attribute, values):
  observed = {}
  for value in values:
      subset = [x for x in examples if x[attribute] == value]
      label, _, frequency = mostCommonClass(subset, None)
      observed[value] = {label: frequency}
  return observed

# calculates the expected frequency
def expectedFrequencies(examples, observed):
    total = len(examples)
    label, _, frequency = mostCommonClass(examples, 0)
    
    expected = {}
    for value, count in observed.items():
        expected[value] = {label: (frequency/total) * sum(count.values())}
    return expected

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  tot = 0
  correct = 0
  for e in examples:
    c = e["Class"]
    feature = dict((i, e[i]) for i in e if i != "Class")
    ans = evaluate(node, feature)
    if ans == c:
      correct += 1
    tot += 1
  
  return correct/tot
    


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''  
  #next level node
  while node.children:  # while the current node is not a leaf
    splitVal = example[node.label]
    if splitVal not in node.children.keys():
      splitVal = '?'
    node = node.children[splitVal]

  # if can be casted to int, return int, otherwise return as string
  try:
    return int(node.label)
  except:
    return node.label 


#testAttr, testEnt = informationGain(parse("HW1/tennis.data"),getAttributes(parse("HW1/tennis.data")))
#print(parse("HW1/candy.data"))


