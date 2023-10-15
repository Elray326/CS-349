import ID3, parse, random
import matplotlib.pyplot as plt

def testID3AndEvaluate():
  data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
  tree = ID3.ID3(data, 0)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=1, b=0))
    if ans != 1:
      print("ID3 test failed.")
    else:
      print("ID3 test succeeded.")
  else:
    print("ID3 test failed -- no tree returned")

def testPruning():
  # data = [dict(a=1, b=1, c=1, Class=0), dict(a=1, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1), dict(a=0, b=0, c=0, Class=1), dict(a=0, b=0, c=1, Class=0)]
  # validationData = [dict(a=0, b=0, c=1, Class=1)]
  data = [dict(a=0, b=1, c=1, d=0, Class=1), dict(a=0, b=0, c=1, d=0, Class=0), dict(a=0, b=1, c=0, d=0, Class=1), dict(a=1, b=0, c=1, d=0, Class=0), dict(a=1, b=1, c=0, d=0, Class=0), dict(a=1, b=1, c=0, d=1, Class=0), dict(a=1, b=1, c=1, d=0, Class=0)]
  validationData = [dict(a=0, b=0, c=1, d=0, Class=1), dict(a=1, b=1, c=1, d=1, Class = 0)]
  tree = ID3.ID3(data, 0)
  ID3.prune(tree, validationData)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=0, b=0, c=1, d=0))
    if ans != 1:
      print("pruning test failed.")
    else:
      print("pruning test succeeded.")
  else:
    print("pruning test failed -- no tree returned.")


def testID3AndTest():
  trainData = [dict(a=1, b=0, c=0, Class=1), dict(a=1, b=1, c=0, Class=1), 
  dict(a=0, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1)]
  testData = [dict(a=1, b=0, c=1, Class=1), dict(a=1, b=1, c=1, Class=1), 
  dict(a=0, b=0, c=1, Class=0), dict(a=0, b=1, c=1, Class=0)]
  tree = ID3.ID3(trainData, 0)
  fails = 0
  if tree != None:
    acc = ID3.test(tree, trainData)
    if acc == 1.0:
      print("testing on train data succeeded.")
    else:
      print("testing on train data failed.")
      fails = fails + 1
    acc = ID3.test(tree, testData)
    if acc == 0.75:
      print("testing on test data succeeded.")
    else:
      print("testing on test data failed.")
      fails = fails + 1
    if fails > 0:
      print("Failures: ", fails)
    else:
      print("testID3AndTest succeeded.")
  else:
    print("testID3andTest failed -- no tree returned.")	

# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
  withPruning = []
  withoutPruning = []
  data = parse.parse(inFile)
  avgWithPruning = []
  avgWithoutPruning = []
  listOf_Xticks = []
  validationRange = 50
  for j in range(9,300, 10):
    print(j)
    listOf_Xticks.append(j+1)
    for i in range(100):
      random.shuffle(data)
      train = data[:j]
      valid = data[j+1:j+validationRange]
      test = data[j+validationRange+1:]
      #valid = data[len(data)//2:3*len(data)//4]
      #test = data[3*len(data)//4:]
      #test = data[j+1:]
    
      tree = ID3.ID3(train, 'democrat')
      #acc = ID3.test(tree, train)
      #print("training accuracy: ",acc)
      #acc = ID3.test(tree, valid)
      #print("validation accuracy: ",acc)
      #acc = ID3.test(tree, test)
      #print("test accuracy: ",acc)
    
  
      #acc = ID3.test(tree, train)
      #print("pruned tree train accuracy: ",acc)
      #acc = ID3.test(tree, valid)
      #print("pruned tree validation accuracy: ",acc)
      acc = ID3.test(tree, test)
      print("no pruning test accuracy: ",acc)
      withoutPruning.append(acc)

      new_tree = ID3.prune(ID3.ID3(train, 'democrat'), valid)
      acc = ID3.test(new_tree, test)
      print("pruned tree test accuracy: ",acc)
      withPruning.append(acc)
      
      #tree = ID3.ID3(train, 'democrat')
      
      
    print(withPruning)
    print(withoutPruning)
    avgWithPruning.append(sum(withPruning)/len(withPruning))
    avgWithoutPruning.append(sum(withoutPruning)/len(withoutPruning))
    print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))


  fig = plt.figure()
  ax1 = fig.add_subplot()
  plt.xticks(range(len(listOf_Xticks)),listOf_Xticks)
  ax1.plot(avgWithPruning, label = 'With Pruning')
  ax1.plot(avgWithoutPruning, label = 'Without Pruning')
  plt.legend(loc='upper left')
  plt.xlabel("# Training Examples")
  plt.ylabel("Accuracy on Test Data")
  plt.show()


#testPruning()
testPruningOnHouseData("HW1/house_votes_84.data")
#testID3AndEvaluate()
#testID3AndTest()
