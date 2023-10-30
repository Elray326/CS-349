import numpy as np
from numpy.linalg import norm
from sklearn import metrics


# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (int(a[i]) - int(b[i])) ** 2
    dist = dist ** 0.5
    return(dist)

"""
# Euclidian distance test
point1 = np.array((1, 2, 3, 11, 22, 3, 1, -155))
point2 = np.array((1, 1, 1, 5, 3, 4, 10, 8))
 
# calculating Euclidean distance
# using linalg.norm()
dist_ref = np.linalg.norm(point1 - point2)
dist = euclidean(point1, point2)

# printing Euclidean distance
print("Reference:",dist_ref)
print("Ours:",dist)
"""

# returns Cosine Similarity between vectors a dn b
def cosim(a,b):
    intA = [int(x) for x in a]
    intB = [int(x) for x in b]
    dist = sum(intA * intB) / (vecSumSqrt(a) * vecSumSqrt(b))

    return(dist)

# gets the square root of the sum of the vector
def vecSumSqrt(vec):
    dist = 0
    for i in range(len(vec)):
        dist += int(vec[i]) ** 2
    return dist ** 0.5

"""
# Cosine Similarity Test
# define two lists or array
A = np.array([2,1,2,3,2,9])
B = np.array([3,4,2,4,5,5])

 
# compute cosine similarity
ref_cosine = np.dot(A,B)/(norm(A)*norm(B))
cosine = cosim(A,B)
print("Reference Cosine Similarity:", ref_cosine)
print("Cosine Similarity:", cosine)
"""

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    k = 20
    labels = []
    totalCount = 0
    correct = 0
    for j in range(len(query)):
        distArr = []
        countArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(train)): #loop over the entire training set for each query example
            currTrainNum = train[i][0]
            if metric == 'euclidean':
                # array with the number displayed in the training image and the distance
                distArr.append([currTrainNum, euclidean(train[i][1], query[j][1])])
            elif metric == 'cosim':
                distArr.append([currTrainNum, cosim(train[i][1], query[j][1])])
        
        #sorts the array in order by distance
        distArr.sort(key=lambda x: x[1])
        # find the k lowest distances
        for m in range(k):
            trainNumber = distArr[m][0]
            countArray[int(trainNumber)] += 1
        # figure out which number has lowest distance associated with it and put it in labels with expected
        totalCount += 1
        if countArray.index(max(countArray)) == int(query[j][0]):
            correct += 1
        print([countArray.index(max(countArray)), query[j][0]])
        labels.append([countArray.index(max(countArray)), query[j][0]])
    # determine percent correct and return labels
    print(correct / totalCount)
    confusion_matrix = metrics.confusion_matrix(labels, predicted)
    return(labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    return(labels)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            #trimmed 100 out of each side
            for i in range(100,684):
                #don't add one out of every 4 pixels
                if i % 4 != 0:
                    attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    #show('valid.csv','pixels')
    # print(read_data('valid.csv')[0][1])
    knn(read_data('train.csv'), read_data('valid.csv'), 'euclidean')
    knn(read_data('train.csv'), read_data('valid.csv'), 'cosim')

if __name__ == "__main__":
    main()