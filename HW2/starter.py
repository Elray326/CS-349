# Spencer Rothfleisch, Louie Shapiro, Max Ward
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (float(a[i]) - float(b[i])) ** 2
    dist = dist ** 0.5
    return(dist)

# returns Cosine Similarity between vectors a dn b
def cosim(a,b, knn = True):
    dist = 0
    if knn:
        dist = sum(int(x)*int(y) for x, y in zip(a, b)) /(vecSumSqrt(a) * vecSumSqrt(b))
    else:
        dist = sum(int(x)*int(y) for x, y in zip(a, b)) /(1+(vecSumSqrt(a) * vecSumSqrt(b)))

    return(dist)

# gets the square root of the sum of the vector
def vecSumSqrt(vec):
    dist = 0
    for i in range(len(vec)):
        dist += int(vec[i]) ** 2
    return dist ** 0.5

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    train, query = pcaData(train, query)
    k = 10
    predicted = []
    actual = []
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
        if metric == 'cosim':
            distArr.sort(key=lambda x: x[1], reverse= True)
        elif metric == 'euclidean':
            distArr.sort(key=lambda x: x[1])
        # find the k lowest distances
        for m in range(k):
            trainNumber = distArr[m][0]
            weight = 1 / (distArr[m][1] + 1e-5)  # Adding a small value to avoid division by zero
            countArray[int(trainNumber)] += weight
        # figure out which number has lowest distance associated with it and put it in labels with expected
        totalCount += 1
        if countArray.index(max(countArray)) == int(query[j][0]):
            correct += 1
        # determined, actual
        print([countArray.index(max(countArray)), query[j][0]])
        predicted.append(countArray.index(max(countArray)))
        actual.append(int(query[j][0]))
    # determine percent correct
    accuracy = correct / totalCount
    print(accuracy)

    # generate and display confusion matrix
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cm_display.plot()
    title = metric + " accuracy: " + str(accuracy)
    plt.title(title)
    plt.show()
    return(predicted)



# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    k = 10
    totalCount = 0

    label_dist = [0] * k
    for label in range(len(train)):
        label_dist[int(train[label][0])] += 1
    
    print("Distribution of Number Labels:", label_dist)

    # Convert 0-255 to 1 and 0 to improve accuracy
    train = binaryConverter(train)
    query = binaryConverter(query)

    # initialize K means, each with nAttributes (should be 784)
    nAttributes = len(train[0][1])
    means = []

    trainIndex = 0
    random.shuffle(train)
    trainNums = len(train) // 10
    for i in range(k):
        s = [0] * len(train[0][1])
        mean = []
        for j in range(trainNums):
            for x in range(len(train[i][1])):
                #adding coordinate value to total coordinate value of mean
                s[x] += float(train[trainIndex][1][x])   
            trainIndex += 1
        mean = [n / trainNums for n in s]
        binaryMean = [round(n) for n in mean]
        means.append(binaryMean)
        
    nAttributes = len(means[0])
    classLabels = [0] * len(train)

    hehexd = 0
    while totalCount < 1:
        oldMeans = means[:]

        #calculate distance from each of the K means to every data point
        distMatrix  = [[0] * k for i in range(len(train))]
        for i in range(len(train)): 
            for j in range(len(means)):
                if metric == "euclidean":
                    distMatrix[i][j] = euclidean(means[j],train[i][1])
                    #assigning index of closest mean to the point
                elif metric == "cosim":
                    distMatrix[i][j] = cosim(means[j], train[i][1],False)
                    #assigning index of closest mean to the point
            if metric == "euclidean":
                idx = distMatrix[i].index(min(distMatrix[i]))
            elif metric == "cosim":
                idx = distMatrix[i].index(max(distMatrix[i]))
                    
            classLabels[i] = idx
        meanCounts = [0] * k
        meanSums = [[0] * nAttributes for i in range(k)]

        for i in range(len(classLabels)):
            mean = classLabels[i]
            for j in range(len(train[i][1])):
                #adding coordinate value to total coordinate value of mean
                meanSums[mean][j] += float(train[i][1][j])      
            meanCounts[mean] += 1
        print(meanCounts)

        zeroEle = False
        for m in meanCounts:
            if m == 0:
                zeroEle = True
        if zeroEle:
            means = []
            for i in range(k):
                means.append([])
                for j in range(nAttributes):
                    means[i].append(random.uniform(0,256))
        else:
            for i in range(k):
                meanTotal = meanSums[i]
                newMean = []

                # determining new means
                if metric == "euclidean":
                    newMean = [n / meanCounts[i] for n in meanTotal]
                else:
                    tMean = [n / meanCounts[i] for n in meanTotal]
                    for t in tMean:
                        if t < .25:
                            newMean.append(0)
                        else:
                            newMean.append(1)

                means[i] = newMean
                
        total = 0
        for i in range(len(means)):
            for j in range(len(means[0])):
                total += abs(means[i][j] - oldMeans[i][j])
        print(hehexd, total)
        hehexd += 1
        if total < 5:
            totalCount += 1
    
    meanLabels =[[0] * k for i in range(k)]
    for i in range(len(train)):
        trainMean = classLabels[i] # this is not the actual final label yet
        trainActual = int(train[i][0])
        meanLabels[trainMean][trainActual] += 1
    modes = [0] * k
    for i in range(k):
        mode = meanLabels[i].index(max(meanLabels[i]))
        modes[i] = mode

    correct = 0

    actual = []
    predicted = []

    queryLabels = [0] * len(query)
    distMatrixQ  = [[0] * k for i in range(len(query))]
    for i in range(len(query)): 
        for j in range(len(means)):
            if metric == "euclidean":
                distMatrixQ[i][j] = euclidean(means[j],query[i][1])
                #assigning index of closest mean to the point
            elif metric == "cosim":
                distMatrixQ[i][j] = cosim(means[j], query[i][1],False)
                #assigning index of closest mean to the point
        if metric == "euclidean":
            idx = distMatrixQ[i].index(min(distMatrixQ[i]))
        elif metric == "cosim":
            idx = distMatrixQ[i].index(max(distMatrixQ[i]))
                
        queryLabels[i] = idx

    for i in range(len(query)):
        trainActual = int(query[i][0])
        actual.append(trainActual)

        guess = modes[queryLabels[i]]
        predicted.append(guess)
        if trainActual == guess:
            correct += 1
    
    accuracy = correct/len(query)
    print("accuracy: ", accuracy)
    print("Modes:",modes)
    
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cm_display.plot()
    title = metric + " accuracy: " + str(accuracy)
    plt.title(title)
    plt.show()

    return(means)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)

def processDataForPCA(dataset):
    # Separate labels and image data
    labels = [item[0] for item in dataset]
    image_data = [item[1] for item in dataset]
    
    # Convert image data to a NumPy array
    image_data = np.array(image_data)
    
    return labels, image_data

def pcaData(train, valid, threshold=0.95):
    # Process training and verification datasets
    train_labels, train_data = processDataForPCA(train)
    verify_labels, verify_data = processDataForPCA(valid)

    # Standardize the data
    scaler = StandardScaler()
    # Fit the scaler on the training data only to ensure proper dimensions
    scaler.fit(train_data)  
    train_data_scaled = scaler.transform(train_data)
    verify_data_scaled = scaler.transform(verify_data)

    # Fit PCA on the training data
    pca = PCA(n_components=threshold)
    # Fit PCA only on the training data to ensure proper dimensions
    pca.fit(train_data_scaled)  

    # Transform both training and verification data
    train_data_pca = pca.transform(train_data_scaled)
    verify_data_pca = pca.transform(verify_data_scaled)

    # Recombine labels with the transformed image data
    transformedTrain = [(label, pc) for label, pc in zip(train_labels, train_data_pca)]
    transformedVerify = [(label, pc) for label, pc in zip(verify_labels, verify_data_pca)]
    return transformedTrain, transformedVerify

# converts the data set to be a hard coded 1 or 0 depending on threshold
def binaryConverter(data):
    # set border value
    borderValue = 120
    # iterate through data and convert to 1 or 0
    for i in range(len(data)):
        for j in range(len(data[0][1])):
            if int(data[i][1][j]) < borderValue:
                data[i][1][j] = 0
            else:
                data[i][1][j] = 1
    return data
            
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
    # Euclidean KNN
    print("Euclidean KNN")
    knn(read_data('train.csv'), read_data('valid.csv'), 'euclidean')
    # Cosim KNN
    print("Cosim KNN")
    knn(read_data('train.csv'), read_data('valid.csv'), 'cosim')
    # Euclidean K-Means
    print("Euclidean K-Means")
    kmeans(read_data('train.csv'), read_data('valid.csv'), 'euclidean')
    # Cosim K-Means
    print("Cosim K-Means")
    kmeans(read_data('train.csv'), read_data('valid.csv'), 'cosim')

if __name__ == "__main__":
    main()