import sys
import scipy.io.arff as sia
import numpy as np
import random
import math
import pylab


def errMsg():
    print 'Usage: python dt-learn.py <train-set-file> <test-set-file> m'

def compEntropy(y):
    count = {i:0 for i in labels}
    nTotal = len(y)
    for yi in y:
        count[yi] += 1
    
    temp = [-count[i]/float(nTotal)*np.log2(count[i]/float(nTotal)) for i in count if count[i]!=0] # log2(x) = log(x)/log(2)
    return sum(temp), count
    
def compCondEntropy(i, xi, y):
    temp = [sum(xi==j)*compEntropy(y[xi==j])[0] for j in classes[i][1]]
    return sum(temp)/float(len(y))
    
def findBestSplit(xi, y):
    n = len(y)
    min_dummyH = n
    split_pos = -1
    for i in xrange(1,n):
        if xi[i] != xi[i-1]:
            tmp_dummyH = i*compEntropy(y[:i])[0] + (n-i)*compEntropy(y[i:])[0]
            if tmp_dummyH < min_dummyH:
                min_dummyH = tmp_dummyH
                split_pos = i
    return split_pos, min_dummyH/float(n)
            
def hasNoDiffFeats(dataset):
    for i in xrange(1, dataset.shape[0]):
        if min(dataset[0] == dataset[i]) == True and max(dataset[0] == dataset[i]) == True:
            continue
        else:
            return False
    return True
      
def DecisionTree(trSet, m):
    ## empty node
    if trSet.shape[0] == 0:
        return None
    else:
        if len(trSet.shape) == 1:
            trSet = trSet.reshape(-1, trSet.size)
        H, L = compEntropy(trSet[:,-1])
        L0 = {i:0 for i in labels}
        # find the plurality class of this node
        plurality = labels[0]
        for i in labels:
            if L[i] > L[plurality]:
                plurality = i

        if max(trSet[:,-1]) == min(trSet[:,-1]) or trSet.shape[0]<m  \
                                     or hasNoDiffFeats(trSet[:,:-1]):
            return ('Leaf', plurality, L);
    
        else:
            bestFeatId = -1
            maxInfoGain = -1
            for i in xrange(nFeats): # i-th feature                
                if classes[i][0] == 'numeric':
                    assert classes[i][1] == None
                    trSetToFloat = np.array(trSet[:,i], dtype = 'float64')
                    trSetSorted  = trSet[trSetToFloat.argsort()] 
                    s, H_i = findBestSplit(trSetSorted[:,i], trSetSorted[:,-1])

                    if H-H_i > maxInfoGain:
                        maxInfoGain = H - H_i
                        bestFeatId = i
                        
                        threshold = 0.5 * (float(trSetSorted[:,i][s]) + float(trSetSorted[:,i][s-1]))
                        
                    
                elif classes[i][0] == 'nominal':
                    trSetSorted = trSet[trSet[:,i].argsort()] 
                    H_i = compCondEntropy(i, trSetSorted[:,i], trSetSorted[:,-1])

                    if H-H_i > maxInfoGain:
                        maxInfoGain = H - H_i
                        bestFeatId = i
                    
            if maxInfoGain <= 0:
                return ('Leaf', plurality, L);            
         
            ## divide the data into branches
            if classes[bestFeatId][0] == 'numeric':
                temp = np.array(trSet[:,bestFeatId], dtype = 'float64')
                leftSubTree  = DecisionTree(trSet[temp <= threshold, :], m)
                rightSubTree = DecisionTree(trSet[temp  > threshold, :], m)
                if leftSubTree == None:
                    #leftSubTree = ('Leaf', plurality, L0)
                    leftSubTree = ('Leaf', labels[0], L0)
                if rightSubTree == None:
                    #rightSubTree = ('Leaf', plurality, L0)
                    rightSubTree = ('Leaf', labels[0], L0)
                # return ((feats[bestFeatId], threshold, L), (leftSubTree, rightSubTree))
                return ((bestFeatId, threshold, L), (leftSubTree, rightSubTree))
                
            elif classes[bestFeatId][0] == 'nominal':
                subTrees = []
                for attri in classes[bestFeatId][1]:
                    child = DecisionTree(trSet[trSet[:,bestFeatId] == attri, :], m)
                    if child == None:
                        #child = ('Leaf', plurality, L0)
                        child = ('Leaf', labels[0], L0)
                    subTrees.append(child)
                # return ((feats[bestFeatId], None, L), tuple(subTrees))
                return ((bestFeatId, None, L), tuple(subTrees))

def traverseDecisionTree(X, dTree):

    # if not reach leaves
    if len(dTree) == 2:
        featId = dTree[0][0]
        
        # nominal attributes
        if dTree[0][1] == None:
            for i in xrange(len(classes[featId][1])):
                if classes[featId][1][i] == X[featId]:
                    classId = i
                    break
            # get to classId-th subtree
            y_pred = traverseDecisionTree(X, dTree[1][classId])
            
        # numerical attributes
        else:
            #if X[featId] <= dTree[0][1]:
            if float(X[featId]) <= dTree[0][1]:
                # get to left subTree
                y_pred = traverseDecisionTree(X, dTree[1][0])
            else:
                # get to right subTree
                y_pred = traverseDecisionTree(X, dTree[1][1])
                
        return y_pred
    # get to leaves    
    else:
        return dTree[1] 
               

def testTree(tSet, dtree, display = True):
    Nt = tSet.shape[0]
    nCorrect = 0
    if display:
        print "---------------------------------------------------------------"
        print " Testing Results:"
        print "-"*49
        print ('{0:15s}|{1:16s}|{2:16s}'.format("Test sample id", "actual label", "predicted label"))
        print "-"*49
    for i in xrange(Nt): 
        y_pred = traverseDecisionTree(tSet[i,:-1], dtree)
        if display:
            print('{0:<15d}|{1:16s}|{2:16s}'.format(i+1, tSet[i,-1], y_pred))
        if y_pred == tSet[i,-1]:
            nCorrect += 1
    if display:
        print 
        print " # of correct classified instances = ", nCorrect
        print "   # of total classified instances = ", Nt  
        print "---------------------------------------------------------------" 
    return nCorrect, Nt  
        
        
def visTree(dtree, level=0):
    indent = ('|'+' '*8)*level
    # tree
    if len(dtree) == 2: 
        # nominal attributes    
        if dtree[0][1] == None: #if classes[dtree[0]][0]][0] == 'nominal': 
            for i in range(len(classes[dtree[0][0]][1])): 
                print indent + feats[dtree[0][0]] + ' = ' + classes[dtree[0][0]][1][i],   
                if dtree[1][i][0] == 'Leaf':      
                    try: # non-empty leaf
                        print ' ', dtree[1][i][2], ': ' + dtree[1][i][1]
                                
                    except: # empty leaf
                        print  ' : ' + dtree[1][i][1]
                else:
                    print ' ', dtree[1][i][0][2]
                    visTree(dtree[1][i], level+1)

                
        # numeric attributes
        else: 
            print indent + feats[dtree[0][0]] + ' <= ' + str(dtree[0][1]),    
            if dtree[1][0][0] == 'Leaf':  
                try: # non-empty leaf
                    print ' ', dtree[1][0][2], ': ' + dtree[1][0][1]
                                
                except: # empty leaf
                    print  ' : ' + dtree[1][0][1]
            else:
                print ' ', dtree[1][0][0][2]
                
                visTree(dtree[1][0], level+1)

            
            print indent + feats[dtree[0][0]] + ' > ' + str(dtree[0][1]),
            if dtree[1][1][0] == 'Leaf':       
                try: # non-empty leaf
                    print ' ', dtree[1][1][2], ': ' + dtree[1][1][1]
                                
                except: # empty leaf
                    print  ' : ' + dtree[1][1][1]
            else:
                print ' ', dtree[1][1][0][2]       
                #print indent*level,
                visTree(dtree[1][1], level+1) 
    else:
        pass
                     
def drawLearningCurves(train, test, fraction, times = 1, m = 4):
    Ntr = train.shape[0]
    Nt = test.shape[0]
    table = np.zeros([len(fraction),4])
    for i in xrange(len(fraction)):
        nCorrect = []
        if fraction[i] == 1:
            tree = DecisionTree(train, m)
            nCorrect.append(testTree(test, tree, False)[0])           
        else: 
            for t in xrange(times):
                random.seed(t)
                inds = random.sample(range(Ntr), int(math.ceil(fraction[i]*Ntr)))

                trainSampled = train[inds] 
                tree = DecisionTree(trainSampled, m)
                nCorrect.append(testTree(test, tree, False)[0])
        table[i,:] = np.array([fraction[i], min(nCorrect)/float(Nt), np.mean(nCorrect)/float(Nt), max(nCorrect)/float(Nt)])
    
    pylab.figure(1)
    pylab.plot(table[:,0], table[:,1], 'rx')
    pylab.plot(table[:,0], table[:,1], 'r', label = 'minimum')
    pylab.hold(True)
    pylab.plot(table[:,0], table[:,2], 'gx')
    pylab.plot(table[:,0], table[:,2], 'g', label = 'average')
    pylab.hold(True)
    pylab.plot(table[:,0], table[:,3], 'bx')
    pylab.plot(table[:,0], table[:,3], 'b', label = 'maximum')
    
    pylab.title("Learning Curve\n (m = %d)" %m)
    pylab.xlabel("# of training samples in percentage")
    pylab.ylabel("test-set accuracy")
    #pylab.legend(["minimum", "average", "maximum"], loc = "lower right")
    pylab.legend(loc = 'lower right')
    #pylab.savefig("prob3-2.jpg")
    #pylab.show() 
    
def treeSizeTuning(train, test, m):

    accuracy = []
    for mi in m:
        tree = DecisionTree(train, mi)
        nCorrect, Nt = testTree(test, tree, False)
        accuracy.append(nCorrect/float(Nt))

    pylab.figure(2)
    pylab.plot(m, accuracy, 'bx')
    pylab.plot(m, accuracy, 'r')
    pylab.title("Tuning the size of the tree")
    pylab.xlabel("m")
    pylab.ylabel("test-set accuracy")
    #pylab.figure()
    #pylab.savefig("prob3-2.jpg")      
    
                                                                                                                                                          
args = [arg for arg in sys.argv]
try:
    trainFile = args[1]
    testFile = args[2] 
    m = int(args[3])

    ## load training and test data
    trainData = sia.loadarff(trainFile)
    testData = sia.loadarff(testFile)  
    
    ## reshape the datasets
    train = np.array([[i for i in trainData[0][j]] for j in range(trainData[0].shape[0])])
    test = np.array([[i for i in testData[0][j]] for j in range(testData[0].shape[0])])

    ## get the feature names and the class names
    feats = trainData[1].names()
    classes = [trainData[1][feat] for feat in trainData[1].names()]
    labels = classes[-1][1]
    
    nFeats = len(feats)-1
    nLabels = len(labels)
    tree = DecisionTree(train, m)
    
    ## visualize the tree structure
    print "---------------------------------------------------------------"
    print " Visualize the decision tree:"
    print "-"*49
    visTree(tree, 0)
    
    ## show test results
    testTree(test, tree)
    
    ## [P2] plot learning curves
    runtimes = 10
    sampleRatio = [0.05, 0.1, 0.2, 0.5, 1]
    drawLearningCurves(train, test, sampleRatio, runtimes)
    
    ## [P3] plot test-set accuracy vs size of the tree 
    mlist = [2, 5, 10, 20]   
    treeSizeTuning(train, test, mlist)
    
    pylab.show()
except:
    errMsg()
    
    

''' test the code '''
'''
if 1:
    trainFile = "./heart_train.arff"
    testFile = "./heart_test.arff"
    m = 2
    ## load training and test data
    trainData = sia.loadarff(trainFile)
    testData = sia.loadarff(testFile)  
    
    ## reshape the datasets
    train = np.array([[i for i in trainData[0][j]] for j in range(trainData[0].shape[0])])
    test = np.array([[i for i in testData[0][j]] for j in range(testData[0].shape[0])])

    ## get the feature names and the class names
    feats = trainData[1].names()
    classes = [trainData[1][feat] for feat in trainData[1].names()]
    labels = classes[-1][1]
    
    nFeats = len(feats)-1
    nLabels = len(labels)
    tree = DecisionTree(train, m)
    
    ## show test results
    testTree(test, tree)
'''

