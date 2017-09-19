import bz2, random, numpy
from random import randint

# Function to find cosine similarity
def cosinesim(v1, v2):
    return numpy.dot(v1, v2)/(numpy.linalg.norm(v1)* numpy.linalg.norm(v2))

# Get a random document
def get_doc(filename):
    with bz2.open(filename, "rt") as df:
        doc = df.readlines()
    return " ".join(doc[randint(0, len(doc))].split(","))

# Apply Chinese Restaurent Process to the document vectors
def crp(vecs):
    clusterVec = [[0.0] * 25]         # tracks sum of vectors in a cluster
    clusterIdx = [[]]        # array of index arrays. e.g. [[1, 3, 5], [2, 4, 6]]
    ncluster = 0
    # probablity to create a new table if new customer
    # is not strongly "similar" to any existing table
    pnew = 1.0/ (1 + ncluster)  
    N = len(vecs)
    rands = [random.random() for x in range(N)]         # N rand variables sampled from U(0, 1)
 
    for i in range(N):
        maxSim = -1
        maxIdx = 0
        v = vecs[i]

        for j in range(ncluster):
            sim = cosinesim(v, clusterVec[j])
            if sim > maxSim:
                maxIdx = j
                maxSim = sim
            if maxSim < pnew:
                if rands[i] < pnew:
                    clusterVec.append(v)
                    clusterIdx.append([i])
                    ncluster += 1
                    pnew = 1.0 / (1 + ncluster)
                continue

        clusterVec[maxIdx] = clusterVec[maxIdx] + v
        clusterIdx[maxIdx] = clusterIdx[maxIdx] + [i]

        if(ncluster == 0):
            ncluster += 1
 
    return clusterIdx
