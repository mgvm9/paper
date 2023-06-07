import matplotlib
from cluster import *
import numpy as np
import matplotlib.pyplot as plt 
from numpy import inf
from numpy import sqrt 
import itertools as itt
from SymmetricMatrix import SymmetricMatrix
from collections import defaultdict
from collections import Counter
from UserKNN import UserKNN

class K_IncDBSCAN:
    dataSet = []
    count = 0
    visited = []
    curCores = []
    newCores = []
    Clusters = []
    potentialCLusters = []
    Outlier = 0
    num = 0
    
    def __init__(self):
        self.Outlier = cluster('Outlier')
        
    def K_IncDBSCAN(
        self,
        D,
        eps,
        MinPts,
        K,
        ):
        
        self.D = D
        self.k = K
        self.knn_model = UserKNN(D,K)
        self.Dic = defaultdict(dict)
        self.user_sim = self.knn_model.user_sim
        for iterator in range(K):
            tuple = self.D.GetTuple(iterator)
            user = tuple[0]
            item = tuple[1]
            #update similarity-------------------
            u = self.D.GetUserInternalId(user)
            i = self.D.GetItemInternalId(item)
            self.knn_model._UpdateSimilarities(u, i)
            #------------------------------------
            self.dataSet.append([user,item])
            name = 'Cluster' + str(self.count)
            C = cluster(name)
            self.count += 1
            C.addPoint([user,item])
            self.curCores.append(tuple)
            self.Clusters.append(C)

        counter = self.D.size - self.k

        for iterator_ in range(self.k, counter):
            tuple = self.D.GetTuple(iterator_)
            user = tuple[0]
            item = tuple[1]
            #update similarity-------------------
            u = self.D.GetUserInternalId(user)
            i = self.D.GetItemInternalId(item)
            self.knn_model._UpdateSimilarities(u, i)
            #print(self.knn_model.user_sim.matrix[0])
            ##-----------------------------------  
            self.dataSet.append([user,item])
            self.incrementalAdd([user,item], eps, MinPts)
            for core in self.newCores:
                if core not in self.curCores:
                    self.curCores.append(core)
                
        Outlierpoints = []
        for Outlierp in self.Outlier.getPoints():
            Outlierpoints.append(Outlierp)
        for pts in Outlierpoints:
            for clust in self.Clusters:
                #print ('Cluster Points ')
                clust.printPoints()
                if clust.has(pts) and self.Outlier.has(pts):
                    print ('\n Point to REMOVE' + str(pts))
                    self.Outlier.remPoint(pts)    
        for clust in self.Clusters:
            #print(clust.name)
            for core in self.curCores:
                #print(core)
                help = [core[0],core[1]]
                if clust.has(help):
                    core_user = core[0]
                    self.Dic[clust.name][core_user] = 1
                    for point in self.dataSet:
                        user = point[0]
                        if(user!=core_user):
                            temp = self.user_sim.Get(core_user,user)
                            self.Dic[clust.name][user] = temp
        #print(self.Dic)

        n_zeros = np.count_nonzero(self.knn_model.user_sim.matrix==0)
        print("zeros")
        print(n_zeros)
        print("non zeros")
        not_zero =  np.count_nonzero(self.knn_model.user_sim.matrix!=0)
        print(not_zero)

          #___________________________________________________________________________________________________________________________________________________________________________________

    def expandCluster(
        self,
        point,
        NeighbourPoints,
        C,
        eps,
        MinPts,
        ):
        self.visited = []

        C.addPoint(point)

        for p in NeighbourPoints:
            if p not in self.visited:
                self.visited.append(p)
                np = self.regionQuery(p, eps)
                if len(np) >= MinPts:
                    for n in np:
                        if n not in NeighbourPoints:
                            NeighbourPoints.append(n)

            for c in self.Clusters:
                if not c.has(p):
                    if not C.has(p):
                        C.addPoint(p)

            if len(self.Clusters) == 0:
                if not C.has(p):
                    C.addPoint(p)

        self.Clusters.append(C)
        #print ('\n' + C.name + '\n')
        C.printPoints()
    #_________________________________________________________________________________________________________________________

    def expandCluster2(
        self,
        point,
        NeighbourPoints,
        C,
        eps,
        MinPts,
        ):
        self.visited = []

        C.addPoint(point)

        for p in NeighbourPoints:
            if p not in self.visited:
                self.visited.append(p)
                np = self.regionQuery(p, eps)
                if len(np) >= MinPts:
                    for n in np:
                        if n not in NeighbourPoints:
                            NeighbourPoints.append(n)

            for c in self.potentialCLusters:
                if not c.has(p):
                    if not C.has(p):
                        C.addPoint(p)

            if len(self.potentialCLusters) == 0:
                if not C.has(p):
                    C.addPoint(p)

        self.potentialCLusters.append(C)
        #print ('\n' + C.name + '\n')
        C.printPoints() 
        

    #____________________________________________________________________________________________________________________________________________________________________________________
    def regionQuery(self, P, eps):
        user = P[0]
        item = P[1]
        result = []
        similarities = []
        temp_Dic = defaultdict(dict)
        u = self.D.GetUserInternalId(user)
        for point in self.dataSet:
            temp_user = point[0]
            temp_user = self.D.GetUserInternalId(temp_user)
            temp_sim = self.user_sim.Get(u,temp_user)
            if temp_sim >= eps:
                result.append([point[0],point[1]])
            
        return result
    #_________________________________________________________________________________________________________________________________________________________________________________________
    def findNearestCluster(self, p):
        closest = self.curCores[0]
        user = p[0]
        u = self.D.GetUserInternalId(user)
        for core in self.curCores:
            user_closest = closest[0]
            user_core = core[0]
            u_closest = self.D.GetUserInternalId(user_closest)
            u_core = self.D.GetUserInternalId(user_core)
            if(self.knn_model.Get(u,u_closest)< self.knn_model.Get(u,u_core)): 
                closest = core
        return closest
    #___________________________________________________________________________________________________________________________________________________
    def incrementalAdd(
        self,
        p,
        eps,
        Minpts,
        ):
        self.num = self.num + 1
        #print ('\nADDING point ' + str(self.num))
        self.visited = []
        self.newCores = []
        UpdSeedIns = []
        foundClusters = []
        potentialSize = 1
        NeighbourPoints = self.regionQuery(p, eps)
        #if the p has enough points in its neighborhood had it to the new core record 
        if len(NeighbourPoints) >= Minpts:
            self.newCores.append(p)
        self.visited.append(p)
        potentialSize += len(NeighbourPoints)
        #visit the neighbors of p
        for pt in NeighbourPoints:
            #if the neighbor of the neighbor hasnt been visited before add it to the visited
            
            #print(self.visited)
            #print(pt)
            
            if pt not in self.visited:
                self.visited.append(pt)
                np = self.regionQuery(pt, eps)
                if len(np) >= Minpts:
                    #if the neighborhood of the neighbor is big enough add each of said neighbor of neighbors to the neighborhood of p
                    for n in np:
                        if n not in NeighbourPoints:
                            NeighbourPoints.append(n)
                    #if p neighbor with a big enough neighborhood isnt in the Core record add to the new core record        
                    if pt not in self.curCores:
                        self.newCores.append(pt)
                    potentialSize += len(np)    
        #for each pof these new cores                
        for core in self.newCores:
            corehood = self.regionQuery(core, eps)
            #print ('the corehood is:', corehood)
            for elem in corehood:
                #print ('The Minpts are:', Minpts)
                #print (self.regionQuery(elem, eps))
                if len(self.regionQuery(elem, eps)) >= Minpts:
                    #had these new cores to the updateSeedList
                    if elem not in UpdSeedIns:
                        UpdSeedIns.append(elem)

        #if there are no points in UpdateSeedList p is an outlier
        if len(UpdSeedIns) < 1:
            self.Outlier.addPoint(p)
        else:
            #there are Seeds for the update
            findCount = 0
            #If any of the seed is in an already existate cluster update findcount, if said cluster isnt in found clusters add
            for seed in UpdSeedIns:
                for clust in self.Clusters:
                    if clust.has(seed):
                        findCount += 1
                        if clust.name not in foundClusters:
                            foundClusters.append(clust.name)
                            break
            #Creating a new cluster
            # if the seeds werent found in any existing cluster we need to create a new one          
            if len(foundClusters) == 0:
                flag = 0
                for clust in self.Clusters:
                    if(clust.size() < Minpts):
                        for pt in clust.getPoints():
                            self.Outlier.addPoint(pt)
                        self.Clusters.remove(clust)
                        self.count += 1
                        name = "Cluster" + str(self.count)
                        C = cluster(name)
                        self.expandCluster(UpdSeedIns[0],self.regionQuery(UpdSeedIns[0],eps), C, eps, Minpts)
                        flag = 1
                        #print("Deleting CLuster thats too small and expanding P")
                        break
                if(flag == 0) :
                    smallest = self.Clusters[0]
                    #First look for the smallest cluster already created
                    for clust in self.Clusters:
                        if(clust.size() < smallest.size()):
                            smallest=clust 
                    if(potentialSize > smallest.size()):
                        closest = self.Clusters[0]
                        #original = self.Clusters[0]
                        min_dist = inf
                        for pt in smallest.getPoints():
                            u, i = self.D.AddFeedback(pt[0], pt[1])
                            for clust in self.Clusters:
                                for point in clust.getPoints():
                                    u_, i_ = self.D.AddFeedback(point[0], point[1])
                                    if(self.user_sim.Get(u,u_) <min_dist and clust != smallest):
                                        closest = clust
                                        original = clust
                                        min_dist = self.user_sim.Get(u,u_)                  
                        if(min_dist <= eps):
                            #merge them
                            master = closest
                            #print('close')
                            closest.printPoints()
                            #print('small')
                            smallest.printPoints()
                            for pt in smallest.getPoints():
                                if not master.has(pt):
                                 master.addPoint(pt)   
                            self.Clusters.remove(smallest)
                            self.Clusters.remove(original)
                            self.Clusters.append(master)
                            self.count += 1
                            name = "Cluster" + str(self.count)
                            C = cluster(name)
                            self.expandCluster(UpdSeedIns[0],self.regionQuery(UpdSeedIns[0],eps), C, eps, Minpts)
                            #print("Merging smallest with closest and expanding p")
                        else:
                            self.potentialCLusters.append(smallest)
                            for pt in smallest.getPoints():
                                self.Outlier.addPoint(pt)
                            self.Clusters.remove(smallest)
                            self.count += 1
                            name = "Cluster" + str(self.count)
                            C = cluster(name)
                            self.expandCluster(UpdSeedIns[0],self.regionQuery(UpdSeedIns[0],eps), C, eps, Minpts)
                            #print("Deleting smallest and expanding p")
                    else:
                        self.count +=1
                        name = "Cluster" + str(self.count)
                        C = cluster(name)
                        self.expandCluster2(UpdSeedIns[0],self.regionQuery(UpdSeedIns[0],eps), C, eps, Minpts)
                        self.Outlier.addPoint(p)
                        #print("p becomes outlier")
            #Adding a Point P to already existing cluster                      
            elif len(foundClusters) == 1:
                originalCluster = -1
                newCluster = -1
                for c in self.Clusters:
                    if c.name == foundClusters[0]:
                        originalCluster = c
                        newCluster = c
                newCluster.addPoint(p)
                #Replace with a need freshly made cluster
                if len(UpdSeedIns) > findCount:
                    for seed in UpdSeedIns:
                        if not newCluster.has(seed):
                            newCluster.addPoint(seed)
                self.Clusters.remove(originalCluster)
                self.Clusters.append(newCluster)
            else:
                if len(self.potentialCLusters) > 0 :
                    master_name = foundClusters[0]
                    master = self.Clusters[0]
                    original = self.Clusters[0]
                    foundClusters.pop(0)
                    for clust in self.Clusters:
                        if clust.name == master_name:
                            master = clust
                            original = clust
                    while(len(foundClusters)>0 and len(self.potentialCLusters)>0):
                        merge_name = foundClusters[0]
                        foundClusters.pop(0)
                        original_merge = self.Clusters[0]
                        merge = self.Clusters[0]
                        for clust in self.Clusters:
                            if clust.name == merge_name:
                                original_merge = clust
                                merge = clust
                        for pt in merge:
                            if not master.has(pt):
                                master.addPoint(pt)
                        toadd = self.potentialCLusters[0]
                        self.Clusters.append(toadd)        
                        self.potentialCLusters.remove(toadd)
                        self.Clusters.remove(original_merge)
                                       

                    for seed in UpdSeedIns:
                        if not master.has(seed):
                            master.addPoint(seed)
                    master.addPoint(p)
                    self.Clusters.remove(original)
                    self.Clusters.append(master) 


                
                else:
                    smallest1_name = foundClusters[0]
                    for clust in self.Clusters:
                        if clust.name == smallest1_name:
                            smallest1 = clust
                    minsize = smallest1.size()
                    for clust in foundClusters:
                        for c in self.Clusters:
                            if c.name == clust:
                                size = c.size()
                                if size < minsize:
                                    smallest1 = c
                                    minsize = size
                    master = smallest1        
                    for seed in UpdSeedIns:
                        if not master.has(seed):
                            master.addPoint(seed)    
                    master.addPoint(p)  
                    self.Clusters.remove(smallest1)
                    self.Clusters.append(master)

