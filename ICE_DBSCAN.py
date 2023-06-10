from numba import njit
from collections import defaultdict
from implicit_data import ImplicitData
import numpy as np
from BISGD import BISGD
import math
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from numba import njit
import matplotlib
from cluster import *
import numpy as np
import matplotlib.pyplot as plt 
from numpy import inf
from numpy import sqrt 
import itertools as itt
from collections import Counter
from UserKNN import UserKNN
import math
import time
from array import *



class ICE_DBSCAN(BISGD):
    def __init__(self, data: ImplicitData, eps: float = 0.25, MinPts: int = 3, k: int = 10,
        num_clusters: int = 10, cl_num_iterations: int = 10, cl_learn_rate: float = 0.01, cl_regularization: float = 0.1,         
        num_factors: int = 10, num_iterations: int = 10, learn_rate: float = 0.01, regularization: float = 0.1, 
        random_seed: int = 1):
        """    Constructor.
        Keyword arguments:
        data -- ImplicitData object
        num_clusters -- Number of clusters (int, default 10)
        cl_num_iterations -- Number of iterations of the clustering algorithm (int, default 10)
        cl_learn_rate -- Learn rate of the clustering algorithm (float, default 0.1)
        cl_regularization -- Regularization factor of the clustering algorithm (float, default 0.1)
        num_factors -- Number of latent features (int, default 10)
        num_iterations -- Maximum number of iterations (int, default 10)
        learn_rate -- Learn rate, aka step size (float, default 0.01)
        regularization -- Regularization factor (float, default 0.01)
        random_seed -- Random seed (int, default 1)
        eps -- maximum distance from the cluster center
        MinPts -- minimum number of points in a cluster
        K -- number of clusters"""

        self.cl_num_iterations = cl_num_iterations
        self.cl_learn_rate = cl_learn_rate
        self.cl_regularization = cl_regularization
        self.k = k
        self.eps = eps
        self.MinPts = MinPts
        self.iterator = 0

        super().__init__(data, num_factors, num_iterations, num_clusters, learn_rate, regularization, regularization, random_seed)

    def _InitModel(self):
        super()._InitModel()
        self.metamodel_users = [np.abs(np.random.normal(0.0, 0.1, self.num_nodes)) for _ in range(self.data.maxuserid + 1)]
        self.metamodel_items = [np.abs(np.random.normal(0.0, 0.1, self.num_nodes)) for _ in range(self.data.maxuserid + 1)]
        self.dataSet = []
        self.count = 0
        self.Dic = defaultdict(dict)
        #self.visited = []
        self.curCores = []
        #newCores = []
        self.Clusters = []
        self.potentialCLusters = []
        self.neighborhood = {}
        self.total_dist = {}
        #Outlier = 0
        self.num = 0
        self.Outlier = cluster('Outlier')
        self.neighborhood_dist = {}
        self.neighborhood_register = {}


    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True):
        """
        Incrementally updates the model.
        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """

        user_id, item_id = self.data.AddFeedback(user, item)

        
        


        if(self.iterator < self.k):
            
            
            if len(self.user_factors[0])  == self.data.maxuserid :
                self.metamodel_users.append(np.abs(np.random.normal(0.0, 0.1, self.num_nodes)))
                
                for node in range(self.k):
                    self.user_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))

            if len(self.item_factors[0]) == self.data.maxitemid:
                self.metamodel_items.append(np.abs(np.random.normal(0.0, 0.1, self.num_nodes)))

                #self.metamodel_items.append(np.abs(np.random.normal(0.0, 0.1, self.num_nodes)))
                for node in range(self.k):
                    #self.item_factors[node] = np.append(self.item_factors[node], np.random.normal(0.0, 0.1, self.num_factors))
                    self.item_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))

                    
            self._UpdateFactorsMeta(user_id, item_id)
            user_vector = self.metamodel_users[user_id]
      
            self.dataSet.append([user,item])
            name = 'Cluster' + str(self.count)
            C = cluster(name)
            self.count += 1
            C.addPoint([user,item])
            self.curCores.append(tuple)
            self.Clusters.append(C)
            C.core = [user,item]
            self._UpdateFactors(user_id, item_id, self.iterator)
            self.iterator = self.iterator + 1 
            self.total_dist[tuple([user,item])] = 0
        else:

            if len(self.user_factors[0])  == self.data.maxuserid :
                self.metamodel_users.append(np.abs(np.random.normal(0.0, 0.1, self.num_nodes)))
                for node in range(self.k):
                    self.user_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))   
            if len(self.item_factors[0]) == self.data.maxitemid:
                self.metamodel_items.append(np.abs(np.random.normal(0.0, 0.1, self.num_nodes)))
                for node in range(self.k):
                    self.item_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))
            self._UpdateFactorsMeta(user_id, item_id)
            ##-----------------------------------  
            self.dataSet.append([user,item])
  
            self.total_dist[tuple([user,item])] = 0

            st = time.time()

            self.incrementalAdd([user,item], self.eps, self.MinPts)

            et = time.time()
            elapsed_time = et - st
            for core in self.newCores:
                if core not in self.curCores:
                    self.curCores.append(core)
                    for clust in self.Clusters:
                        if clust.has(core):
                            clust.core = core
            user_vector = self.metamodel_users[user_id]
            temp_dic ={}
            for clust in self.Clusters:
                core = clust.core
                core_user = core[0]
                self.Dic[clust.name][core_user] = 0
                u = self.data.GetUserInternalId(core_user)
                user_vector_core = self.metamodel_users[u]
                dist = np.linalg.norm(user_vector - user_vector_core)
                self.Dic[int(clust.name.replace("Cluster", ""))][user] = dist
                temp_dic[int(clust.name.replace("Cluster", ""))] = dist
            sorted_keys = sorted(temp_dic, key=temp_dic.get)
            for node in np.argsort(sorted_keys)[:int(np.round(self.num_nodes*(1-0.368)))]:
                self._UpdateFactors(user_id, item_id, node)

                
        

        
    def _UpdateFactorsMeta(self, user_id, item_id, update_users: bool = True, update_items: bool = True, target: int = 1):
        p_u = self.metamodel_users[user_id]
        q_i = self.metamodel_items[item_id]
        for _ in range(int(self.cl_num_iterations)):
            err = target - np.inner(p_u, q_i)

            if update_users:
                delta = self.cl_learn_rate * (err * q_i - self.cl_regularization * p_u)
                p_u += delta
                p_u[p_u<0] = 0.0

            if update_items:
                delta = self.cl_learn_rate * (err * p_u - self.cl_regularization * q_i)
                q_i += delta
                q_i[q_i<0] = 0.0

        self.metamodel_users[user_id] = p_u
        self.metamodel_items[item_id] = q_i

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
        print ('\n' + C.name + '\n')
        C.printPoints()
    #_________________________________________________________________________________________________________________________


    #____________________________________________________________________________________________________________________________________________________________________________________
    def regionQuery(self, P, eps):
        user = P[0]
        item = P[1]
        result = []
        result_ = []
        u = self.data.GetUserInternalId(user)
        user_vector = self.metamodel_users[u]
        total_dist = 0
        tuple_P = tuple(P)
        if tuple_P in self.neighborhood_register.keys():
            result = self.neighborhood_register[tuple_P]
            self.neighborhood[tuple_P] = self.neighborhood[tuple_P] + tuple_P
            self.total_dist[tuple_P]= self.total_dist[tuple_P] + 1
        else: 
            for point in self.dataSet:
                temp_user = point[0]
                temp_user = self.data.GetUserInternalId(temp_user)
                temp_user_vector = self.metamodel_users[temp_user]
                dist = np.linalg.norm(user_vector - temp_user_vector)
                if(dist<=eps):
                    point_tuple = tuple(point)
                    self.neighborhood_dist[point_tuple] = self.total_dist[point_tuple]
                    result.append([point[0],point[1]])
                    result_.append(user)
                    total_dist = total_dist +1
            tuple_ = tuple(result_)
            tuple_P = tuple(P)
            self.neighborhood[tuple_P] = tuple_
            self.neighborhood_dist[tuple_P] = total_dist
            self.total_dist[tuple_P] = total_dist
            self.neighborhood_register[tuple_P] = result
        return result
    #_________________________________________________________________________________________________________________________________________________________________________________________

    #___________________________________________________________________________________________________________________________________________________
    def incrementalAdd(
        self,
        p,
        eps,
        Minpts,
        ):
        user = p[0]
        item = p[1]
        self.num = self.num + 1
        self.visited = []
        self.newCores = []
        UpdSeedIns = []
        foundClusters = []
        potentialSize = 1
        st = time.time()
        NeighbourPoints = self.regionQuery(p, eps)
        if len(NeighbourPoints) >= Minpts:
            sorted_keys = sorted(self.neighborhood_dist, key=self.neighborhood_dist.get)

            potential_core = sorted_keys[0]
            potential_core_user = potential_core[0]
            potential_core_item = potential_core[1]
            potential_core_point = [potential_core_user,potential_core_item]
            potential_core_point = np.asarray(potential_core)

            if potential_core not in self.curCores:
                self.newCores.append(potential_core)

            corehood = self.regionQuery(potential_core_point, eps)    
            potentialSize += len(self.neighborhood[potential_core]) + 1 
        et = time.time()
        elapsed_time = et - st
        st = time.time()
        for core in self.newCores:
            corehood = self.regionQuery(core, eps)
            #print ('the corehood is:', corehood)
            for elem in corehood:
                if len(self.regionQuery(elem, eps)) >= Minpts:
                    #had these new cores to the updateSeedList
                    if elem not in UpdSeedIns:
                        UpdSeedIns.append(elem)
        et = time.time()
        elapsed_time = et - st
        #print("_______________________________________________________________________________________________________________________________________________________________")
        #print('Execution time for each of the new cores of p:', elapsed_time, 'seconds')
        #print("_______________________________________________________________________________________________________________________________________________________________")
        #if there are no points in UpdateSeedList p is an outlier
        if len(UpdSeedIns) < 1:
            self.Outlier.addPoint(p)
        else:
            #there are Seeds for the update
            findCount = 0
            st = time.time()
            #If any of the seed is in an already existate cluster update findcount, if said cluster isnt in found clusters add
            for seed in UpdSeedIns:
                for clust in self.Clusters:
                    if clust.has(seed):
                        findCount += 1
                        if clust.name not in foundClusters:
                            foundClusters.append(clust.name)
                            break
            et = time.time()
            elapsed_time = et - st
            #print("_______________________________________________________________________________________________________________________________________________________________")
            #print('Execution time If any of the seed is in an already existate cluster update findcount, if said cluster isnt in found clusters add:', elapsed_time, 'seconds') 
            #print("_______________________________________________________________________________________________________________________________________________________________")
            st = time.time()
            """"
            print("point 2")
            for clust in self.Clusters:
                clust.printPoints()"""
            
            #Creating a new cluster
            # if the seeds werent found in any existing cluster we need to create a new one          
            if len(foundClusters) == 0:
                flag = 0
                smallest = self.Clusters[0]
                for clust in self.Clusters:
                    if(clust.size() < smallest.size()):
                        smallest=clust 
                if(smallest.size()< potentialSize):        
                    name = smallest.name
                    self.Clusters.remove(smallest)
                            #self.count += 1
                            #name = "Cluster" + str(self.count)
                    C = cluster(name)
                    C.core = [user,item]
                    self.expandCluster(UpdSeedIns[0],self.regionQuery(UpdSeedIns[0],eps), C, eps, Minpts)
                    flag = 1
                    #print("Deleting CLuster thats too small and expanding P")
                else:
                    self.Outlier.addPoint(p)

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
                
                """"
                print("point 6")
                for clust in self.Clusters:
                    clust.printPoints()"""
                
            else:
                closest = math.inf
                closest_core = self.Clusters[0]
                for clust in foundClusters:
                    for c in self.Clusters:
                        if c.name == clust:
                            core = c.core
                            core_user = core[0]
                            u = self.data.GetUserInternalId(core_user)
                            user_vector_core = self.metamodel_users[u]
                            user = p[0]
                            user_id = self.data.GetUserInternalId(user)
                            user_vector = self.metamodel_users[user_id]
                            dist = np.linalg.norm(user_vector - user_vector_core)
                            if dist < closest :
                                closest = dist
                                closest_core = c
                closest_core.addPoint(p)
                for seed in UpdSeedIns:
                    if not closest_core.has(seed):
                        closest_core.addPoint(seed)
                                
                            
            #print("_______________________________________________________________________________________________________________________________________________________________")
            #print('Execution time Adding a Point P to already existing cluster:', elapsed_time, 'seconds')      
            #print("_______________________________________________________________________________________________________________________________________________________________")
                    
###--------------------------------------------------------------------------------------------------------
##auxiliar funtions

    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))

    def sumvector(V,x):
        result = []
        for v in V:
            temp = v + x
            result.append(temp)
        return result    

    def multvector(x,V):
        result = []
        for v in V:
            temp = v * x
            result.append(temp)
        return result    

    def TransformVec(self,V):
        #new = normalize(V[:,np.newaxis], axis=0).ravel()
        for i in range(len(V)):
            V[i] = self.sigmoid(V[i])
        D = LA.norm(V)
        D=1/D
        new = D * V
        return new
    
