import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture as GMM
import copy
import random

class Forest_nfdu:
    def __init__(self, sample_size,  n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees



    def fit(self, X:np.ndarray, attr_prob, all_data, improved="median"):
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        numX, numQ = X.shape
        self.trees = []
        self.tree_object = []

        height_limit = int(np.ceil(np.log2(self.sample_size)))
        self.height_limit = height_limit
        for k in range(self.n_trees):
            tree_root, tree = Tree(height_limit).fit(X, attr_prob, all_data, improved=improved)
            self.trees.append(tree_root)
            self.tree_object.append(tree)

        return self



    def onePathLength(self, obs, tree, original_path_length, weighted_path_length,idx_lst):
        if type(tree) == exNode:
            weighted_path_length += 1
            idx_lst.append(tree.nodeidx)
            return original_path_length  + tree.num_delta, weighted_path_length, idx_lst, tree.nodeidx
        else:
            a = tree.splitAtt
            idx_lst.append(tree.nodeidx)
            original_path_length += 1
            weighted_path_length += 1
            if obs[a] < tree.splitValue:
                return self.onePathLength(obs, tree.left, original_path_length, weighted_path_length,idx_lst)
            else:
                return self.onePathLength(obs, tree.right, original_path_length, weighted_path_length,idx_lst)





    def find_weighted_path(self, node):
        weighted_path_length = 0
        deep = 0
        # node = node.father_node
        while node != None:
            weighted_path_length += node.w
            node = node.father_node
            deep+=1
        return weighted_path_length, deep



    # ----------------------------------------------
    def exnode_find(self, X: np.ndarray) -> np.ndarray:
        numX, numQ = X.shape
        exnodeidx_list = []
        for i in range(numX):
            obs = X[i, :]
            exnode_instance_forest = []
            for k in self.trees:
                idx = self.exNode_tree(obs, k)
                exnode_instance_forest.append(idx)
            exnodeidx_list.append(exnode_instance_forest)

        return np.array(exnodeidx_list)

    def exNode_tree(self, obs, tree):
        if type(tree) == exNode:
            return tree.nodeidx
        else:
            a = tree.splitAtt
            if obs[a] < tree.splitValue:
                return self.exNode_tree(obs, tree.left)
            else:
                return self.exNode_tree(obs, tree.right)

class inNode:
    def __init__(self,lower_bound, upper_bound,idx,space_volume):
        self.nodeidx = idx
        self.left = None
        self.right = None
        self.splitAtt = 0
        self.splitValue = 0
        self.space_volume = space_volume
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
class exNode:
    def __init__(self, lower_bound, upper_bound,idx,space_volume, node_deep):
        self.nodeidx = idx
        self.space_volume = space_volume
        self.num_delta = 0
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.node_deep = node_deep

    def c(self, num):
        if num > 2:
            return 2 * (np.log(num - 1) + 0.5772156649) - 2 * (num - 1) / num
        elif num == 2:
            return 1
        else:
            return 0

class NFDUTree:

    def __init__(self, height_limit):
        self.height_limit = height_limit

    def fit(self, X:np.ndarray, attr_prob, all_data, improved="random"):

        self.attr_prob = attr_prob
        _, numQ = X.shape
        current_height = 0
        first_time = True
        self.idx = 0

        lower_bound = np.zeros(X.shape[1])
        upper_bound = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            upper_bound[j] = np.max(X[:, j])
            lower_bound[j] = np.min(X[:, j])


        cha = upper_bound - lower_bound
        space_volume_next_node = np.prod(cha[np.nonzero(cha)])*(10**len(np.nonzero(cha)[0]))
        root = inNode(lower_bound, upper_bound, self.idx, space_volume_next_node)
        self.BuildTreeStructure(root, X, X.shape[1], lower_bound, upper_bound, 0, self.height_limit)


        self.improved = improved

        self.root = root
        self.idx_node_dict = {}
        self.find_node_character(root, self.idx_node_dict)


        self.idx_exnode_dict = {}
        self.find_exnode_character(root, self.idx_exnode_dict)
       
        return self.root, self

    def BuildTreeStructure(self, root, X, dim, lower_bound, upper_bound, current_tree_depth, height_limit):


        r = random.random()
        splitAtt = np.random.choice(range(X.shape[1]), 1, replace=False, p=np.array(self.attr_prob).ravel())
        splitValue = np.mean(X[:,splitAtt])
        root.splitAtt = splitAtt
        root.splitValue = splitValue



        t = upper_bound[splitAtt]
        upper_bound[splitAtt] = splitValue
        cha = upper_bound - lower_bound
        left_idx = set([i for i in range(len(X)) if X[i, splitAtt] <= splitValue])
        X_left = X[list(left_idx)]

        space_volume_next_node = np.prod(cha[np.nonzero(cha)])*(10**len(np.nonzero(cha)[0]))

        if current_tree_depth +1<height_limit:
            self.idx += 1
            left = inNode(lower_bound, upper_bound, self.idx, space_volume_next_node)
            root.left = left
            self.BuildTreeStructure(left, X_left, dim, lower_bound, upper_bound, current_tree_depth+1, height_limit)
        else:
            self.idx += 1
            left = exNode(lower_bound, upper_bound,self.idx,space_volume_next_node, current_tree_depth)
            root.left = left


        upper_bound[splitAtt] = t
        lower_bound[splitAtt] = splitValue

        cha = upper_bound - lower_bound
        space_volume_next_node = np.prod(cha[np.nonzero(cha)])*(10**len(np.nonzero(cha)[0]))

        right_idx = set([i for i in range(len(X))]) - left_idx
        X_right = X[list(right_idx)]
        if current_tree_depth + 1 < height_limit:
            self.idx += 1
            right = inNode(lower_bound, upper_bound,self.idx,space_volume_next_node)
            root.right = right
            self.BuildTreeStructure(right, X_right,dim,  lower_bound, upper_bound, current_tree_depth + 1, height_limit)
        else:
            self.idx += 1
            right = exNode(lower_bound, upper_bound,self.idx,space_volume_next_node, current_tree_depth)
            root.right = right


   

    def find_exnode_character(self,root, dict):
        if type(root) == inNode:
            
            self.find_exnode_character(root.left, dict)
            self.find_exnode_character(root.right, dict)

        elif type(root) == exNode:
            dict[root.nodeidx] = root

        else:
            print("Wrong")
            exit()


    def find_node_character(self,root, dict):
        if type(root) == inNode:
            dict[root.nodeidx] = root
            self.find_node_character(root.left, dict)
            self.find_node_character(root.right, dict)

        elif type(root) == exNode:
            dict[root.nodeidx] = root

        else:

            print("Wrong")
            exit()


    def countNode(self, root):
        if type(root) == inNode:
            print(self.countNode(root.left), self.countNode(root.right))
            return 1 + self.countNode(root.left) + self.countNode(root.right)
        if type(root) == exNode:
            return 1



    def countExNode(self,root, exnode_dict):
        if type(root) == inNode:
            return self.countExNode(root.left, exnode_dict) + self.countExNode(root.right,exnode_dict)
        if type(root) == exNode:
            exnode_dict[root.nodeidx] = root
            return 1


