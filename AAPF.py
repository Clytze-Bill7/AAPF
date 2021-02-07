import numpy as np
import matplotlib.pyplot as plt
from forest_nfdu import NFDUTree,exNode,inNode
from lib.performance import roc_auc, f1, pr,plot_performance_auc_pr_f1, save_results
from sklearn.manifold import TSNE
import random
from skmultiflow.data import DataStream
import time
from queue import Queue
from collections import Counter

import copy


class AAPF:

    def __init__(self, data, label, ps, data_name, query_method):
        # data parameters
        self.data = data
        self.label = label
        self.data_name = data_name
        self.anomaly_ratio = ps.anomaly_ratio
        # tree parameters
        self.n_trees = ps.num_trees
        self.n_trees_batch = ps.num_trees_batch
        self.sample = ps.tree_size
        self.c_value = self.c(self.sample)
        # sliding window parameters
        self.BATCH_SIZE = ps.window_size
        self.query_ratio = ps.query_ratio
        self.query_method = query_method
        self.forest_queue = Queue(maxsize=self.n_trees)
        # self.unlabel_to_train_ratio = 1 - self.query_ratio
        self.unlabel_to_train_ratio = ps.pseudo_label_ratio
        self.height_limit = int(np.ceil(np.log2(self.sample)))

        self.un_ratio = 0.5*self.query_ratio
        self.dist_ratio = self.query_ratio - self.un_ratio
        self.prior_knowledge = ps.prior_knowledge

    def run(self, train_size, iter):
        stream = DataStream(self.data, self.label)
        self.attr_prob = self.attribute_split_probability()
        roc_all = []
        f1_all = []
        pr_all = []
        self.batch = 0
        print("Begin Trainning and Testing-----------")
        start_time = time.time()
        dist_old = []

        score_roc_numpy = []
        score_pr_numpy = []
        while stream.has_more_samples():
            self.batch += 1
            print(str(iter) + "_" + str(self.batch))
            X_test, y_test = stream.next_sample(self.BATCH_SIZE)
            query_num = int(self.query_ratio*len(X_test))

            # 模型初始化
            self.iforest_construction(X_test)

            # 这个node_count和node_idx全是基于X得到的信息
            node_count_X_forest = []
            node_idx_X_forest = []
            all_node_idx_forest = []
            for t in range(len(self.forest_queue.queue)):
                node_count_X_tree, node_idx_X_tree, all_node_idx_tree_dict = self.node_and_data(X_test, self.forest_queue.queue[t][0])
                node_count_X_forest.append(node_count_X_tree)
                node_idx_X_forest.append(node_idx_X_tree)
                all_node_idx_forest.append(all_node_idx_tree_dict)

            # 模型计算得分
            score_forest, exnode_idx_forest, data_count_in_path_forest_lst = self.tree_anomaly_score_each_data(
                X_test, node_count_X_forest, node_idx_X_forest, all_node_idx_forest)
            score_data = np.mean(score_forest, 1)

            # num_min, num_max = self.get_min_max_num_forest(node_count_X_forest, exnode_idx_tree)
            num_min = 1
            num_max = self.BATCH_SIZE
            # 模型得分roc评估
            if self.batch > train_size // self.BATCH_SIZE:
                if (np.sum(y_test) != 0) and (np.sum(y_test) != len(y_test)):
                    roc_batch = roc_auc(y_test, score_data)
                    f1_batch = f1(y_test, score_data, 1)
                    pr_batch = pr(y_test, score_data)
                    print(roc_batch, pr_batch)
                    roc_all.append(roc_batch)
                    f1_all.append(f1_batch)
                    pr_all.append(pr_batch)
                    score_roc_numpy.append(roc_batch)
                    score_pr_numpy.append(pr_batch)
                else:
                    score_roc_numpy.append('-')
                    score_pr_numpy.append('-')

            # # 查询数据选取
            dist_new = self.subspace_distribution_individual(X_test, node_count_X_forest, node_idx_X_forest)
            if self.batch == 1:
                query_idx = random.sample(range(len(X_test)), query_num)
            else:
                kl_dist = self.kl_divergence_dic(dist_old[-(self.n_trees - self.n_trees_batch):],
                                                 dist_new[:(self.n_trees - self.n_trees_batch)], X_test, node_idx_X_forest)

                index = self.query_data(self.query_method, node_idx_X_forest, y_test, dist_new, kl_dist, score_data,
                                        data_count_in_path_forest_lst, exnode_idx_forest, query_num, score_forest)
                query_idx = list(index[:query_num])
            #
            # # 模型更新
            self.query_anomaly_num = len([i for i,x in enumerate(y_test[query_idx]) if x == 1 ])
            self.query_normal_num = len(query_idx) - self.query_anomaly_num
            # print("!!!", self.query_anomaly_num, self.query_normal_num)
            self.query_anomaly_node = (exnode_idx_forest[query_idx])[y_test[query_idx] == 1]
            self.query_normal_node = (exnode_idx_forest[query_idx])[y_test[query_idx] == 0]

            # important_tree_index = get_important_trees(node_count_X_forest[query_idx], y_test)
            important_tree_index = range(self.n_trees)

            pseudo_normal_idx, pseudo_anomaly_idx = self.select_unlabel_data(self.unlabel_to_train_ratio, exnode_idx_forest, important_tree_index, query_idx)
            for i in range(len(X_test)):
                if i in query_idx:
                    self.weight_update(y_test[i], node_count_X_forest, exnode_idx_forest[i],
                                       num_min, num_max, 1, self.query_normal_num, self.query_normal_num)
                elif i in pseudo_normal_idx:
                    self.weight_update(0, node_count_X_forest, exnode_idx_forest[i], num_min, num_max, self.query_ratio, len(pseudo_normal_idx), len(pseudo_anomaly_idx))
                elif i in pseudo_anomaly_idx:
                    self.weight_update(1, node_count_X_forest, exnode_idx_forest[i], num_min, num_max, self.query_ratio, len(pseudo_normal_idx), len(pseudo_anomaly_idx))

            dist_old = dist_new
        end_time = time.time()


        # name = 'nfdu_'+self.data_name+"_"+str(self.query_method)+"_"+str(self.query_ratio)
        name = 'nfdu_'+self.data_name+"_"+str(self.query_method)+"_rp_"+str(self.prior_knowledge)+"_ru_"+str(self.unlabel_to_train_ratio)
        save_results(name, iter, np.mean(roc_all),np.mean(pr_all), np.mean(f1_all), end_time - start_time,
                     self.data_name, self.anomaly_ratio, self.query_ratio)

        print(score_roc_numpy)
        print(score_pr_numpy)
        print(name, self.data_name + " is done")


        # plot_performance_auc_pr_f1(range(len(roc_all)),
        #                            np.array(roc_all), "AUC_ROC", np.array(pr_all), "AUC_PR", np.array(f1_all), "f1",
        #                            "NFDU_"+self.query_method+"_"+str(self.query_ratio)+"_",  self.data_name, k=iter)

        return np.mean(roc_all),np.mean(pr_all), name
    # def get_important_trees(self,original_num_array, query_label):
    #     important_tree_nums = int(self.important_trees_ratio*self.n_trees)
    #     normal = np.zeros(self.n_trees)
    #     anomaly = np.zeros(self.n_trees)
    #     normal_num = 0
    #     anomaly_num = 0
    #
    #     for i in range(len(original_num_array)):
    #         new_num_array = np.zeros(len(original_num_array[i]))
    #         for t in range(len(original_num_array[i])):
    #             new_num_array[t] = original_num_array[i][t]
    #
    #         if query_label[i] == 0:
    #             normal+=(new_num_array / np.sum(new_num_array))
    #             normal_num+=1
    #         else:
    #             anomaly+=(new_num_array / np.sum(new_num_array))
    #             anomaly_num+=1
    #
    #
    #     if normal_num==0:
    #         score =  - anomaly / anomaly_num
    #     elif anomaly_num == 0:
    #         score = normal / normal_num
    #     else:
    #         score = normal / normal_num - anomaly / anomaly_num
    #
    #     return np.argsort(-np.abs(score))[:important_tree_nums]




    def select_unlabel_data(self, unlabel_to_train_ratio, exnode_idx_forest, important_tree_index, query_idx):

        if self.query_anomaly_num !=0:
            normal_ratio = self.query_normal_num / (self.query_normal_num + self.query_anomaly_num+1)
            anomaly_ratio = self.query_anomaly_num / (self.query_normal_num + self.query_anomaly_num+1)
        else:
            normal_ratio = (self.query_normal_num-1) / (self.query_normal_num + self.query_anomaly_num+1)
            anomaly_ratio = 1 / (self.query_normal_num + self.query_anomaly_num+1)


        # normal_ratio = unlabel_to_train_ratio * 0.9
        # anomaly_ratio = unlabel_to_train_ratio - normal_ratio

        data_num = len(exnode_idx_forest)
        unlable_socre = self.get_unlabel_pseudo_label_eachdata(exnode_idx_forest, important_tree_index)
        score_normal = np.argsort(unlable_socre)
        score_anomaly = np.argsort(-unlable_socre)
        pseudo_normal_idx = []
        for idx in list(score_normal):
           if len(pseudo_normal_idx) == int(data_num * unlabel_to_train_ratio*normal_ratio):
               break
           else:
               if idx not in query_idx:
                   pseudo_normal_idx.append(idx)

        pseudo_anomaly_idx = []
        for idx in list(score_anomaly):
           if len(pseudo_anomaly_idx) == int(data_num * unlabel_to_train_ratio*anomaly_ratio):
               break
           else:
               if idx not in query_idx:
                   pseudo_anomaly_idx.append(idx)
        return pseudo_normal_idx, pseudo_anomaly_idx

    def tree_anomaly_score_each_data(self, X, node_count_X_forest, node_idx_X_forest, all_node_idx_forest):

        score = np.zeros((len(X), self.n_trees))
        exnode_idx = np.zeros((len(X), self.n_trees))
        data_count_in_path_forest_lst = []
        for t in range(len(self.forest_queue.queue)):
            data_count_in_path_tree_lst = []
            for i in range(len(X)):
                num_delta, exnodeindex, all_node_lst = self.onePath(X[i], self.forest_queue.queue[t][0],0,[])
                # if num_delta == float('inf') or np.isnan(num_delta):
                #     print(num_delta)
                #     exit()
                data_count_in_path_lst = []
                for node_idx in all_node_lst:
                    data_count_in_path_lst.append(all_node_idx_forest[t][node_idx])
                data_count_in_path_tree_lst.append(data_count_in_path_lst)
                score[i,t] = -(node_count_X_forest[t][exnodeindex] + num_delta)
                exnode_idx[i,t] = exnodeindex
            data_count_in_path_forest_lst.append(data_count_in_path_tree_lst)
        return score, exnode_idx, data_count_in_path_forest_lst





    def onePath(self, obs, node, path_length, all_node_lst):
        if type(node) == exNode:
            all_node_lst.append(node.nodeidx)
            return node.num_delta, node.nodeidx, all_node_lst
        else:
            a = node.splitAtt
            path_length += 1
            all_node_lst.append(node.nodeidx)
            if obs[a] < node.splitValue:
                return self.onePath(obs, node.left, path_length, all_node_lst)
            else:
                return self.onePath(obs, node.right, path_length, all_node_lst)





    def get_unlabel_pseudo_label_eachdata(self, idx_instance_forest_batch_X, important_tree_index):
        data_num = len(idx_instance_forest_batch_X)
        score = np.zeros(data_num)
        for i in range(data_num):
            idx_instance_forest_batch = idx_instance_forest_batch_X[i]
            num_anomaly_array = np.zeros(len(important_tree_index))
            num_normal_array = np.zeros(len(important_tree_index))
            for t in range(len(important_tree_index)):
                if idx_instance_forest_batch[important_tree_index[t]] in self.query_anomaly_node[:, important_tree_index[t]]:
                    num_anomaly_array[t] = list(self.query_anomaly_node[:, important_tree_index[t]]).count(
                        idx_instance_forest_batch[important_tree_index[t]])
                if idx_instance_forest_batch[important_tree_index[t]] in self.query_normal_node[:, important_tree_index[t]]:
                    num_normal_array[t] = list(self.query_normal_node[:, important_tree_index[t]]).count(
                        idx_instance_forest_batch[important_tree_index[t]])
            # num_score_array[i] = np.mean(num_anomaly_array - num_normal_array)
            anomaly = np.mean(num_anomaly_array)
            normal = np.mean(num_normal_array)
            if self.query_normal_num == 0:
                num_score_array = - anomaly / self.query_anomaly_num
            elif self.query_anomaly_num == 0:
                num_score_array = normal / self.query_normal_num
            else:
                num_score_array = (normal / self.query_normal_num) - (anomaly / self.query_anomaly_num)

            score[i] = (-num_score_array + 1) / 2
        return score


    def iforest_construction(self, X):
        if self.batch == 1:
            build_tree_num = self.n_trees
        else:
            build_tree_num = self.n_trees_batch

        for t in range(build_tree_num):
            # 1.树的根节点 和 2.树
            tree_root, tree = NFDUTree(self.height_limit).fit(X, self.attr_prob, self.data, improved="random")
            # 3.树的所有节点字典，字典中保留的是树所有节点的实体
            # 4.树的所有外节点字典，字典中保留的是树外节点的实体
            if self.batch == 1:
                self.forest_queue.put((tree_root, tree, tree.idx_node_dict, tree.idx_exnode_dict))
            else:
                self.forest_queue.get()
                self.forest_queue.put((tree_root, tree, tree.idx_node_dict, tree.idx_exnode_dict))


    def node_and_data(self, X, tree_root):
        node_lst = []
        data_idx_in_node = {}
        all_node_idx_tree_lst = []
        for i in range(len(X)):
            nodeidx_data, allnode_idx_each_data_lst = self.exnode_find(X[i], tree_root, [])
            all_node_idx_tree_lst.extend(allnode_idx_each_data_lst)
            node_lst.append(nodeidx_data)
            if nodeidx_data in data_idx_in_node:
                data_idx_in_node[nodeidx_data].append(i)
            else:
                data_idx_in_node[nodeidx_data] = [i]

        node_count_X = Counter(node_lst)
        all_node_idx_tree_dict = Counter(all_node_idx_tree_lst)
        return node_count_X, data_idx_in_node, all_node_idx_tree_dict

    def exnode_find(self, obs, node, allnode_idx_lst):
        if type(node) == exNode:
            # 除了返回路径，还要返回当前数据落到哪一个exnode里了
            allnode_idx_lst.append(node.nodeidx)
            return node.nodeidx,allnode_idx_lst
        else:
            a = node.splitAtt
            allnode_idx_lst.append(node.nodeidx)

            if obs[a] < node.splitValue:
                return self.exnode_find(obs, node.left, allnode_idx_lst)
            else:
                return self.exnode_find(obs, node.right, allnode_idx_lst)





    def minmax(self,score):
        score_mm = (score - np.min(score)) / (np.max(score) - np.min(score))
        return score_mm

    def standard(self,score):
        score_standard = (score - np.mean(score)) / np.std(score)
        return score_standard


    def query_data(self, choose_method, node_idx_X_forest, y, dist_new, kl_dist, score,
                   data_count_in_path_forest_lst, exnode_idx_forest, query_num, score_forest):

        if choose_method == "random":
            sort_idx = list(range(len(y)))
            random.shuffle(sort_idx)
        elif choose_method == "uncertainty":
            sort_idx = self.select_uncertainty(y, data_count_in_path_forest_lst, exnode_idx_forest)

        elif choose_method == "drift":
            sort_idx = self.select_drift( kl_dist, node_idx_X_forest, y)
        elif choose_method == "dynamic_uncertainty":
            sort_idx = self.select_dynamic_uncertainty(kl_dist, node_idx_X_forest, y, data_count_in_path_forest_lst, exnode_idx_forest)
        elif choose_method == "anomaly":
            sort_idx = np.argsort(-score)
        elif choose_method == "percentile":
            sort_idx = np.argsort(np.abs(score - np.percentile(score, 100 - self.anomaly_ratio)))
        elif choose_method == "noupdate":
            sort_idx = []
        elif choose_method == "anomaly_label":
            sort_idx = []
            for i in range(len(y)):
                if y[i] == 1:
                    sort_idx.append(i)

        elif choose_method == "inconsistency":
            std_score = np.std(score_forest,1)
            sort_idx = np.argsort(-std_score)

        else:
            print("Query Method Error")
            exit()

        return sort_idx




    def select_dynamic_uncertainty(self,kl_dist, node_idx_X_forest, y, data_count_in_path_forest_lst, exnode_idx_forest):
        dist_score = np.zeros(len(y))
        uncertainty_score = np.zeros(len(y))
        for i in range(len(y)):
            for t in range(len(kl_dist)):
                for node in node_idx_X_forest[t]:
                    if i in node_idx_X_forest[t][node]:
                        dist_score[i] += np.abs(kl_dist[t][node])
                        break

        for i in range(len(y)):
            tmp = 0
            for t in range(len(data_count_in_path_forest_lst)):
                exnode_idx = exnode_idx_forest[i][t]
                exnode = self.forest_queue.queue[t][2][exnode_idx]
                # tmp += np.mean(data_count_in_path_forest_lst[t][i])/(data_count_in_path_forest_lst[t][i][-1] + exnode.num_delta+1000)
                tmp += (np.mean(data_count_in_path_forest_lst[t][i])/(data_count_in_path_forest_lst[t][i][-1])) / self.n_trees
            uncertainty_score[i] = tmp

        # print("dynamic=", y[np.argsort(-dist_score)[:100]])
        # print("uncertainty=", y[np.argsort(-uncertainty_score)[:100]])

        # if self.batch %10 == 1 and  self.batch!=1:
        #     score = dist_score
        # else:
        #     score = uncertainty_score

        score = dist_score + uncertainty_score
        print(np.mean(dist_score), np.mean(uncertainty_score))
        score_minmax = (score - np.min(score)) / (np.max(score) - np.min(score))
        score_minmax_average = score_minmax / np.sum(score_minmax)
        index = np.random.choice(range(len(score)), int(self.query_ratio*len(score)), replace= False, p=score_minmax_average.ravel())

        # index = np.argsort(-score)

        # distscore_minmax = (dist_score - np.min(dist_score)) / (np.max(dist_score) - np.min(dist_score))
        # distscore_minmax_average = distscore_minmax / np.sum(distscore_minmax)
        # dist_index = np.random.choice(range(len(distscore_minmax_average)), int(self.dist_ratio*len(distscore_minmax_average)),
        #                          replace= False, p=distscore_minmax_average.ravel())
        #
        # uncertaintyscore_minmax = (uncertainty_score - np.min(uncertainty_score)) / (np.max(uncertainty_score) - np.min(uncertainty_score))
        # unscore_minmax_average = uncertaintyscore_minmax / np.sum(uncertaintyscore_minmax)
        # un_index = np.random.choice(range(len(unscore_minmax_average)), int(2*self.un_ratio*len(unscore_minmax_average)),
        #                          replace= False, p=unscore_minmax_average.ravel())
        #
        # rest_idx = []
        # for id in un_index:
        #     if len(rest_idx) < int(self.un_ratio*len(unscore_minmax_average)):
        #         if id not in dist_index:
        #             rest_idx.append(id)
        #     else:
        #         break
        # index = list(dist_index)+rest_idx

        # print(len(index))
        # index = np.argsort(-score)
        #
        # print(np.sort(score))


        return index


    def select_uncertainty(self, y, data_count_in_path_forest_lst, exnode_idx_forest):
        uncertainty_score = np.zeros(len(y))
        for i in range(len(y)):
            tmp = 0
            for t in range(len(data_count_in_path_forest_lst)):
                exnode_idx = exnode_idx_forest[i][t]
                exnode = self.forest_queue.queue[t][2][exnode_idx]
                # tmp += np.mean(data_count_in_path_forest_lst[t][i])/(data_count_in_path_forest_lst[t][i][-1] + exnode.num_delta+1000)
                tmp += (np.mean(data_count_in_path_forest_lst[t][i])/(data_count_in_path_forest_lst[t][i][-1])) / self.n_trees
            uncertainty_score[i] = tmp


        score = uncertainty_score
        score_minmax = (score - np.min(score)) / (np.max(score) - np.min(score))
        score_minmax_average = score_minmax / np.sum(score_minmax)
        index = np.random.choice(range(len(score)), int(self.query_ratio*len(score)), replace= False, p=score_minmax_average.ravel())
        # index = np.argsort(-score)

        return index

    def select_drift(self, kl_dist, node_idx_X_forest, y):
        dist_diff = np.zeros(len(y))
        for i in range(len(y)):
            for t in range(len(kl_dist)):
                for node in node_idx_X_forest[t]:
                    if i in node_idx_X_forest[t][node]:
                        dist_diff[i] += kl_dist[t][node]
                        break
        # self.dynamic_score = np.mean(dist_diff)
        # print("score_threshold=", np.mean(self.dynamic_score))
        dist_diff_minmax = (dist_diff - np.min(dist_diff)) / (np.max(dist_diff) - np.min(dist_diff))
        dist_diff_minmax_average = dist_diff_minmax / np.sum(dist_diff_minmax)
        index = np.random.choice(range(len(dist_diff)), int(self.query_ratio*len(dist_diff)), replace= False, p=dist_diff_minmax_average.ravel())

        # index = np.argsort(-dist_diff)
        # self.dynamic_score = np.mean(score)
        return index

    def kl_divergence_dic(self, p_dist, q_dist, X, q_idx):
        kl_dist_forest = []
        for t in range(len(p_dist)):
            kl_dist_new_tree = {}
            for node_idx in q_idx[t]:
                kl_dist_new_tree[node_idx] = (q_dist[t][node_idx] - p_dist[t][node_idx]) / p_dist[t][node_idx]
                # kl_dist_new_tree[node_idx] = (q_dist[t][node_idx] - p_dist[t][node_idx])
            kl_dist_forest.append(kl_dist_new_tree)
        return kl_dist_forest

    def subspace_distribution_individual(self, X_test, node_count_X_forest, node_idx_X_forest):
        dic_dist_forest = []
        for t in range(self.n_trees):
            dict_dist = {}
            for node in self.forest_queue.queue[t][3]:
                if node in node_count_X_forest[t]:
                    dict_dist[node] = (node_count_X_forest[t][node] + 0.5) / (len(self.forest_queue.queue[t][3])/2 + X_test.shape[0])
                else:
                    # batch在模型上未出现的叶节点处的处理
                    dict_dist[node] = 0.5 / (len(self.forest_queue.queue[t][3])/2 + X_test.shape[0])
            dic_dist_forest.append(dict_dist)
        return dic_dist_forest


    def c(self, num):
        if num > 2:
            return 2 * (np.log(num - 1) + 0.5772156649) - 2 * (num - 1) / num
        elif num == 2:
            return 1
        else:
            return 0

    #
    def weight_update(self, y, node_count_X_forest, exnodeidx_forest, path_min, path_max, gamma, normal_num, anomaly_num):

        # if y==1:
        #     ratio = gamma
        # elif y==0:
        #     if anomaly_num == 0:
        #         ratio = 1 / normal_num * gamma
        #     else:
        #         ratio = anomaly_num / normal_num *gamma
        ratio = gamma


        for t in range(self.n_trees_batch, self.n_trees):
            exnode = self.forest_queue.queue[t][3][exnodeidx_forest[t]]
            num_new =  node_count_X_forest[t][exnodeidx_forest[t]] + exnode.num_delta

            diff = y - (path_max - num_new) / (path_max - path_min)
            # diff =  y - (1 / (1 +  num_new))


            normalanomaly = (path_max ** (1 - y)) * (path_min ** (y))
            path_var = ratio * (normalanomaly - num_new) * np.abs(diff)
            exnode.num_delta = exnode.num_delta + path_var
            # print(num_new, diff,normalanomaly,path_var, exnode.num_delta)
            # print(y, num_new, num_new+path_var, "({},{})".format(path_min[t- self.n_trees_batch], path_max[t- self.n_trees_batch]))
            # print(y, exnodeidx_forest[t], path_var,  num_new, num_new+path_var, "({},{})".format(path_min, path_max))


    def attribute_split_probability(self):
        probability = np.zeros(self.data.shape[1])
        if self.data_name in ["ids2017_2","ids2017_3","ids2017_4","ids2017_5","ids2017_6","ids2017_7","ids2017_8"]:
            # important_idx = [0, 3, 4, 11, 12, 16, 18, 33, 34, 49, 54, 56, 57, 58, 62, 64]
            important_idx = [10, 62, 3, 12, 18, 23, 0, 72, 69, 25, 65, 29, 43, 66, 7, 36, 45, 51]
            # important_idx = []
            ratio =self.prior_knowledge
        elif self.data_name =="unsw_nb15":
            # important_idx = [7, 28, 6, 9, 8, 37, 27, 38, 11, 4, 1, 30, 10]
            important_idx = [7, 21, 16, 19, 28, 36, 37, 6, 31, 15]
            ratio = self.prior_knowledge
        else:
            important_idx = []
            ratio = 1

        for j in range(self.data.shape[1]):
            if j in important_idx:
                probability[j] = ratio
            else:
                probability[j] = 1


        probability /= np.sum(probability)


        return list(probability)








