
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pdb as pdb
from .supervised_learner import SupervisedLearner

class Clustering(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """
    k = 7
    labels = []
    curr_centr_members = []
    orig_feat = []
    orig_label = []
    cont_or_nom = []
    # creative portion toggle
    diff_dist = False

    def __init__(self):
        pass

    def train(self, features, labels):
        #features.shuffle(labels)
        self.orig_feat = features
        self.orig_label = labels

        # init centroids
        centroids = []
        feat = [features.data]
        label = [labels.data]
        for i in range(len(feat[0])):
            feat[0][i].append(label[0][i][0])

        for i in range(self.k):
            centroids.append(feat[0][i])

        for i in range(len(feat[0][0])):
            if i  == len(feat[0][0]) - 1:
                self.cont_or_nom.append(labels.value_count(0))
            else:
                self.cont_or_nom.append(features.value_count(i))
        
        prev_cent = []
        iter = 1
        while True:
            print("** ** ** ** ** ** ** *")
            print("Iteration ", iter)
            print("** ** ** ** ** ** ** *")
            self.print_centroids(centroids)
            centroids = self.find_centroids(feat, label, centroids)
            self.print_size_centroids()
            self.print_centr_members()
            sse = self.find_sse(centroids, feat)
            self.silhouette(feat)
            if centroids == prev_cent:
                break
            else:
                prev_cent = centroids
            iter = iter + 1
        print("")
        print("FINAL CENTROIDS: ", centroids)
        print("finished!")

    def silhouette(self, feat):
        silhouette = 0.0
        for i in range(len(feat[0])):
            sill = 0.0
            a = 0.0
            b = float('inf')
            # for every centroid
            for j in range(len(self.curr_centr_members)):
                dist = 0
                # for every centroid[j] member/index into features
                for k in range(len(self.curr_centr_members[j])):
                    dist = dist + self.eu_dist(feat[0][self.curr_centr_members[j][k]], feat[0][i])
                dist = dist/len(self.curr_centr_members[j])
                if i in self.curr_centr_members:
                    a = dist
                else:
                    if b > dist:
                        b = dist
            sil = b - a / max(a, b)
            silhouette += sil
        silhouette = silhouette / len(feat[0])
        print("SILHOUETTE: ", silhouette)

    def find_centroids(self, features, labels, centroids):
        new_centroid = []
        centroid_members = []

        # make returning lists the correct k size
        for i in range(self.k):
            centroid_members.append([])
            new_centroid.append([])
        for i in range(len(features[0])):
            # lowest distance to centroid
            low_dist = float('inf')
            # index to current lowest distance to centroid
            index = float('inf')
            distances = []
            for j in range(len(centroids)):
                dist = 0.0
                dist = self.eu_dist(features[0][i], centroids[j])
                distances.append(dist)
                # if distance is lower, keep the index and value
                if dist < low_dist or low_dist == float('inf'):
                    low_dist = dist
                    index = j
            # keep track of centroid members
            centroid_members[index].append(i)

        # calculate centroids
        self.calc_centroids(centroid_members, new_centroid, features)

        self.curr_centr_members = centroid_members
        return new_centroid

    def calc_centroids(self, centroid_members, new_centroid, features):
        # iterating through centroids
        for i in range(len(centroid_members)):
            # each row in features
            for j in range(len(features[0][0])):
                # iterating through feature's index related to centroid members
                arr = []
                for k in range(len(centroid_members[i])):
                    item = features[0][centroid_members[i][k]][j]
                    # filters out cont unassigned data
                    if not item == float('inf'):
                        if not self.cont_or_nom[j] == 0:
                            arr.append(item)
                        elif self.cont_or_nom[j] == 0:
                            arr.append(item)
                if len(arr) == 0:
                    new_centroid[i].append(float("inf"))
                # if nominal take most common att
                elif not self.cont_or_nom[j] == 0:
                    att_val = max(set(arr), key=arr.count)
                    new_centroid[i].append(att_val)
                # if continuous average data
                else:
                    att_val = np.sum(arr) / len(arr)
                    if(att_val == float('inf')):
                        print("INF INF INF INF INF INF")
                    new_centroid[i].append(att_val)
    
    def eu_dist(self, inst, centr):
        distance = 0.0

        if self.diff_dist:
            return self.manhattan_dist(inst, centr)
        else:
            for i in range(len(inst)):
                # unknown doesn't match
                if inst[i] == float('inf') or centr[i] == float('inf'):
                        distance += 1
                # nominal
                elif self.cont_or_nom[i] > 0:
                    # only increases if they are not equal
                    if not inst[i] == centr[i]:
                        distance += 1
                # continuous
                else:
                    distance += (inst[i] - centr[i])**2
            return np.sqrt(distance)

    def manhattan_dist(self, inst, centr):
        distance = 0
        for i in range(len(inst)):
            # unknown doesn't match
            if inst[i] == float('inf') or centr[i] == float('inf'):
                distance += 1
            # nominal
            elif self.cont_or_nom[i] > 0:
                # only increases if they are not equal
                if not inst[i] == centr[i]:
                    distance += 1
            # continuous
            else:
                distance += abs(inst[i] - centr[i])
        return np.sqrt(distance)

    def find_sse(self, centr, feat):
        print("SSE for centroid ")
        sse = 0.0

        # index into feature indexes
        for i in range(len(self.curr_centr_members)):
            centr_sse = 0.0
            for j in range(len(self.curr_centr_members[i])):
                inst = feat[0][self.curr_centr_members[i][j]]
                centr_sse += self.eu_dist(inst, centr[i])**2
            print("sse {}: {}".format(i, centr_sse))
            sse += centr_sse

        print("TOTAL SSE: ", sse)
        print("")
        return sse

    def predict(self, features, labels):
        return [[]]


    def print_centroids(self, centroid):
        print("CENTROIDS")
        for i in range(len(centroid)):
            print("centroid {}: {}".format(i, centroid[i]))
        print("")

    def print_size_centroids(self):
        print("SIZE OF CENTROIDS")
        for i in range(len(self.curr_centr_members)):
            print("{}: {}".format(i, len(self.curr_centr_members[i])))
        print("")

    def print_centr_members(self):
        print("CLUSTER MEMBERS")
        for i in range(len(self.curr_centr_members)):
            print("centroid {}: {}".format(i, self.curr_centr_members[i]))
        print("")
