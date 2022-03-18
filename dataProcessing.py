import pickle
from os import listdir
import copy
import math
import numpy as np
import lgsvl
import csv
import os
import operator
from functools import reduce


def write_csv_file(path, head, data):
    if not os.path.exists('clusterData'):
        os.mkdir('clusterData')
    paths = 'clusterData' + '/' + str(path) + '.csv'
    try:
        with open(paths, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, dialect='excel')
            if head is not None:
                writer.writerow(head)
            writer.writerow(data)
            # for row in data:writer.writerow(row)
            print("Write a CSV file to path %s Successful." % path)
    except Exception as e:
        print("Write an CSV file to path: %s, Case: %s" % (path, e))


def is_ahead(target_transform, current_transform):
    target_vector = np.array([target_transform.position.x - current_transform.position.x,
                              target_transform.position.z - current_transform.position.z])
    norm_target = np.linalg.norm(target_vector)
    if norm_target < 0.001:
        return True
    fwd = lgsvl.utils.transform_to_forward(current_transform)
    forward_vector = np.array([fwd.x, fwd.z])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    if d_angle == 0:
        # OneLaneBefore
        position = 0
    elif d_angle < 37.0:
        # before
        position = 1
    elif 37.0 <= d_angle <= 143:
        # parall
        position = 2
    elif d_angle == 180:
        position = 3
    else:
        # after
        position = 4
    return d_angle, position


def getNpcAngleAndPositionAndDistance(npcLocation, egoLocation):
    # get angle
    # get position
    angle, position = is_ahead(npcLocation, egoLocation)
    # get distance
    distance = math.sqrt(
        (npcLocation.position.x - egoLocation.position.x) ** 2 + (npcLocation.position.z - egoLocation.position.z) ** 2)

    # angle = 1
    # position = 2
    # distance = 3
    return angle, position, distance


def getNpcSpeed(npcSpeed):
    velocity = math.sqrt(npcSpeed.x ** 2 + npcSpeed.z ** 2)
    return velocity


def getAllCheckpoints(ck_path, sizeOfNpc, times):
    onlyfiles = [f for f in listdir(ck_path)]

    for i in range(len(onlyfiles)):
        with open(ck_path + '/' + onlyfiles[i], "rb") as f:
            if "generation" not in onlyfiles[i]:
                continue
            try:
                prevPop = pickle.load(f)
                for gene in prevPop:
                    npcLocation = copy.deepcopy(gene.npcLocation)
                    egoLocation = copy.deepcopy(gene.egoLocation)
                    npcSpeed = copy.deepcopy(gene.npcSpeed)
                    lengOfTime = len(gene.egoLocation)
                    count = 0
                    gens = [[] for i in range(times)]
                    for j in range(0, lengOfTime, 12):
                        npcs = []
                        for k in range(sizeOfNpc):
                            angle, position, distance = getNpcAngleAndPositionAndDistance(npcLocation[k][j],
                                                                                          egoLocation[j])
                            speed = getNpcSpeed(npcSpeed[k][j])
                            npcs.append(copy.deepcopy([angle, position, distance, speed]))
                        if count <= 3:
                            gens[0] += copy.deepcopy(npcs)
                        elif 3 < count <= 7:
                            gens[1] += copy.deepcopy(npcs)
                        elif 7 < count <= 11:
                            gens[2] += copy.deepcopy(npcs)
                        elif 11 < count <= 15:
                            gens[3] += copy.deepcopy(npcs)
                        count += 1

                    for w in range(len(gens)):
                        tmp = reduce(operator.add, gens[w])
                        for u in range(len(gene.scenario)):
                            tmp.append(gene.scenario[u][w])
                        write_csv_file('Gen' + str(w), None, tmp)

            except Exception:
                pass


# #############################################################################
# Compute DBSCAN
if __name__ == '__main__':
    getAllCheckpoints('/datas/GaCheckpointsCrossroads', 4, 4)
