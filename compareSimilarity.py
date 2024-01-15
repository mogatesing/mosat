import pickle
from os import listdir
import os
import time
import shutil
import pickle
import math


def comparefile(path, path2):
    """
    classify scenario
    """
    files = [f for f in listdir(path)]
    types = []

    if os.path.exists('classification') == False:
        os.mkdir('classification')

    for filedir in files:
        scenarioTypes = []
        scenariofile = [f for f in listdir(path + '/' + filedir)]
        for scenario in scenariofile:
            mtime = os.stat(path + '/' + filedir + '/' + scenario).st_mtime
            file_modify_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(mtime))
            scenarioTypes.append(file_modify_time)
            AccidentScenarios = [f for f in listdir(path2)]
            for AccidentScenario in AccidentScenarios:
                mtime2 = os.stat(path2 + '/' + AccidentScenario).st_mtime
                file_modify_time2 = time.strftime('%Y-%m-%d-%H-%M', time.localtime(mtime2))
                if file_modify_time2 == file_modify_time:
                    # classification
                    if os.path.exists('classification/' + filedir) == False:
                        os.mkdir('classification/' + filedir)
                    shutil.move(path2 + '/' + AccidentScenario, 'classification/' + filedir)
                    break
        types.append({filedir: scenarioTypes})


def compareNpc(npc1, npc2):
    """
    calculate similarity between two scenarios
    """
    le = min(len(npc1), len(npc2))
    x = 0
    for i in range(le):
        x += math.sqrt((npc1[i].position.x - npc2[i].position.x) ** 2 + (npc1[i].position.z - npc2[i].position.z) ** 2)
    return x / le


def compareScenario(scenarioA, scenarioB):
    """
    calculate average similarity in same type scenarios
    """
    Alocation = scenarioA.npcLocation
    Blocation = scenarioB.npcLocation
    similarity = [0] * len(Alocation)
    for i in range(len(Alocation)):
        simiNpc = 0
        for j in range(len(Blocation)):
            simi = compareNpc(Alocation[i], Blocation[j])
            simiNpc += simi
        simiNpc /= len(Blocation)
        similarity[i] = simiNpc
    return sum(similarity) / len(Alocation)


def typeSimilarity(path):
    """
    calculate similarity of scenario type
    """
    types = [f for f in listdir(path)]
    dictScenario = {}

    for i in range(len(types)):
        dictToTypej = {}
        for j in range(len(types)):
            if i == j:
                continue
            print(types[j])
            scenarioTypeA = [f for f in listdir(path + '/' + types[i])]
            scenarioTypeB = [f for f in listdir(path + '/' + types[j])]
            minBs = []
            similarity = 0
            for k in range(len(scenarioTypeA)):

                mindist = float('inf')
                minB = 0
                simiPop = 0
                simiTypeK = 0
                for g in range(len(scenarioTypeB)):
                    scenarioType1 = pickle.load(open(path + '/' + types[i] + '/' + scenarioTypeA[k], "rb"))
                    scenarioType2 = pickle.load(open(path + '/' + types[j] + '/' + scenarioTypeB[g], "rb"))
                    simi = compareScenario(scenarioType1, scenarioType2)
                    simiTypeK += simi

                simiTypeK /= len(scenarioTypeB)
                similarity += simiTypeK
            similarity /= len(scenarioTypeA)
            dictToTypej[j] = similarity
        dictScenario[i] = dictToTypej

    print(dictScenario)


if __name__ == '__main__':
    typeSimilarity("classification")
