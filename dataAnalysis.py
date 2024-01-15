from os import listdir
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import copy
import pickle
import os


def clusterAnalysis(data, path, i, filename, data1):
    try:
        outputfile = path + '/' + filename + '-classification' + str(i) + '.csv'
        k = 3
        iteration = 500
        kmodel = KMeans(n_clusters=k)
        kmodel.fit(data)

        # Count the number of each category
        r1 = pd.Series(kmodel.labels_).value_counts()
        r2 = pd.DataFrame(kmodel.cluster_centers_)
        r = pd.concat([r2, r1], axis=1)
        r.columns = list(data.columns) + ['number category']
        r = pd.concat([data, pd.Series(kmodel.labels_, index=data.index)], axis=1)
        r = pd.concat([r, data1], axis=1)
        r.columns = list(data.columns) + ['cluster category'] + list(data1.columns)
        r.to_csv(outputfile)
        print("sdfffgdfgfg", r)
        print(r['cluster category'][3])
    except Exception as e:
        print(e)


def fileProcessing(ck_path, npcSize):
    onlyfiles = [f for f in listdir(ck_path)]
    try:
        for i in range(len(onlyfiles)):
            datafile = ck_path + '/' + onlyfiles[i]
            data = pd.read_csv(datafile, header=None, encoding='utf-8', skip_blank_lines=True)
            print(data.shape[1])
            names = []
            for k in range(npcSize * 4):
                names += ['angle', 'position', 'distance', 'speed']
            for g in range(npcSize):
                names += ["npc" + str(g)]
            data.columns = names[:]
            data.dropna(axis=0, how='any', inplace=True)
            data.to_csv(datafile, index=False)
            data = pd.read_csv(datafile, encoding='utf-8')
            clusterAnalysis(data.iloc[:, :-npcSize], ck_path, i, onlyfiles[i], data.iloc[:, -npcSize:])
    except Exception as e:
        print(e)


def genePool(path, npcSize):
    files = [f for f in listdir(path)]

    try:
        actionAtom = []
        actionMotif = []
        speed = {}
        speed0 = []
        speed1 = []
        speed2 = []
        speed3 = []
        for i in range(len(files)):
            if 'classification' not in files[i]:
                continue
            datafile = path + '/' + files[i]
            data = pd.read_csv(datafile, encoding='utf-8')
            lise = data['cluster category'].value_counts()
            mincategoryNum = [lise[0], lise[1], lise[2]]
            mincategory = mincategoryNum.index(min(mincategoryNum))

            for j in range(npcSize):
                npc = data.loc[data['cluster category'] == mincategory, 'npc' + str(j)].values
                for spAndac in npc:
                    sps = []
                    if spAndac[-2] == ']':
                        ac = spAndac[-12:-2].split(',')
                        actionAtom.append([int(x) for x in ac])
                        sp = spAndac[2:-16].split(',')
                        for start in sp:
                            speed3.append(copy.deepcopy(float(start)))
                    else:
                        sp = spAndac[2:-5].split(',')
                        actionMotif.append(int(spAndac[-2]))
                        for stri in sp:
                            start = stri.index(':')
                            sps.append(float(stri[start + 2:]))
                        speed0.append(copy.deepcopy(sps[0]))
                        speed1.append(copy.deepcopy(sps[1]))
                        speed2.append(copy.deepcopy(sps[3]))

        minDeclare = min(speed0)
        maxDeclare = max(speed0)
        minacclare = min(speed1)
        maxacclare = max(speed1)
        minlanchange = min(speed2)
        maxlanchange = max(speed2)
        minatomSpeed = min(speed3)
        maxatomSpeed = max(speed3)

        datas = {}
        datas['minDeclare'] = minDeclare
        datas['maxDeclare'] = maxDeclare
        datas['minacclare'] = minacclare
        datas['maxacclare'] = maxacclare
        datas['minlanchange'] = minlanchange
        datas['maxlanchange'] = maxlanchange
        datas['minatomSpeed'] = minatomSpeed
        datas['maxatomSpeed'] = maxatomSpeed
        datas['actionAtom'] = actionAtom
        datas['actionMotif'] = actionMotif

        if os.path.isfile('data.obj'):
            os.system("rm " + 'data.obj')

        router = open('data.obj', 'wb')
        pickle.dump(datas, router)
        router.truncate()
        router.close()
        return datas

    except:
        print("failure")


def deleteFile(ck_path):
    files = [f for f in listdir(ck_path)]
    try:
        for i in range(len(files)):
            paths = ck_path + '/' + files[i]
            if os.path.exists(paths):
                os.remove(paths)
    except Exception as e:
        print(e)

