import random
import pprint
import pickle
import os
from datetime import datetime
# import util
import copy
import math

import util


class MutlChromosome:
    def __init__(self, bounds, NPC_size, time_size, pools):
        self.ttc = 0
        self.smoothness = 0
        self.pathSimilarity = 0
        self.scenario = [[[] for i in range(time_size)] for j in range(NPC_size)]
        self.bounds = bounds
        self.MinNpcSituations = [[0 for i in range(time_size)] for j in range(NPC_size)]
        self.NPC_size = NPC_size
        self.time_size = time_size
        self.timeoutTime = 300
        self.egoSpeed = []
        self.egoLocation = []
        self.npcSpeed = [[] for i in range(NPC_size)]
        self.npcLocation = [[] for i in range(NPC_size)]
        self.isCollision = None
        self.npcDetail = []
        self.npcAction = []
        self.data = []
        self.pools = pools
        self.weathers = []

    def restart_init(self):
        """
        init of restart operation
        """
        for i in range(self.NPC_size):
            for j in range(self.time_size):
                isAtom = random.randint(0, 1)
                # Atom Gene
                if isAtom == 0:
                    v = []
                    index = random.randrange(0, len(self.pools['actionAtom']))
                    a = self.pools['actionAtom'][index]
                    for k in range(4):
                        v1 = random.uniform(self.pools['minatomSpeed'], self.pools['maxatomSpeed'])
                        v.append(v1)

                    x = random.uniform(self.bounds[4][0], self.bounds[4][1])
                    z = random.uniform(self.bounds[5][0], self.bounds[5][1])
                    n_v = random.uniform(self.bounds[6][0], self.bounds[6][1])
                    idle = random.randint(self.bounds[7][0], self.bounds[7][1])
                    self.scenario[i][j].append(copy.deepcopy(v))
                    self.scenario[i][j].append(copy.deepcopy(a))
                    self.scenario[i][j].append(x)
                    self.scenario[i][j].append(z)
                    self.scenario[i][j].append(n_v)
                    self.scenario[i][j].append(idle)
                # Motif Gene
                else:
                    v1 = random.uniform(self.pools['minDeclare'], self.pools['maxDeclare'])
                    v2 = random.uniform(self.pools['minacclare'], self.pools['maxacclare'])
                    v3 = random.uniform(self.pools['minlanchange'], self.pools['maxlanchange'])
                    v = {"decelerate": v1, "accalare": v2, "stop": 0, "lanechangspeed": v3}
                    a = random.randrange(0, 5)
                    x = random.uniform(self.bounds[4][0], self.bounds[4][1])
                    z = random.uniform(self.bounds[5][0], self.bounds[5][1])
                    n_v = random.uniform(self.bounds[6][0], self.bounds[6][1])
                    idle = random.randint(self.bounds[7][0], self.bounds[7][1])
                    self.scenario[i][j].append(copy.deepcopy(v))
                    self.scenario[i][j].append(copy.deepcopy(a))
                    self.scenario[i][j].append(x)
                    self.scenario[i][j].append(z)
                    self.scenario[i][j].append(n_v)
                    self.scenario[i][j].append(idle)

            row = random.randint(-1, 1)
            col = random.randint(-1, 1)
            while row == 0 and col == 0:
                row = random.randint(-1, 1)
            self.npcDetail.append([row, col])

        for i in range(5):
            self.weathers.append(random.uniform(0, 1))

    def rand_init(self):
        lastState = []
        for i in range(self.NPC_size):
            for j in range(self.time_size):
                isMutli = random.randint(0, 1)
                if lastState:
                    if lastState[-1] == 0 and isMutli == 0:
                        isMutli = 1
                lastState.append(isMutli)
                # Atom
                if isMutli == 0:
                    v = []
                    a = []
                    for k in range(4):
                        v1 = random.uniform(self.bounds[2][0], self.bounds[2][1])  # Init velocity
                        a1 = random.randrange(self.bounds[3][0], self.bounds[3][1])  # Init action
                        v.append(copy.deepcopy(v1))
                        a.append(copy.deepcopy(a1))

                    x = random.uniform(self.bounds[4][0], self.bounds[4][1])
                    z = random.uniform(self.bounds[5][0], self.bounds[5][1])
                    n_v = random.uniform(self.bounds[6][0], self.bounds[6][1])
                    idle = random.randint(self.bounds[7][0], self.bounds[7][1])
                    self.scenario[i][j].append(copy.deepcopy(v))
                    self.scenario[i][j].append(copy.deepcopy(a))

                    self.scenario[i][j].append(x)
                    self.scenario[i][j].append(z)
                    self.scenario[i][j].append(n_v)
                    self.scenario[i][j].append(idle)

                else:
                    v1 = random.uniform(0, 1)
                    v2 = random.uniform(self.bounds[0][0], self.bounds[0][1])
                    v3 = random.uniform(self.bounds[0][0], self.bounds[0][1])
                    v = {"decelerate": v1, "accalare": v2, "stop": 0, "lanechangspeed": v3}
                    a = random.randrange(self.bounds[1][0], self.bounds[1][1])

                    x = random.uniform(self.bounds[4][0], self.bounds[4][1])
                    z = random.uniform(self.bounds[5][0], self.bounds[5][1])
                    n_v = random.uniform(self.bounds[6][0], self.bounds[6][1])
                    idle = random.randint(self.bounds[7][0], self.bounds[7][1])
                    self.scenario[i][j].append(v)
                    self.scenario[i][j].append(a)

                    self.scenario[i][j].append(x)
                    self.scenario[i][j].append(z)
                    self.scenario[i][j].append(n_v)
                    self.scenario[i][j].append(idle)
            row = random.randint(-1, 1)
            col = random.randint(-1, 1)
            while row == 0 and col == 0:
                row = random.randint(-1, 1)

            self.npcDetail.append([row, col])
        for i in range(5):
            self.weathers.append(random.uniform(0, 1))

    def decoding(self):
        """
        save the scenario as file
        """
        s_f = open('scenario.obj', 'wb')
        pickle.dump([self.scenario, self.npcDetail, self.weathers], s_f)
        s_f.truncate()
        s_f.close()

        # rerun the scenario when exception
        for x in range(0, 30):
            if os.path.isfile('result.obj'):
                os.remove("result.obj")
            os.system("python3 WeatherPesdraintNpcWithMutation2/simulation.py scenario.obj result.obj")
            resultObj = None

            # Read fitness score
            if os.path.isfile('result.obj'):
                f_f = open('result.obj', 'rb')
                resultObj = pickle.load(f_f)
                f_f.close()

            if resultObj is not None and resultObj['ttc'] != '' and resultObj['pathSimilarity'] != '':
                return resultObj
            else:
                util.print_debug(" ***** " + str(x) + "th/10 trial: Fail to get fitness, try again ***** ")

        return None

    # calculate the distance between two vehicle
    def findTwoVehicle(self, j, i):
        npc_x = self.npcLocation[j][i].position.x
        npc_y = self.npcLocation[j][i].position.y
        npc_z = self.npcLocation[j][i].position.z
        ego_x = self.egoLocation[i].position.x
        ego_y = self.egoLocation[i].position.y
        ego_z = self.egoLocation[i].position.z
        distance = math.sqrt((ego_x - npc_x) ** 2 + (ego_y - npc_y) ** 2 + (ego_z - npc_z) ** 2)
        return distance

    def func(self, gen=None, lisFlag=False):
        """
        handling of end-of-run scenarios
        """
        resultObj = self.decoding()
        self.ttc = float(resultObj['ttc'])
        self.smoothness = float(resultObj['smoothness'])
        self.pathSimilarity = float(resultObj['pathSimilarity'])
        self.MinNpcSituations = resultObj['MinNpcSituations']

        self.egoSpeed = resultObj['egoSpeed']
        self.egoLocation = resultObj['egoLocation']
        self.npcSpeed = resultObj['npcSpeed']
        self.npcLocation = resultObj['npcLocation']
        self.isCollision = resultObj['isCollision']
        self.npcActions = resultObj['npcAction']

        print("self.isCollision", self.isCollision)
        if self.isCollision:
            collisionTime = self.egoSpeed.index("collision")
            self.egoSpeed.pop(collisionTime)
            self.egoLocation.pop(collisionTime)
        else:
            collisionTime = -1
        NpcNums = len(self.npcSpeed)
        k = 0

        for i in range(0, len(self.egoSpeed), 12):
            npcAction = []
            if i <= collisionTime < i + 12:
                isCritical = True
            else:
                isCritical = False
            npcEgodistance = []
            npcVelo = []
            for j in range(NpcNums):
                npcEgodistance.append(self.findTwoVehicle(j, i))
                npcAction.append(self.npcActions[j][k])
                npcVelo.append(self.npcSpeed[j][i])

            self.data.append(npcEgodistance + npcAction + npcVelo + [isCritical])

            k += 0
            if isCritical:
                break
        print("self.data", len(self.data))

        if self.isCollision:
            # An accident
            print("collision=============")
            util.print_debug(" ***** Found an accident where ego is at fault ***** ")
            # Dump the scenario where causes the accident
            if not os.path.exists('AccidentScenarioCrossroads'):
                os.mkdir('AccidentScenarioCrossroads')
            now = datetime.now()
            date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
            ckName = 'AccidentScenarioCrossroads/accident-gen' + str(gen) + '-' + date_time
            if lisFlag:
                ckName = ckName + "-LIS"
            a_f = open(ckName, 'wb')
            pickle.dump(self, a_f)
            a_f.truncate()
            a_f.close()


if __name__ == '__main__':
    a = [[1, 2], [0, 4], [0, 67], [0, 3]]
    chromosome = MutlChromosome(a, 5, 10)
    chromosome.rand_init()
    pprint.pprint(chromosome.scenario)
