import os
import copy
import random
import pickle
import sys
from datetime import datetime
import util
import math
from MutlChromosome import MutlChromosome
import generateRestart
import dataAnalysis
import dataProcessing


class MultiObjGenticAlgorithm:
    def __init__(self, bounds, pm, pc, pop_size, NPC_size, time_size, max_gen):
        self.bounds = bounds
        self.pm = pm
        self.pc = pc
        self.pop_size = pop_size
        self.NPC_size = NPC_size
        self.time_size = time_size
        self.max_gen = max_gen
        self.pop = []
        self.bests = [0] * max_gen
        self.bestIndex = 0
        self.g_best = None
        self.ck_path = None  # Checkpoint path, if set, GE will start from the checkpoint (population object)
        self.touched_chs = []  # Record which chromosomes have been touched in each generation

        self.isInLis = False  # Set flag for local iterative search (LIS)
        self.minLisGen = 2  # Min gen to start LIS
        self.numOfGenInLis = 5  # Number of gens in LIS
        self.hasRestarted = False
        self.lastRestartGen = 0
        self.bestYAfterRestart = 0

    def set_checkpoint(self, ck_path):
        self.ck_path = ck_path

    def take_checkpoint(self, obj, ck_name):
        if os.path.exists('GaCheckpointsCrossroads') == False:
            os.mkdir('GaCheckpointsCrossroads')
        ck_f = open('GaCheckpointsCrossroads/' + ck_name, 'wb')
        pickle.dump(obj, ck_f)
        ck_f.truncate()
        ck_f.close()

    def index_of(self, a, list):
        """
        Function to find index of list
        """
        for i in range(0, len(list)):
            if list[i] == a:
                return i
        return -1

    def sort_by_values(self, list1, values1):
        """
        Function to sort by values
        """
        sorted_list1 = []
        while (len(sorted_list1) != len(list1)):
            if self.index_of(min(values1), values1) in list1:
                sorted_list1.append(self.index_of(min(values1), values1))
            values1[self.index_of(min(values1), values1)] = math.inf
        return sorted_list1

    def crowding_distance(self, front):
        """
        Function to calculate crowding distance
        """
        distance = [0 for i in range(0, len(front))]

        values1 = []
        values2 = []
        values3 = []
        for i in range(len(self.pop)):
            values1.append(self.pop[i].ttc)
            values2.append(self.pop[i].smoothness)
            values3.append(self.pop[i].pathSimilarity)
        maxTtc = max(values1)
        minTtc = min(values1)
        normalizationValue1 = maxTtc - minTtc
        maxSmoothness = max(values2)
        minSmoothness = min(values2)
        normalizationValue2 = maxSmoothness - minSmoothness
        maxPathSimilarity = max(values3)
        minPathSimilarity = min(values3)
        normalizationValue3 = maxPathSimilarity - minPathSimilarity
        # print("normalizationValue",normalizationValue)
        sorted1 = self.sort_by_values(front, values1)
        sorted2 = self.sort_by_values(front, values2)
        sorted3 = self.sort_by_values(front, values3)

        distance[0] = sys.maxsize
        distance[len(front) - 1] = sys.maxsize
        for k in range(1, len(front) - 1):
            distance[k] += (self.pop[sorted1[k + 1]].ttc - self.pop[sorted1[k - 1]].ttc) / normalizationValue1
        for k in range(1, len(front) - 1):
            distance[k] += (self.pop[sorted2[k + 1]].smoothness - self.pop[
                sorted2[k - 1]].smoothness) / normalizationValue2
        for k in range(1, len(front) - 1):
            distance[k] += (self.pop[sorted3[k + 1]].pathSimilarity - self.pop[
                sorted3[k - 1]].pathSimilarity) / normalizationValue3
        return distance

    def fast_non_dominated_sort(self):
        """
        Function to carry out NSGA-II's fast non dominated sort
        """
        S = [[] for i in range(len(self.pop))]
        self.front = [[]]
        n = [0 for i in range(len(self.pop))]
        rank = [0 for i in range(len(self.pop))]

        for p in range(len(self.pop)):
            S[p] = []
            n[p] = 0
            for q in range(len(self.pop)):
                if (self.pop[p].ttc > self.pop[q].ttc and self.pop[p].smoothness >= self.pop[q].smoothness and self.pop[
                    p].pathSimilarity >= self.pop[q].pathSimilarity) or \
                        (self.pop[p].smoothness > self.pop[q].smoothness and self.pop[p].ttc >= self.pop[q].ttc and
                         self.pop[p].pathSimilarity >= self.pop[q].pathSimilarity) or \
                        (self.pop[p].pathSimilarity > self.pop[q].pathSimilarity and self.pop[p].ttc >= self.pop[
                            q].ttc and
                         self.pop[p].smoothness >= self.pop[q].smoothness):
                    if q not in S[p]:
                        S[p].append(q)
                elif (self.pop[p].ttc < self.pop[q].ttc and self.pop[p].smoothness <= self.pop[q].smoothness and
                      self.pop[
                          p].pathSimilarity <= self.pop[q].pathSimilarity) or \
                        (self.pop[p].smoothness < self.pop[q].smoothness and self.pop[p].ttc <= self.pop[q].ttc and
                         self.pop[p].pathSimilarity <= self.pop[q].pathSimilarity) or \
                        (self.pop[p].pathSimilarity < self.pop[q].pathSimilarity and self.pop[p].ttc <= self.pop[
                            q].ttc and
                         self.pop[p].smoothness <= self.pop[q].smoothness):
                    n[p] = n[p] + 1
            if n[p] == 0:
                rank[p] = 0
                if p not in self.front[0]:
                    self.front[0].append(p)

        i = 0
        while (self.front[i] != []):
            Q = []
            for p in self.front[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if (n[q] == 0):
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i = i + 1
            self.front.append(Q)
        del self.front[len(self.front) - 1]
        return self.front

    def ga(self):
        """
        Genetic Algorithm
        """
        # Load from checkpoint if not none
        if self.ck_path is not None:
            ck = open(self.ck_path, 'rb')
            self.pop = pickle.load(ck)
            ck.close()
        elif not self.isInLis:
            self.init_pop()

        # Start evolution
        for i in range(self.max_gen):  # i th generation.
            now = datetime.now()
            date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
            print(" \n\n*** " + str(i) + "th generation ***")
            util.print_debug(" \n\n*** " + str(i) + "th generation ***" + date_time)
            # Make sure we clear touched_chs history book every gen
            self.touched_chs = []
            self.beforePop = []
            for t in range(len(self.pop)):
                self.beforePop.append(copy.deepcopy(self.pop[t]))
            self.cross()
            self.mutation()
            flag = []
            for m in range(len(self.beforePop)):
                for h in range(len(self.pop)):
                    if self.beforePop[m].scenario == self.pop[h].scenario:
                        # self.beforePop.pop(m)
                        if m not in flag:
                            flag.append(m)
            flag.reverse()
            for index in flag:
                self.beforePop.pop(index)
            self.pop += self.beforePop

            self.select_NDsort_roulette()
            # The best fitness score in current generation
            best = self.pop[0]
            # Record the scenario with the best fitness score in i th generation
            self.bests[i] = best
            self.generate_best = copy.deepcopy(self.pop[:self.pop_size])

            noprogress = False
            ave = 0
            if i >= self.lastRestartGen + 5:
                for j in range(i - 5, i):
                    ave += self.bests[j].ttc + self.bests[j].smoothness + self.bests[j].pathSimilarity
                ave /= 5
                if ave <= best.ttc:
                    self.lastRestarGen = i
                    noprogress = True

            # Record the best fitness score across all generations
            self.g_best = copy.deepcopy(self.pop[0])

            N_generation = self.pop
            N_b = self.g_best  # Record the scenario with the best score over all generations

            # Update the checkpoint of the best scenario so far
            self.take_checkpoint(N_b, 'best_scenario.obj')

            # Checkpoint this generation
            self.take_checkpoint(N_generation, 'last_gen.obj')

            # Checkpoint every generation
            self.take_checkpoint(N_generation, 'generation-' + str(i) + '-at-' + date_time)

            if noprogress == True and not self.isInLis:
                util.print_debug(" ###### Restart Based on Generation: " + str(i) + " ###### " + date_time)
                oldCkName = 'GaCheckpointsCrossroads'
                dataProcessing.getAllCheckpoints(oldCkName, self.NPC_size, self.time_size)
                dataAnalysis.fileProcessing('clusterData', self.NPC_size)
                pools = dataAnalysis.genePool('clusterData', self.NPC_size)
                nums = (len(pools['actionAtom']) + len(pools['actionMotif'])) // 2
                newPop = generateRestart.generateRestart(nums, self.bounds, self.NPC_size, self.time_size, pools)
                dataAnalysis.deleteFile('clusterData')
                self.pop = copy.deepcopy(newPop)
                self.hasRestarted = True
                best, self.bestIndex = self.find_best()
                self.bestYAfterRestart = best.ttc
                self.lastRestartGen = i

        return self.g_best

    def init_pop(self):
        for i in range(self.pop_size):
            chromosome = MutlChromosome(self.bounds, self.NPC_size, self.time_size, None)
            chromosome.rand_init()
            chromosome.func()
            self.pop.append(chromosome)

    def mutation(self):
        k = 0
        while (k < len(self.pop)):
            eachChs = self.pop[k]
            k += 1
            # Check mutation probability
            if self.pm >= random.random():
                npc_index = random.randint(0, eachChs.NPC_size - 1)
                time_index = random.randint(0, eachChs.time_size - 1)

                # Record which chromosomes have been touched
                self.touched_chs.append(eachChs)
                actionIndex = random.randint(0, 2)
                print("eachChs.MinNpcSituations", eachChs.MinNpcSituations)

                # Abandon when distance of two vehicle is too large
                if not eachChs.MinNpcSituations:
                    v = []
                    a = []
                    for k in range(4):
                        v1 = random.uniform(self.bounds[2][0], self.bounds[2][1])  # Init velocity
                        a1 = random.randrange(self.bounds[3][0], self.bounds[3][1])  # Init action
                        v.append(copy.deepcopy(v1))
                        a.append(copy.deepcopy(a1))

                    if actionIndex == 0:
                        if isinstance(eachChs.scenario[npc_index][time_index][0], list):
                            eachChs.scenario[npc_index][time_index][0] = copy.deepcopy(v)
                        else:
                            v4 = random.uniform(0, 1)
                            v5 = random.uniform(self.bounds[0][0], self.bounds[0][1])
                            v6 = random.uniform(self.bounds[0][0], self.bounds[0][1])
                            v7 = {"decelerate": v4, "accalare": v5, "stop": 0, "lanechangspeed": v6}
                            eachChs.scenario[npc_index][time_index][0] = copy.deepcopy(v7)
                    else:
                        if isinstance(eachChs.scenario[npc_index][time_index][1], list):
                            eachChs.scenario[npc_index][time_index][1] = copy.deepcopy(a)
                        else:
                            eachChs.scenario[npc_index][time_index][1] = random.randrange(self.bounds[1][0],
                                                                                          self.bounds[1][1])
                # Mutation
                else:
                    npcs = len(eachChs.MinNpcSituations)
                    times = len(eachChs.MinNpcSituations[0])
                    minDist = eachChs.MinNpcSituations[0][0][0]
                    minSituation = eachChs.MinNpcSituations[0][0][1]
                    minNpcIndex = eachChs.MinNpcSituations[0][0][2]
                    minTimesIndex = 0
                    for i in range(npcs):
                        for j in range(times):
                            if eachChs.MinNpcSituations[i][j] == 130:
                                continue
                            if minDist > eachChs.MinNpcSituations[i][j][0]:
                                minDist = eachChs.MinNpcSituations[i][j][0]
                                minSituation = eachChs.MinNpcSituations[i][j][1]
                                minNpcIndex = eachChs.MinNpcSituations[i][j][2]
                                minTimesIndex = j

                    if isinstance(eachChs.scenario[minNpcIndex][minTimesIndex][0], dict):
                        if minSituation == "OneLaneBefore":
                            eachChs.scenario[minNpcIndex][minTimesIndex][0]["decelerate"] = random.uniform(0, 1)
                        elif minSituation == 'before':
                            eachChs.scenario[minNpcIndex][minTimesIndex][0]["decelerate"] = random.uniform(0, 1)
                        elif minSituation == 'parall':
                            eachChs.scenario[minNpcIndex][minTimesIndex][0]["lanechangspeed"] = random.uniform(1, 2)
                        elif minSituation == "OneLaneAfter":
                            eachChs.scenario[minNpcIndex][minTimesIndex][0]["accalare"] = random.uniform(1, 2)
                        else:
                            eachChs.scenario[minNpcIndex][minTimesIndex][0]["accalare"] = random.uniform(1, 2)
                    else:
                        v = []
                        a = []
                        for k in range(4):
                            v1 = random.uniform(self.bounds[2][0], self.bounds[2][1])  # Init velocity
                            a1 = random.randrange(self.bounds[3][0], self.bounds[3][1])  # Init action
                            v.append(copy.deepcopy(v1))
                            a.append(copy.deepcopy(a1))
                        if actionIndex == 0:
                            eachChs.scenario[npc_index][time_index][0] = copy.deepcopy(v)
                        else:
                            eachChs.scenario[npc_index][time_index][1] = copy.deepcopy(a)

                le = len(eachChs.scenario) // 2
                # choice chromosome [i] ~[i+le]
                for i in range(le):
                    if self.pm >= random.random():
                        preindex = random.randint(0, len(eachChs.scenario[i]) - 1)
                        afindex = random.randint(0, len(eachChs.scenario[i]) - 1)

                        while preindex > afindex:
                            preindex = random.randint(0, len(eachChs.scenario[i]) - 1)
                        while preindex <= afindex:
                            tmp = eachChs.scenario[i][preindex]
                            eachChs.scenario[i][preindex] = eachChs.scenario[i + le][preindex]
                            eachChs.scenario[i + le][preindex] = tmp
                            preindex += 1

                for scenario in eachChs.scenario:
                    if self.pm >= random.random():
                        random.shuffle(scenario)

    def cross(self):
        """
        Implementation of random crossover
        """
        flag = []
        for k in range(int(len(self.pop) / 2.0)):
            # Check crossover probability
            if self.pc > random.random():
                # randomly select 2 chromosomes(scenarios) in pops
                i = 0
                j = 0
                while i == j or (i in flag or j in flag):
                    i = random.randint(0, self.pop_size - 1)
                    j = random.randint(0, self.pop_size - 1)

                flag.append(i)
                flag.append(j)
                pop_i = self.pop[i]
                pop_j = self.pop[j]

                # Record which chromosomes have mutated or crossed
                self.touched_chs.append(self.pop[i])
                self.touched_chs.append(self.pop[j])

                # Select cross index
                swap_index = random.randint(0, pop_i.NPC_size - 1)
                temp = copy.deepcopy(pop_j.scenario[swap_index])
                pop_j.scenario[swap_index] = copy.deepcopy(pop_i.scenario[swap_index])
                pop_i.scenario[swap_index] = temp

    def select_NDsort_roulette(self):
        """
        Pareto sort on scenarios target fitness values
        """
        sum_f = 0
        for i in range(0, len(self.pop)):
            if self.pop[i].ttc == 0:
                self.pop[i].ttc = -sys.maxsize - 1
        v = []
        # Start sort then select
        crowding_distance_values = []
        non_dominated_sorted_solution = self.fast_non_dominated_sort()
        # Calculate the fitness value of crowding distance for each target in the scenarios
        for i in range(0, len(non_dominated_sorted_solution)):
            crowding_distance_values.append(
                self.crowding_distance(non_dominated_sorted_solution[i][:]))
        # Sort non-dominant relationships in target fitness values in the scenarios
        for i in range(0, len(non_dominated_sorted_solution)):
            non_dominated_sorted_solution2_1 = [
                self.index_of(non_dominated_sorted_solution[i][j], non_dominated_sorted_solution[i]) for j in
                range(0, len(non_dominated_sorted_solution[i]))]

            front22 = self.sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values[i][:])
            front = [non_dominated_sorted_solution[i][front22[j]] for j in
                     range(0, len(non_dominated_sorted_solution[i]))]
            front.reverse()

            for value in front:
                selectedChromosome = MutlChromosome(self.bounds, self.NPC_size, self.time_size, None)
                selectedChromosome.scenario = self.pop[value].scenario
                selectedChromosome.ttc = self.pop[value].ttc
                selectedChromosome.smoothness = self.pop[value].smoothness
                selectedChromosome.pathSimilarity = self.pop[value].pathSimilarity
                selectedChromosome.MinNpcSituations = self.pop[value].MinNpcSituations
                selectedChromosome.npcDetail = self.pop[value].npcDetail
                selectedChromosome.npcAction = self.pop[value].npcAction
                selectedChromosome.egoSpeed = self.pop[value].egoSpeed
                selectedChromosome.egoLocation = self.pop[value].egoLocation
                selectedChromosome.npcSpeed = self.pop[value].npcSpeed
                selectedChromosome.npcLocation = self.pop[value].npcLocation
                selectedChromosome.isCollision = self.pop[value].isCollision
                selectedChromosome.weathers = self.pop[value].weathers
                v.append(selectedChromosome)

                if len(v) == self.pop_size:
                    break

        self.pop = copy.deepcopy(v)

    def find_best(self):
        best = copy.deepcopy(self.pop[0])
        bestIndex = 0
        return best, bestIndex


if __name__ == '__main__':
    bounds = [[0, 70], [0, 3]]
    algorithm = MultiObjGenticAlgorithm(bounds, 0.4, 0.8, 4, 4, 5, 30)
    algorithm.ga()
