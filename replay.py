import math
import pickle
from environs import Env
import lgsvl
import random


def replay(scenario, sim):

    if sim.current_scene == lgsvl.wise.DefaultAssets.map_sanfrancisco:
        sim.reset()
    else:
        sim.load(lgsvl.wise.DefaultAssets.map_sanfrancisco)

    ego_location = scenario.egoLocation
    ego_speed = scenario.egoSpeed
    npc_location = scenario.npcLocation
    npc_speed = scenario.npcSpeed

    # create ego
    ego_state = lgsvl.AgentState()
    ego_state.transform = ego_location[0]
    ego = sim.add_agent("SUV", lgsvl.AgentType.NPC, ego_state)

    # create npc
    n = len(npc_location)
    m = len(ego_location)
    npc = []
    for i in range(n):
        npc_state = lgsvl.AgentState()
        npc_state.transform = npc_location[i][0]
        name = random.choice(["Sedan", "Jeep", "Hatchback"])
        npc.append(sim.add_agent(name, lgsvl.AgentType.NPC, npc_state))

    # util function
    def cal_speed(speed):
        return math.sqrt(speed.x ** 2 + speed.y ** 2 + speed.z ** 2)

    def on_waypoint(agent, index):
        print("waypoint {} reached".format(index))

    # ego waypoints
    ego_waypoints = []
    for i in range(1, len(ego_location)):
        wp = lgsvl.DriveWaypoint(position=ego_location[i].position, angle=ego_location[i].rotation,
                                 speed=ego_speed[i])
        ego_waypoints.append(wp)
    ego.follow(ego_waypoints)

    # npc wypoints
    for i in range(n):
        npc_waypoints = []
        for j in range(1, len(npc_location[i])):
            wp = lgsvl.DriveWaypoint(position=npc_location[i][j].position, angle=npc_location[i][j].rotation,
                                     speed=cal_speed(npc_speed[i][j]))
            npc_waypoints.append(wp)
        npc[i].follow(npc_waypoints)

    # set simulation camera
    cam_tr = ego_location[0]
    up = lgsvl.utils.transform_to_up(cam_tr)
    forward = lgsvl.utils.transform_to_forward(cam_tr)
    cam_tr.position += up * 3 - forward * 5
    sim.set_sim_camera(cam_tr)

    # run simulation
    cnt = 0
    while cnt < m:
        cnt += 1
        sim.run(0.25)


def main(path):
    with open(path, 'rb') as f:
        scenario = pickle.load(f)

    # initial simulation
    env = Env()
    sim = lgsvl.Simulator(env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host),
                          env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))

    for i in range(4):
        print("scenario[{}]".format(i))
        replay(scenario[i], sim)


if __name__ == '__main__':
    main('D:/Workspace/WeatherPesdraintNpcWithMutation2/GaCheckpointsCrossroads/generation--at-01-01-2022-00-20-18')
