# MOSAT: Multi-Objective Search-based ADS Testing

This project contains the implementation of MOSAT to test Apollo in SVL simulator. 
MOSAT applies multi-objective genetic algorithm to generate virtual scenarios which can find safety violations of ADSs.

The generation approach requires the following dependencies to run:

	1. SVL simulator: https://www.svlsimulator.com/
	
	2. Apollo autonomous driving platform: https://github.com/ApolloAuto/apollo


# Prerequisites

* A 8-core processor and 16GB memory minimum
* Ubuntu 18.04 or later
* Python 3.6 or higher
* NVIDIA graphics card: NVIDIA proprietary driver (>=455.32) must be installed
* CUDA upgraded to version 11.1 to support Nvidia Ampere (30x0 series) GPUs
* Docker-CE version 19.03 and above
* NVIDIA Container Toolkit


# SVL - A Python API for SVL Simulator

Documentation is available on: https://www.svlsimulator.com/docs/

# Apollo - A high performance, flexible architecture which accelerates the development, testing, and deployment of Autonomous Vehicles

Website of Apollo: https://apollo.auto/

Installation of Apollo6.0: https://www.svlsimulator.com/docs/system-under-test/apollo6-0-instructions/

# Run
To replay the recorded safety-violation scenarios, execute the main() of replay.py and set the file path of the scenario to be replayed.


