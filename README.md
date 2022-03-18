# Generate Safety-Critical Scenarios For Testing ADSs in Simulator Based on Influential Traffic Factors

This project contains the implementation of CRISCO for constructing virtual scenarios to test Apollo in SVL simulator. CRISCO generates scenarios by extracted influential factors of traffic accidents. 

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

Installation of Apollo5.0: https://www.svlsimulator.com/docs/system-under-test/apollo5-0-instructions/

# Run
To generate scenarios, execute the main() of generate_test_scenarios.py in generation_scenario directory; to stop the running, please press Ctrl+C until the program exits.
To replay the recorded safety-violation scenarios, execute the main() of reproduce_all_safety_violations.py in reproduce_safety_violation_scenarios directory(reproduce all generated safety-violation scenarios) or reproduce_scenario.py in reproduce_safety_violation_scenarios directory(reproduce the generated safety-violation scenarios on specified one road type)

# Customize Influential Factors
Environment: Environment file in UserDefined directory
Behavior Patterns: BehaviorPatterns file in UserDefined directory
Accident-prone Area: RoadDistrict file in UserDefined directory

