# Rank-constrained mixed-integer optimization for heterogeneous sensor location in route reconstruction
This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](https://github.com/Chenyi-Fu/Robust-Optimization-with-Moment-Dispersion-Ambiguity/blob/main/LICENSE).

Code for the paper "Rank-constrained mixed-integer optimization for heterogeneous sensor location in route reconstruction" by Chenyi Fu, Li Chen, Zhiyuan Lou.

# Cite
To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.
Below is the BibTex for citing this snapshot of the repository.
> @misc{fu2026rank,
  author =        {Chenyi Fu, Li Chen, Zhiyuan Lou},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Rank-constrained mixed-integer optimization for heterogeneous sensor location in route reconstruction}},
  year =          {2026},
  note =          {Available for download at https://github.com/Chenyi-Fu/A-generalized-optimization-framework-for-Heterogeneous-sensor-location-problem/edit/main},
  }

# Description
This code is used to generate the results in Section 5.

# Installation
1. Clone this repository
2. Install Gurobi (https://www.gurobi.com/). Gurobi is a commercial mathematical programming solver. Free academic licenses are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/).
3. Run the Python files

# Results
All detailed results for Tables 5-8 are available in the results folder, while other results are already in the paper.

# Replicating
1. Run the codes in "l20_test-download" to generate the results of GNN-PnS.
2. Run the "main_sensor.py" to generate the results of HSLM-MIP with and without VIs, and the 2-Step Heuristic.
3. Run the "main_sensor_benders.py" to generate the results of Benders decomposition.
4. Run the "main_model_random_cost.py" to generate the results of the 2-Step Heuristic in Table 5.
5. Run the "main_sensor_benders_random_cost.py" to generate the results of Benders decomposition in Table 5.
6. Run the "main_sensor_GA.py" to generate the results of GA.
7. Run the "main_sensor_graphmethod.py" to generate the results of the graph method.
8. "sensor_data.py" includes the data used.

# Support
For support in using this software, please get in touch with the author (cyfu@nwpu.edu.cn). Note that the software has been tested on a Windows OS only.
