Replication package for "FedCSL-CM"
=
* Assembly date: 2026/03/05
* Author: Zhongyi Wang wzy172292979@163.com

Overview & contents
-
The code in this replication material generates the results of FedCSL-CM for the paper "A decentralized credit risk prediction method based on federated cost-sensitive learning"
* `creditdatasets/`: folder of raw data files
* `data description/`：folder of detailed dataset descriptions
* `models/`: folder of key modules in FedCM-CSL
* `utils/`: folder for data segmentation
* `model_param/`: folder of saved model parameters of FedCM-CSL
* `fedkd_model_param/`: folder of saved model parameters of FedKD
* `save/`: folder saved significance test analysis figures
  
Instructions & computational requirements
-

Running the `FedCSL-CM.py/` file will produce the results of FedCSL-CM.

Running the `FedAvg.py/`, `fedProx.py/`,`SCAFFOLD.py/`, `FedNova.py/` and `FedKD.py/` files will produce the results of FedAvg, FedProx, SCAFFOLD, FedNova, and FedKD.

Running the `FedAvg-RUS.py/`,`FedAvg-ROS.py/` and `FedAvg-SMOTE.py/` files will produce the results of FedAvg-RUS, FedAvg-ROS, and FedAvg-SMOTE.

Running the `FedCSL.py/` and `FedCM.py/` files will produce the results of FedCSL and FedCM.

Running the `FedCSL-CM_parameter_γ.py/`, `FedCSL-CM_parameter_α.py/`, and `FedCSL-CM_parameter_λ.py/` files will produce the results of FedCSL-CM.

The programming language is Python (version 3.8). The versions of the other packages and libraries used can be found in the `requirements.txt/` file.

Computing environment
-
* The operating system is Windows 11.

Data availability and provenance
-
The experimental datasets are located at `creditdatasets/`. Detailed descriptions of all datasets are are located at `data description/`.

Hardware and expected runtime
-
* The type of computer used is a 13th Gen Intel(R) Core (TM) i7-13700H @ 2.40GHz and 16.0GB RAM.
* The expected runtime is approximately 6 to 7 hours.
