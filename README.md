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

Table 1 presents the dataset statistics and detailed descriptions of the client settings in the manuscript.

Running the `FedCSL-CM.py/` file will produce the results of the FedCSL-CM method.

Running the `local_learning_methods.py/` file will produce the results of LR, SVM, XGB, LightGBM, and MLP in Table 3.

Running the `FedAvg.py/`, `fedProx.py/`,`SCAFFOLD.py/`, `FedNova.py/` and `FedKD.py/` files will produce the results of FedAvg, FedProx, SCAFFOLD, FedNova, and FedKD in Table 5.

Running the `FedAvg-RUS.py/`,`FedAvg-ROS.py/` and `FedAvg-SMOTE.py/` files will produce the results of FedAvg-RUS, FedAvg-ROS, and FedAvg-SMOTE in Table 7.

Running the `FedCSL.py/` and `FedCM.py/` files will produce the results of FedCSL and FedCM in Table 8.

Modifying the training set ratio parameter in the  `FocalUpatede.py` file and run files `FedAvg.py/`, `fedProx.py/`,`SCAFFOLD.py/`, `FedNova.py/`, `FedKD.py/`, and `FedCSL-CM.py/` accordingly will produce the results shown in Table 9.

Running the `FedCSL-CM_parameter_γ.py/`, `FedCSL-CM_parameter_α.py/`, and `FedCSL-CM_parameter_λ.py/` files will produce the results of FedCSL-CM in Table 10, 11 and 12.

Running the `Table_04 and Figure_02.py/` and `Table_06 and Figure_03.py/` files will produce the corresponding significance test analysis tables and figures.

Running the `Figure_04.py/` file will produce the result of Figure 4.

The results reported in Table 13 can be obtained by executing the aforementioned scripts.

Please note that when running the above files, there is no need to modify any parameters. You only need to change the `args.dataset` parameter to the corresponding dataset name. No modifications are required for the `options.py/` file either.

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
