# Deep Reinforcement Learning framework for fault detection in continuous chemical processes

This repository contains the related files for an academic project.

## Abstract
Modern industrial processes are subject to increasingly higher quality, safety, environmental and economic standards. However, faults with different severities that occur continuously in industrial processes can impact these requirements. Therefore, more reliable fault detection systems are needed. The earlier the detection, the greater the chance of at least mitigating potential losses. The complexity of processes and the recent increase in data availability have led to the use of data-driven approaches for fault detection in industrial systems. Most ML applications for this task employ supervised learning by mapping process variables and fault types. Deep Reinforcement Learning (DRL) has recently been applied for this purpose. DRL consists of building intelligent agents that interact with the process environment by performing actions and receiving rewards based on the quality of such actions. Most RL-based fault detection applications in industrial systems concern rotating machines. In chemical processes, most applications concern process control. This work develops an RL-based framework for fault detection in continuous chemical processes. Some of the challenges in this case concern the large number of variables, non-linear relationships and noisy measurements. The framework includes training, validation and testing steps for the DRL agents using the Deep Q-Learning algorithm. The classic Tennessee Eastman Process (TEP) benchmark was used as a case study. The PCA statistical technique, commonly used for process monitoring, was employed for comparison purposes. The results based on RL were at least equivalent to those obtained with PCA. More specifically, based on a grouping of faults considering their level of detection difficulty, the RL approach presented considerably lower values of MDR (missed detection rate) and TTD (time to detection), while maintaining reasonable values of FAR (false alarm rate) for all hard-to-detect group containing five faults. Faults initially undetectable in this case became detectable with MDR close to zero. Also, a Supervised Learning (SL) approach was developed in order to compare the proposed approach to a more complex and also traditional method. The comparison showed that both approaches generate similar performance, with the SL-based is less computationally costly, however being much more sensitive to hyperparameter tuning, neural network architecture and regularization, and data preparation. RL-based approaches were proven less dependent on those aformentioned matters and adequate for end-to-end frameworks from raw data detection, and thus can be very useful for fault detection in continuous industrial processes.

**Keywords**: Chemical process, Fault detection, Reinforcement learning, Deep Q-Learning, Machine learning, TEP benchmark

**Authors**: MANTOVANI, R. F., QUININO, R. C. Q, OLIVEIRA, E. D., BRAGA, A. P., ALMEIDA, G. M.

## Folders in this repository
- agents: contains files related to the Reinforcement Learning agents ("DQN_agent.py" - where the agents are stored and defined; "agent_params.py": auxiliary file with parameters for the RL agents);
- environment: contains the Custom RL environment developed to provide the interaction between the agents and the TEP simulation;
- models: contains the trained RL models;
- supervised_learning: this folder contains everything related to the Supervised Learning baselines, i.e., the trained models, the definition of the models and an utility file with some auxiliary functions;

## Files in this repository
- main.py: main training loop for the RL agents
- pca_baseline.py: main file for the PCA baseline
- supervised.py: main training loop for the SL models;
- utils.py: auxiliary functions;
