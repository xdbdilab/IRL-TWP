# Task-aware Workload Prediction for Big Data Analytics Jobs in Clouds

## Abstract

Accurate workload prediction plays an essential role in resource provisioning in clouds, as it helps avoid over- and
under-provisioning of resources. Big data analytics jobs that process large-scale datasets are long-running and often have lower priorities
than transaction jobs; thus, many tasks constituting an analytics job may be queued on the scheduling list for a certain period of time until
dispatched. Based on this observation, we investigate a Task-aware Workload Prediction (TWP) problem to predict workload sequences
based on task execution states. The main goal of our work is to improve the accuracy of workload prediction by utilizing the information about
the tasks to be scheduled. Towards this goal, we fuse task information into a prediction algorithm and design an Inverse Reinforcement
Learning-based Task-aware Workload Prediction (IRL-TWP) algorithm to solve the TWP problem. IRL-TWP combines a Task Execution State
Extraction LSTM (TESE-LSTM), which extracts high-dimensional estimated task execution states, and a Generative Imitation Learning-based
(GAIL-TWP) algorithm to predict workload sequences. Extensive experiments on two well-known datasets demonstrate that the proposed
algorithm is more accurate and robust for long-term workload prediction in comparison with four state-of-the-art 
