# Enhancing Deployment-time Predictive Model Robustness for
Code Analysis and Optimization: Artifacts Evaluation Instructions

## Preliminaries

This interactive Jupyter notebook provides a small-scale demo for showcasing task definition, client RL search, client RL parameter tuning, and evaluation of case studies reported in the paper.

The main results of our CGO 2024 paper apply Prom to 5 case studies to detect their misprediction samples. The evaluation presented in our paper ran on a much larger dataset and for longer. This notebook contains minimal working examples designed to be evaluated in a reasonable amount of time (approximately 20 minutes).

## Instructions for Experimental Workflow:

Before you start, please first make a copy of the notebook by going to the landing page. Then select the checkbox next to the notebook titled *main.ipynb*, then click "**Duplicate**".

Click the name of the newly created Jupyter Notebook, e.g. **AE-Copy1.ipynb**. Next, select "**Kernel**" > "**Restart & Clear Output**". Then, repeatedly press the play button (the tooltip is "run cell, select below") to step through each cell of the notebook.

Alternatively, select each cell in turn and use "**Cell**"> "**Run Cell**" from the menu to run specific cells. Note that some cells depend on previous cells being executed. If any errors occur, ensure all previous cells have been executed.

## Important Notes

**Some cells can take more than half an hour to complete; please wait for the results until step to the next cell.**

High load can lead to a long wait for results. This may occur if multiple reviewers are simultaneously trying to generate results.

The experiments are customisable as the code provided in the Jupyter Notebook can be edited on the spot. Simply type your changes into the code blocks and re-run using **Cell > Run Cells** from the menu.

## Links to The Paper

For each step, we note the section number of the submitted version where the relevant technique is described or data is presented.

The main results are presented in Figures 7-10 and Table 2 of the submitted paper.



# Demo 1: The Prom Tutorial 

This demo corresponds to the simplified search space definition example given in Figure 2. Note that we have refactored the code; hence there are small changes in the API. This is a small-scale demo for case study 2 of thread coarsening. The full-scale evaluation used in the paper takes over 48 hours to run.

## Step 1. Train the underlying model

This problem develops a model to determine the optimal OpenCL GPU thread coarsening factor for performance optimization. Following the original paper, an ML model predicts a coarsening factor (ranging  from 1 to 32) for a test OpenCL kernel, where 1 indicates no coarsening. Following the setup of  the DeepTune, we train and test the models using the labeled dataset  from them, comprising 17 OpenCL kernels from three benchmark  suites across four GPU platforms.

we train the baseline model using  leave-one-out cross-validation, which involves training the baseline model on 16 OpenCL kernels and testing on another one.

#### Training dataset and calibration dataset partitioning

```
#Partition your dataset into a training dataset and a calibration dataset:
python data_partition.py --dataset /path/to/your/dataset --calibration_ratio 0.1
```

#### Training the underlying model

```
# Train your model using the training dataset. This script will load the training data, train the model, and save it for later use
python model_definition.py --model trained_model.pkl
```

#### Training the anomaly detector

```
# Train the prom anolamy detector
python prom_train.py --calibration_data calibration_data.pkl
```



## Step 2. Prom on deployment

First, to introduce data drift, we train the ML models on OpenCL benchmarks from two suites and then test the trained model on another left-out benchmark suite.

#### Native deployment

```
# Test the anomaly detector with the calibration dataset:
python test_anomaly_detector.py --calibration_data calibration_data.pkl --model trained_model.pkl --non_func all
```

#### Detecting drifting samples

```
# Deploy your model to detect drifting data:
python deploy_model.py --model trained_model.pkl --test_data /path/to/test_data.pkl --non_func all
```



## Step 3. Improve Deployment Time Performance

Prom can enhance the performance of deployed ML systems through incremental learning.

#### Retraining the model with incremental learning:

```
# Periodically retrain your model with new data to incorporate changes and improve accuracy:
python retrain_model.py --model trained_model.pkl --new_data /path/to/new_data.pkl --retrain_ratio 0.05
```



# Demo 2: Experimental Evaluation

Here, we provide a small-sized evaluation to showcase the working mechanism of the Prom on five case studies. A full-scale evaluation, which takes more than a week to run, is provided through the Docker image (with detailed instructions on our project Github).

### Case Study 1: Thread Coarsening (Section 6.1)

This problem develops a model to determine the optimal OpenCL GPU thread coarsening factor for performance optimization. Following other works, an ML model predicts a coarsening factor (ranging  from 1 to 32) for a test OpenCL kernel, where 1 indicates no coarsening. Underlying models. We train the baseline model using  leave-one-out cross-validation, which involves training the base-  line model on 16 OpenCL kernels and testing on another one. We  then repeat this process until all benchmark suites have been tested  once. To introduce data drift, we train the ML models on OpenCL  benchmarks from two suites and then test the trained model on  another left-out benchmark suite.

This demo corresponds to Figure 7(a), 8(a), 9(a) and 11(a) of the submitted manuscript.

*approximate runtime = 10 minutes for one benchmark*

```
# Train the underlying model and deploy to drifting environment. The results correspond to Figure 7(a).
python model_definition.py --model trained_model.pkl
python prom_train.py --calibration_data calibration_data.pkl

# Using prom to detect the drifting data. The results correspond to Figure 8(a) and Figure 11(a).
python deploy_model.py --model trained_model.pkl --test_data /path/to/test_data.pkl --non_func all

# Increment learning with detected misprediction samples. The results correspond to Figure 9(a).
python retrain_model.py --model trained_model.pkl --new_data /path/to/new_data.pkl --retrain_ratio 0.05
```



### Case Study 2: Loop Vectorization (Section 6.2)

This task constructs a predictive model to determine the optimal Vectorization Factor (VF) and Interleaving Factor (IF) for individual  vectorizable loops in C programs [34, 48]. Following [34], we ex-  plore 35 combinations of VF (1, 2, 4, 8, 16, 32, 64) and IF (1, 2, 4, 8, 16). We initially allocate 80% (4800)  of loop programs for training the model, reserving the remaining 20% (1200) for testing its performance. To introduce data drift, we  use loop programs generated from 14 benchmarks for training  and evaluate the model on the programs from the remaining 4  benchmarks. This ensures that the function and content of test  samples are not encountered during the training phase.

This demo corresponds to Figure 7(b), 8(b), 9(b) and 11(b) of the submitted manuscript.

*approximate runtime = 10 minutes for one benchmark*

```
# Train the underlying model and deploy to drifting environment. The results correspond to Figure 7(b).
python model_definition.py --model trained_model.pkl
python prom_train.py --calibration_data calibration_data.pkl

# Using prom to detect the drifting data. The results correspond to Figure 8(b) and Figure 11(b).
python deploy_model.py --model trained_model.pkl --test_data /path/to/test_data.pkl --non_func all

# Increment learning with detected misprediction samples. The results correspond to Figure 9(b).
python retrain_model.py --model trained_model.pkl --new_data /path/to/new_data.pkl --retrain_ratio 0.05
```

### Case Study 3: Heterogeneous Mapping (Section 6.3)

This task develops a binary classifier to determine if the CPU or  the GPU gives faster performance for an OpenCL kernel. We train and evaluate the baseline  model using 10-fold cross-validation. This involves training a model  on programs from all but one of the sets and then testing it on the  programs from the remaining set. To introduce data drift, we train  the models using 6 benchmark suites and then test the trained  models on the remaining suite. We repeat this process until all  benchmark suites have been tested at least once.

This demo corresponds to Figure 7(c), 8(c), 9(c) and 11(c) of the submitted manuscript.

*approximate runtime = 10 minutes for one benchmark*

```
# Train the underlying model and deploy to drifting environment. The results correspond to Figure 7(c).
python model_definition.py --model trained_model.pkl
python prom_train.py --calibration_data calibration_data.pkl

# Using prom to detect the drifting data. The results correspond to Figure 8(c) and Figure 11(c).
python deploy_model.py --model trained_model.pkl --test_data /path/to/test_data.pkl --non_func all

# Increment learning with detected misprediction samples. The results correspond to Figure 9(c).
python retrain_model.py --model trained_model.pkl --new_data /path/to/new_data.pkl --retrain_ratio 0.05
```

### Case Study 4: Vulnerability Detection (Section 6.4)

This task develops an ML classifier to predict if a given C function  contains a potential code vulnerability.

This demo corresponds to Figure 7(d), 8(d), 9(d) and 11(d) of the submitted manuscript. We consider the top-8 most dangerous types of bugs from the 2023 CWE. As with prior approaches, we initially train the  model on 80% of the randomly selected samples and evaluate its  performance on the remaining 20% samples. Then, we introduce  data drift by training the model on data collected between 2013 and 2020 and testing the trained model on samples collected between 2021 and 2023.

*approximate runtime = 10 minutes for one benchmark*

```
# Train the underlying model and deploy to drifting environment. The results correspond to Figure 7(d).
python model_definition.py --model trained_model.pkl
python prom_train.py --calibration_data calibration_data.pkl

# Using prom to detect the drifting data. The results correspond to Figure 8(d) and Figure 11(d).
python deploy_model.py --model trained_model.pkl --test_data /path/to/test_data.pkl --non_func all

# Increment learning with detected misprediction samples. The results correspond to Figure 9(d).
python retrain_model.py --model trained_model.pkl --new_data /path/to/new_data.pkl --retrain_ratio 0.05
```

### Case Study 5: DNN Code Generation (Section 6.5)

This task builds a regression-based cost model to drive the schedule  search process in TVM for DNN code generation on multi-core CPUs. The cost model estimates the potential gain of a schedule (e.g., instruction orders and data placement) to guide the search. For the baseline, we train and test the cost model on  the BERT-base dataset, where the model is trained on 80% randomly  selected samples and then tested on the remaining 20% samples. To introduce data drift, we tested the trained model on the other  three variants of the BERT model and ResNet-50.

This demo corresponds to Table 2 of the submitted manuscript.

*approximate runtime = 10 minutes for one benchmark*

```
# Train the underlying model and deploy to drifting environment. The results correspond to Figure 7(d).
python model_definition.py --model trained_model.pkl
python prom_train.py --calibration_data calibration_data.pkl

# Using prom to detect the drifting data. The results correspond to Figure 8(d) and Figure 11(d).
python deploy_model.py --model trained_model.pkl --test_data /path/to/test_data.pkl --non_func all

# Increment learning with detected misprediction samples. The results correspond to Figure 9(d).
python retrain_model.py --model trained_model.pkl --new_data /path/to/new_data.pkl --retrain_ratio 0.05
```

### Compare to Other CP-based Methods (Section 7.5)

This experiment compares Prom with RISE, developed for  wireless sensing, and TESSERACT, designed for malware classification.

```
# The results correspond to Figure 10.
sh compare_rise.sh 
sh compare_tesseract.sh
```

### 
