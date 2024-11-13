#  Enhancing Deployment-time Predictive Model Robustness for Code Analysis and Optimization: Artifact Instructions for Docker Image
## Preliminaries

This document provides the evaluation of case studies reported in the paper.

The main results of our CGO 2025 paper apply Prom to 5 case studies to detect their drifting 
samples.  


The following step-by-step instructions are provided for using a Docker Image running on a local host.


*Disclaimer: Note that during our testing, we found that the underlying 
device (CPU and GPU model) can affect the performance of the evaluation. 
If your CPU or GPU differs from the setup described in our paper, 
the experimental results may be impacted. Alternatively, you can use our 
[online notebook](http://34.66.10.35:8099/tree/examples/tutorial) 
for evaluation (the password is .*9mYUc?2isT_&Zcy), 
which is configured to match the device setup in our paper.
Refer to 'Appendix: Artifacts Evaluation Instructions' in [our paper](./docs/CGO_25_AE.pdf) 
for the instructions in Section A.2: Interactive Notebooks.
*

## Links to The Paper

For each step, we note the section number of the submitted version where the relevant technique is described or data is presented.

The main results are presented in Figures 7-10 and Table 2 and 3 of the submitted paper.

The following step-by-step instructions are provided for using a Docker Image 
running on a local host.

*Disclaim: Although we have worked hard to ensure that our 
codes are robust, our tool remains a \*research prototype\*. 
It may still have glitches when used in complex, real-life settings. 
If you discover any bugs, please raise an issue, describing how you 
ran the program and the problem you encountered. 
We will get back to you ASAP. Thank you.*

# Step-by-Step Instructions 


## ★ Docker Image

We prepare our artifact within a Docker image to run "out of the box". 
Our docker image was tested on a host machine running Ubuntu 18.04.

## ★ Artifact Evaluation  

Follow the instructions below to use our AE evaluation scripts.

### 1. Setup

Install Docker by following the instructions 
[here](https://docs.docker.com/install/linux/docker-ce/ubuntu/). 
The following instructions assume the host OS runs Linux.

#### 1.1  Fetch the Docker Image

Fetch the docker image from docker hub.

```
$ sudo docker pull wanghuanting/prom:0.1
```

To check the list of images, run:

```
$ sudo docker images
#output
#REPOSITORY                                                               TAG                                 IMAGE ID       CREATED         SIZE
#wanghuanting/prom                                                        0.1                                cc84e8929fe1   2 minutes ago    147GB

```

Run the Docker container

```
$ sudo docker run -it --name prom -p 8099:8099  wanghuanting/prom:0.1 /bin/bash
```


#### 1.2 Setup the Environment

After importing the docker container **and getting into bash** in the container, run the following command to select the conda environment, before using any of the AE scripts:

`````` shell
$ conda activate thread
``````

Then, go to the directory of our tool:

```
# Move the project to the target directory
(thread) $ mv /cgo/PROM/prom/* /cgo/prom/
(thread) $ cd prom/PROM/examples/tutorial/
```


# Demo 1: Tutorial for Prom

This demo corresponds to the simplified drifting detection example shown in Figure 2. 
Note that the code has been refactored, resulting in minor changes to the API. 
This small-scale demo represents case study 1 on thread coarsening. 

This project trains an ML model to predict the optimal OpenCL GPU thread coarsening 
factor (1–32) for performance, using cross-validation on OpenCL kernels across multiple 
suites and GPUs. Prom, an anomaly detector, identifies performance issues, adapts to data drift, 
and boosts deployment performance through incremental learning.

```
# Demo 1: Tutorial for Prom
python ae_tutorial.py
```


# Demo 2: Experimental Evaluation

Here, we provide the evaluation to showcase the working 
mechanism of the Prom on five case studies.

### Case Study 1: Thread Coarsening (Section 6.1)

This problem develops a model to determine the optimal OpenCL GPU thread coarsening factor for performance optimization. Following other works, an ML model predicts a coarsening factor (ranging  from 1 to 32) for a test OpenCL kernel, where 1 indicates no coarsening. Underlying models. We train the baseline model using  leave-one-out cross-validation, which involves training the base-  line model on 16 OpenCL kernels and testing on another one. We  then repeat this process until all benchmark suites have been tested  once. To introduce data drift, we train the ML models on OpenCL  benchmarks from two suites and then test the trained model on  another left-out benchmark suite.

This demo corresponds to Figure 7(a), 8(a), 9(a), 11(a) of the submitted manuscript.

*approximate runtime = 10 minutes for one benchmark*

```
python ae_thread.py
```

## Full-scale evaluation data

We now plot the diagrams using full-scale evaluation data. 
The results correspond to Figure 7(b), 8(b), 9(b) and 11(b) of the submitted manuscript.

```
"""
Figures will be saved to: /cgo/prom/PROM/examples/tutorial/figures_plot/figure/
thread.pdf                  -> Figure 7(a)
detectdrifting_thread.pdf   -> Figure 8(a)
thread_il.pdf               -> Figure 9(a)
individual_thread.pdf       -> Figure 11(a)
"""
python ae_plot.py --case thread
```

### Case Study 2: Loop Vectorization (Section 6.2)

This task constructs a predictive model to determine the optimal Vectorization Factor (VF) and Interleaving Factor (IF) for individual  vectorizable loops in C programs [34, 48]. Following [34], we ex-  plore 35 combinations of VF (1, 2, 4, 8, 16, 32, 64) and IF (1, 2, 4, 8, 16). We initially allocate 80% (4800)  of loop programs for training the model, reserving the remaining 20% (1200) for testing its performance. To introduce data drift, we  use loop programs generated from 14 benchmarks for training  and evaluate the model on the programs from the remaining 4  benchmarks. This ensures that the function and content of test  samples are not encountered during the training phase.

This demo corresponds to Figure 7(b), 8(b), 9(b) and 11(b) of the submitted manuscript.

*approximate runtime = 10 minutes for one benchmark*


```
python ae_loop.py
```


We now plot the diagrams using full-scale evaluation data. 
The results correspond to Figure 7(b), 8(b), 9(b) and 11(b) of the submitted manuscript.

```
"""
Figures will be saved to: /cgo/prom/PROM/examples/tutorial/figures_plot/figure/
loop.pdf                  -> Figure 7(b)
detectdrifting_loop.pdf   -> Figure 8(b)
loop_il.pdf               -> Figure 9(b)
individual_loop.pdf       -> Figure 11(b)
"""
python ae_plot.py --case loop
```

### Case Study 3: Heterogeneous Mapping (Section 6.3)

This task develops a binary classifier to determine if the CPU or  the GPU gives faster performance for an OpenCL kernel. We train and evaluate the baseline  model using 10-fold cross-validation. This involves training a model  on programs from all but one of the sets and then testing it on the  programs from the remaining set. To introduce data drift, we train  the models using 6 benchmark suites and then test the trained  models on the remaining suite. We repeat this process until all  benchmark suites have been tested at least once.

This demo corresponds to Figure 7(c), 8(c), 9(c) and 11(c) of the submitted manuscript.

*approximate runtime = 10 minutes for one benchmark*

```
python ae_dev_docker.py
```


We now plot the diagrams using full-scale evaluation data. 
The results correspond to Figure 7(c), 8(c), 9(c) and 11(c) of the submitted manuscript.

```
"""
Figures will be saved to: /cgo/prom/PROM/examples/tutorial/figures_plot/figure/
dev.pdf                  -> Figure 7(c)
detectdrifting_dev.pdf   -> Figure 8(c)
dev_il.pdf               -> Figure 9(c)
individual_device.pdf    -> Figure 11(c)
"""
python ae_plot.py --case dev
```

### Case Study 4: Vulnerability Detection (Section 6.4)

This task develops an ML classifier to predict if a given C function  contains a potential code vulnerability.

This demo corresponds to Figure 7(d), 8(d), 9(d) and 11(d) of the submitted manuscript. We consider the top-8 most dangerous types of bugs from the 2023 CWE. As with prior approaches, we initially train the  model on 80% of the randomly selected samples and evaluate its  performance on the remaining 20% samples. Then, we introduce  data drift by training the model on data collected between 2013 and 2020 and testing the trained model on samples collected between 2021 and 2023.

*approximate runtime = 10 minutes for one benchmark*

```
python ae_vul.py --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../benchmark/Bug/train.jsonl     --eval_data_file=../../benchmark/Bug/valid.jsonl     --test_data_file=../../benchmark/Bug/test.jsonl --evaluate_during_training
```


We now plot the diagrams using full-scale evaluation data. 
The results correspond to Figure 7(d), 8(d), 9(d) and 11(d) of the submitted manuscript.

```
"""
Figures will be saved to: /cgo/prom/PROM/examples/tutorial/figures_plot/figure/
drifting_vul.pdf         -> Figure 7(d)
detectdrifting_vul.pdf   -> Figure 8(d)
IL_vul.pdf               -> Figure 9(d)
individual_vul.pdf       -> Figure 11(d)
"""
python ae_plot.py --case vul
```

### Case Study 5: DNN Code Generation (Section 6.5)

This task builds a regression-based cost model to drive the schedule  search process in TVM for DNN code generation on multi-core CPUs. The cost model estimates the potential gain of a schedule (e.g., instruction orders and data placement) to guide the search. For the baseline, we train and test the cost model on  the BERT-base dataset, where the model is trained on 80% randomly  selected samples and then tested on the remaining 20% samples. To introduce data drift, we tested the trained model on the other  three variants of the BERT model and ResNet-50.

This demo corresponds to Table 2 of the submitted manuscript.

*approximate runtime = 10 minutes for one benchmark*

```
bash ae_tlp.sh
```


We now plot the diagrams using full-scale evaluation data. 
The results correspond to Table 3 and Figure 8(e) of the submitted manuscript.

```
"""
Figures will be saved to: /cgo/prom/PROM/examples/tutorial/figures_plot/figure/
detectdrifting_tensor.pdf         -> Figure 8(e)
Print(table)                      -> Table 3.
"""
python ae_plot.py --case tlp
```


We now plot the summary table. 
The results correspond to Table 2 of the submitted manuscript.
```
python ae_plot.py --case all
```
### Compare to Other CP-based Methods (Section 7.5)

This experiment compares Prom with conformal prediction-based methods like MAPIE and PUNCC, as well as RISE, developed for wireless sensing, and TESSERACT, designed for malware classification.
```
# The results correspond to Figure 10.
bash ae_compare.sh
```


We now plot the diagrams using full-scale evaluation data. 
The results correspond to Figure 10 of the submitted manuscript.

```
"""
Figures will be saved to: /cgo/prom/PROM/examples/tutorial/figures_plot/figure/
sig.pdf         -> Figure 10
"""
python ae_plot.py --case compare
```

##  Further Analysis (Optional)

This section presents an analysis of certain parameters, corresponding to Section 7.6.

```
"""
Figures will be saved to: /cgo/prom/PROM/examples/tutorial/figures_plot/figure/
gaussian.pdf         -> Figure 13 (a)
"""
python ae_plot.py --case gaussian
```

```
bash ae_cd.sh
```

```
"""
Figures will be saved to: /cgo/prom/PROM/examples/tutorial/figures_plot/figure/
coverage.pdf         -> Figure 13 (b)
"""
python ae_plot.py --case cd
```

```
# clean all temporary files
bash ae_clean.sh
```

Now, you can check the figures in the following directory:
```
cd /cgo/prom/PROM/examples/tutorial/figures_plot/figure/
```