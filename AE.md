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
for evaluation, which is configured to match the device setup in our paper.*

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
$ sudo docker pull wanghuanting/prom:0.7
```

To check the list of images, run:

```
$ sudo docker images
#output
#REPOSITORY                                                               TAG                                 IMAGE ID       CREATED         SIZE
#wanghuanting/prom                                                        0.7                                cc84e8929fe1   2 minutes ago    101GB

```

Run the docker image in a GPU-enabled environment

```
$ sudo docker run -it --name prom -p 8099:8099  wanghuanting/prom:0.7 /bin/bash
```
$ docker start prom 
$ docker exec -it prom /bin/bash 

#### 1.2 Setup the Environment

After importing the docker container **and getting into bash** in the container, run the following command to select the conda environment, before using any of the AE scripts:

`````` shell
$ conda activate thread
``````

Then, go to the root directory of our tool:

```
# Move the project to the target directory
(thread) $ mv /cgo/PROM/prom/* /cgo/prom/
(thread) $ cd prom/PROM/examples/tutorial/
```


# Demo 1: Tutorial for Prom

This demo corresponds to the simplified drifting detection example shown in Figure 2. Note that the code has been refactored, resulting in minor changes to the API. This small-scale demo represents case study 1 on thread coarsening. 


```
# Demo 1: Tutorial for Prom
python ae_tutorial.py
```
## Step 1. Train the underlying model

This problem involves developing a model to determine the optimal OpenCL GPU thread coarsening factor for performance optimization. Following the original paper, an ML model predicts a coarsening factor (ranging from 1 to 32) for a test OpenCL kernel, where 1 indicates no coarsening. Following the setup of DeepTune, we train and test the models on their labeled dataset, which includes 17 OpenCL kernels from three benchmark suites across four GPU platforms (this minimal working example runs on the Titan platform).

We train the baseline model using leave-one-out cross-validation, where the model is trained on 16 OpenCL kernels and tested on the remaining one.
#### Training dataset and calibration dataset partitioning

```
!bash ae_tu.sh
# import sys
# print(sys.version)
# print(sys.executable)
# !conda info --env

import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append('/cgo/prom/PROM')
sys.path.append('../case_study/Thread/')
sys.path.append('/cgo/prom/PROM/src')
sys.path.append('/cgo/prom/PROM/thirdpackage')

from Magni_utils import ThreadCoarseningMa, Magni, make_predictionMa, make_prediction_ilMa
from Thread_magni import load_magni_args_notebook, load_magni_args
from src.prom.prom_util import Prom_utils

print("Starting to partition the training and calibration datasets...")

# Load necessary arguments for the program, related to model configuration or runtime settings
args = load_magni_args()
seed_value = int(args.seed)

# Initialize a thread coarsening model using the approach in Magni et al
prom_thread = ThreadCoarseningMa(model=Magni())

# Define the path to the dataset and the target platform (here, "Tahiti" refers to a GPU architecture or hardware)
dataset_path = "../../benchmark/Thread"
platform = "Tahiti"

# Perform data partitioning for training, validation, and testing
# returns the split data, including features (X) and labels (y)
# returns calibration data and indices for each partition (train, validation, and test).
# args can contain data partitioning hyperparameters (such as shuffle, split ratios, etc.)

split_data = prom_thread.data_partitioning(
    dataset_path, platform=platform, mode='train', calibration_ratio=0.1, args=args
)
(
    X_cc, y_cc,               # The coarsened features and labels (thread coarsening context)
    train_x, valid_x, test_x, # Training, validation, and testing features (input data)
    train_y, valid_y, test_y, # Training, validation, and testing labels (output data)
    calib_data, calib_y,      # Calibration data and labels, for model tuning or confidence calibration
    train_index, valid_index, test_index, # Indexes for the split datasets (training, validation, test)
    y, X_seq, y_1hot          # Other data representations (e.g., one-hot encoding, sequence features/labels)
) = split_data

print("Data splitting process completed!")
```

#### Training the underlying model

```
print("Starting the training process for the underlying model...")

# Initialize the model
# 'args' can contain model hyperparameters (e.g., learning rate, batch size, etc.)
prom_thread.model.init(args)

# Train the model using training data (features and labels)
prom_thread.model.train(
    cascading_features=train_x, # train data features (possibly related to thread coarsening features)
    verbose=True,               # detailed training process output including progress
    cascading_y=train_y         # train data labels
)

origin_speedup_all = [] # original speedup values
speed_up_all = []       # predicted speedup values
improved_spp_all = []   # speedup improvement values (original speedup - retrained speedup)

# Make predictions using the trained model
origin_speedup, all_pre, data_distri = make_predictionMa(
    speed_up_all=speed_up_all,    # speedup predictions from the model (list)
    platform=platform,            # target platform (e.g., "Tahiti") used for prediction
    model=prom_thread.model,      # trained model
    test_x=test_x,                # test data features
    test_index=test_index,        # test data indices
    X_cc=X_cc                     # additional features (possibly coarsened or cascading features)
)

print(f"Thread coarsening speedup on platform '{platform}' is {origin_speedup:.2%}")
```

#### Training the anomaly detector

```
print("Starting the construction of the anomaly detector...")

# trained classifier model
clf = prom_thread.model

# prom parameters (method_params) for different evaluation metrics
method_params = {
    "lac": ("score", True),
    "top_k": ("top_k", True),
    "aps": ("cumulated_score", True),
    "raps": ("raps", True)
}

# Prom object for performing conformal prediction and evaluation
# 'task' : set to "thread" to model thread coarsening task
Prom_thread = Prom_utils(clf, method_params, task="thread")

# Perform conformal prediction
y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
    cal_x=calib_data, cal_y=calib_y, # calibration data and labels
    test_x=test_x, test_y=test_y,    # test data and labels
    significance_level="auto"        # set to "auto" for automatic adjustment
)
print("The anomaly detector has been successfully constructed.")
```



# Train the Prom anolamy detector

## Step 2. Prom on deployment

First, to introduce data drift, we train the ML models on OpenCL benchmarks from two suites and then test the trained model on another left-out benchmark suite.

#### Native deployment



```
# Load pre-trained model for predictions
prom_thread.model.restore(r'../../examples/case_study/Thread/ae_savedmodels/tutorial/Tahiti-underlying.model')

# Make predictions on test data
origin_speedup, all_pre, data_distri = make_predictionMa(
    speed_up_all=speed_up_all,    # Array or list of speed-up values for the model
    platform=platform,            # Target platform for prediction (e.g., CPU, GPU)
    model=prom_thread.model,      # Loaded model used for making predictions
    test_x=test_x,                # Test features or data input for predictions
    test_index=test_index,        # Indices for test data to match predicted values
    X_cc=X_cc                     # Contextual data or additional features for prediction
)

# Print a success message with the deployment speedup result
print(f"Deployment speedup on platform '{platform}' is {origin_speedup:.2%}")

# Calculate average from all speed-up values
origin = sum(speed_up_all) / len(speed_up_all)
```

#### Detecting drifting samples

```
# retrieves evaluation metrics for conformal prediction model
(
    index_ncm_correct, # each row contains indices flagged correctly by that ncm function
    index_all_correct, # each element contains indices flagged correctly by all ncm functions
    Acc_all, F1_all, Pre_all, Rec_all, # accuracy, f1 score, precision and recall
    *_
) = Prom_thread.evaluate_conformal_prediction(
    y_preds=y_preds,             # Predicted labels
    y_pss=y_pss,                 # Prediction scores or probabilities
    p_value=p_value,             # p-value threshold for conformal prediction
    all_pre=all_pre,             # all prediction values
    y=y[test_index],             # True labels or outcomes for test data
    significance_level=0.05      # Significance level for conformal prediction intervals
)
```



## Step 3. Improve Deployment Time Performance

Prom can enhance the performance of deployed ML systems through incremental learning.

#### Retraining the model with incremental learning:

```
# most valuable test instances are moved to the training set
print("Finding the most valuable instances for incremental learning...")
train_index, test_index = Prom_thread.incremental_learning(
    seed_value,
    test_index,
    train_index
)
# Fine-tune the model with updated training set
print(f"Retraining the model on platform '{platform}'...")
prom_thread.model.fine_tune(
    cascading_features=X_seq[train_index], # updated training features
    cascading_y=y_1hot[train_index], # updated one-hot encoded training labels
    verbose=True
)

# Make predictions using the updated model
retrained_speedup, improved_speedup, y_preds = make_prediction_ilMa(
    speed_up_all=speed_up_all,        # predicted speedup values
    platform=platform,
    model=prom_thread.model,
    test_x=X_seq[test_index],
    test_index=test_index,
    X_cc=X_cc,
    origin_speedup=origin_speedup,    # speedup before fine-tuning
    improved_spp_all=improved_spp_all # speedup improvements
)

origin_speedup_all.append(origin_speedup)
speed_up_all.append(retrained_speedup)
improved_spp_all.append(improved_speedup)

mean_acc = sum(Acc_all) / len(Acc_all)
mean_f1 = sum(F1_all) / len(F1_all)
mean_pre = sum(Pre_all) / len(Pre_all)
mean_rec = sum(Rec_all) / len(Rec_all)

# Calculate the average speedup improvement
mean_improved_speed_up = sum(improved_spp_all) / len(improved_spp_all)

# Output the model's average accuracy, precision, recall, and F1 score, formatted as percentages
print(
    "Detection Performance Metrics:\n"
    f"  - Average Accuracy  : {mean_acc * 100:.2f}%\n"
    f"  - Average Precision : {mean_pre * 100:.2f}%\n"
    f"  - Average Recall    : {mean_rec * 100:.2f}%\n"
    f"  - Average F1 Score  : {mean_f1 * 100:.2f}%"
)


# Output the final improved speedup as a percentage
print(f"Final improved speedup percentage: {mean_improved_speed_up * 100:.2f}%")
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
python ae_plot.py --case tlp
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
python ae_plot.py --case compare
```

##  Further Analysis (Optional)

This section presents an analysis of certain parameters, corresponding to Section 7.6.

```
# The results correspond to Figure 13(a).
python ae_plot.py --case gaussian
```

```
# The results correspond to Figure 13(b).
bash ae_cd.sh
```

```
python ae_plot.py --case cd
```

Now, you can check the figures in the following directory:
```
cd /cgo/prom/PROM/examples/tutorial/figures_plot/figure
```