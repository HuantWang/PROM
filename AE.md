#  Enhancing Deployment-time Predictive Model Robustness for Code Analysis and Optimization: Artifact Instructions for Docker Image
## Preliminaries

This documents provides the evaluation of case studies reported in the paper.

The main results of our CGO 2025 paper apply Prom to 5 case studies to detect their drifting samples. The evaluation presented in our paper ran on a much larger dataset and for longer. This notebook contains minimal working examples designed to be evaluated in a reasonable amount of time (approximately 20 minutes).

The following step-by-step instructions are provided for using a Docker Image running on a local host.

*Disclaim: Although we have worked hard to ensure that our codes are robust, our tool remains a \*research prototype\*. It may still have glitches when used in complex, real-life settings. If you discover any bugs, please raise an issue, describing how you ran the program and the problem you encountered. We will get back to you ASAP. Thank you.*

## Links to The Paper

For each step, we note the section number of the submitted version where the relevant technique is described or data is presented.

The main results are presented in Figures 7-10 and Table 2 and 3 of the submitted paper.

The following step-by-step instructions are provided for using a Docker Image running on a local host.

*Disclaim: Although we have worked hard to ensure that our codes are robust, our tool remains a \*research prototype\*. It may still have glitches when used in complex, real-life settings. If you discover any bugs, please raise an issue, describing how you ran the program and the problem you encountered. We will get back to you ASAP. Thank you.*

# Step-by-Step Instructions 


## ★ Docker Image

We prepare our artifact within a Docker image to run "out of the box". 
Our docker image was tested on a host machine running Ubuntu 18.04.

## ★ Artifact Evaluation  

Follow the instructions below to use our AE evaluation scripts.

### 1. Setup

Install Docker by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/). The following instructions assume the host OS runs Linux.

#### 1.1  Fetch the Docker Image

Fetch the docker image from docker hub.

```
$ sudo docker pull wanghuanting/prom:0.3
```

To check the list of images, run:

```
$ sudo docker images
#output
#REPOSITORY                                                               TAG                                 IMAGE ID       CREATED         SIZE
#wanghuanting/prom                                                        0.3                                cc84e8929fe1   2 minutes ago    102GB

```

Run the docker image in a GPU-enabled environment

```
$ sudo docker run -it --name prom -p 8099:8099  wanghuanting/prom:0.3 /bin/bash
$ docker start prom 
$ docker exec -it prom /bin/bash 
```


#### 1.2 Setup the Environment

After importing the docker container **and getting into bash** in the container, run the following command to select the conda environment, before using any of the AE scripts:

`````` shell
$ conda activate thread
``````

Then, go to the root directory of our tool:

```
(thread) $ cd /cgo/prom/PROM/examples/tutorial
```


# Demo 1: Tutorial for Prom

This demo corresponds to the simplified drifting detection example given in Figure 2. Note that we have refactored the code; hence there are small changes in the API. This is a small-scale demo for case study 1 of thread coarsening. The full-scale evaluation used in the paper takes over 24 hours to run.

```
# Demo 1: Tutorial for Prom
python ae_tutorial.py
```
## Step 1. Train the underlying model

This problem develops a model to determine the optimal OpenCL GPU thread coarsening factor for performance optimization. Following the original paper, an ML model predicts a coarsening factor (ranging  from 1 to 32) for a test OpenCL kernel, where 1 indicates no coarsening. Following the setup of  the DeepTune, we train and test the models using the labeled dataset  from them, comprising 17 OpenCL kernels from three benchmark  suites across four GPU platforms.

we train the baseline model using  leave-one-out cross-validation, which involves training the baseline model on 16 OpenCL kernels and testing on another one.
#### Training dataset and calibration dataset partitioning

```
!bash ae_tu.sh
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append('/cgo/prom/PROM')
sys.path.append('../case_study/Thread/')
sys.path.append('/cgo/prom/PROM/src')
sys.path.append('/cgo/prom/PROM/thirdpackage')
from Magni_utils import ThreadCoarseningMa,Magni,make_predictionMa,make_prediction_ilMa
from Thread_magni import load_magni_args_notebook,load_magni_args
from src.prom.prom_util import Prom_utils

print("Starting to partition the training and calibration datasets...")

# Load necessary arguments for the program, related to model configuration or runtime settings
# args = load_magni_args_notebook()
args = load_magni_args()
seed_value = int(args.seed)
# Initialize a thread coarsening model object for the Magni model
prom_thread = ThreadCoarseningMa(model=Magni())

# Define the path to the dataset and the target platform (here, "Tahiti" refer to a GPU architecture or hardware)
dataset_path = "../../benchmark/Thread"
platform = "Tahiti"

# Perform data partitioning for training, validation, and testing.
# The method 'data_partitioning' returns the split data, including features (X) and labels (y)
# Additionally, it returns calibration data and indices for each partition (train, validation, and test).
# Args could contain hyperparameters or settings used during data partitioning (such as shuffle, split ratios, etc.)
X_cc, y_cc,train_x, valid_x, test_x, train_y, valid_y, test_y, \
        calibration_data,cal_y,train_index, valid_index, test_index,y,X_seq,y_1hot=\
            prom_thread.data_partitioning(dataset_path,platform=platform,mode='train', calibration_ratio=0.1,args=args)

# 'X_cc', 'y_cc' : The coarsened features and labels (thread coarsening context)
# 'train_x', 'valid_x', 'test_x' : Training, validation, and testing features (input data)
# 'train_y', 'valid_y', 'test_y' : Training, validation, and testing labels (output data)
# 'calibration_data', 'cal_y' : Calibration data and labels, for model tuning or confidence calibration
# 'train_index', 'valid_index', 'test_index' : Indexes for the split datasets (training, validation, test)
# 'y', 'X_seq', 'y_1hot' : Other data representations (e.g., one-hot encoding, sequence features/labels)
print("Data splitting process completed!")
```

#### Training the underlying model

```
print("Starting the training process for the underlying model...")

# Initialize the model with the provided arguments
# The 'args' contain configurations or hyperparameters needed to set up the model (e.g., learning rate, batch size, etc.)
prom_thread.model.init(args)

# Train the model using the training data (features and labels)
# 'cascading_features' refers to the input features for training (possibly related to thread coarsening features)
# 'cascading_y' is the corresponding target labels for the training data
# 'verbose=True' enables the printing of training progress or detailed output during the training process
prom_thread.model.train(
    cascading_features=train_x,
    verbose=True,
    cascading_y=train_y)

# Initialize empty lists to store results
# 'origin_speedup_all' will hold the original speedup values
# 'speed_up_all' will store the predicted speedup values from the model
# 'improved_spp_all' will track any improvements in speedup across predictions
origin_speedup_all = []
speed_up_all = []
improved_spp_all = []

# Call 'make_predictionMa' to make predictions using the trained model
# Arguments:
#   - 'speed_up_all': A list to collect the speedup predictions from the model
#   - 'platform': The target platform (e.g., "Tahiti") for which the prediction is made
#   - 'model': The trained model (prom_thread.model)
#   - 'test_x': Features for the test data
#   - 'test_index': Indices of the test data
#   - 'X_cc': Additional features (possibly coarsened or cascading features)
origin_speedup, all_pre, data_distri = make_predictionMa(speed_up_all=speed_up_all,
                                                         platform=platform,
                                                         model=prom_thread.model,
                                                         test_x=test_x,
                                                         test_index=test_index,
                                                         X_cc=X_cc)

# Print the original speedup obtained from the prediction for the specified platform
print(f"The speedup on the Titan is {origin_speedup:.2%}")
```

#### Training the anomaly detector

```
print("Starting the construction of the anomaly detector...")

# Set the trained classifier model from prom_thread
clf = prom_thread.model

# Define the prom parameters (method_params) for different evaluation metrics:
method_params = {
    "lac": ("score", True),
    "top_k": ("top_k", True),
    "aps": ("cumulated_score", True),
    "raps": ("raps", True)
}

# Initialize a Prom object for performing conformal prediction and evaluation
# 'task' is set to "thread" indicating the specific task being modeled
Prom_thread = Prom_utils(clf, method_params, task="thread")

# Perform conformal prediction:
#   - 'cal_x' and 'cal_y' are the calibration data and labels,
#   - 'test_x' and 'test_y' are the test data and labels,
#   - 'significance_level' is set to "auto" for automatic adjustment.
y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
    cal_x=calibration_data, 
    cal_y=cal_y, 
    test_x=test_x,
    test_y=test_y,
    significance_level="auto"
)
print("The anomaly detector has been successfully constructed.")
```



# Train the prom anolamy detector

## Step 2. Prom on deployment

First, to introduce data drift, we train the ML models on OpenCL benchmarks from two suites and then test the trained model on another left-out benchmark suite.


#### Native deployment

```
# Load the pre-trained model for predictions
prom_thread.model.restore(r'../../examples/case_study/Thread/ae_savedmodels/tutorial/Tahiti-underlying.model')

# Make predictions on the given data
origin_speedup, all_pre, data_distri = make_predictionMa(
    speed_up_all=speed_up_all,     # Array or list of speed-up values for the model
    platform=platform,             # Target platform for prediction (e.g., CPU, GPU)
    model=prom_thread.model,       # Loaded model used for making predictions
    test_x=test_x,                 # Test features or data input for predictions
    test_index=test_index,         # Indices for test data to match predicted values
    X_cc=X_cc                      # Contextual data or additional features for prediction
)

# Print a success message with the deployment speedup result
print(f"Loading successful, the deployment speedup on the {platform} is {origin_speedup:.2%}")

# Calculate the average original speedup from all speed-up values
origin = sum(speed_up_all) / len(speed_up_all)
```

#### Detecting drifting samples

```
# Perform evaluation of the conformal prediction model, retrieving accuracy and other metrics:
#   - 'index_all_right' stores all indices where predictions are correct,
#   - 'index_list_right' stores lists of indices for each correctly predicted instance,
#   - 'Acc_all' holds the overall accuracy of predictions,
#   - 'F1_all' stores the F1-score for the predictions,
#   - 'Pre_all' holds the precision metric for predictions,
#   - 'Rec_all' contains the recall metric for the predictions.
index_all_right, index_list_right, Acc_all, F1_all, Pre_all, Rec_all, _, _ = \
    Prom_thread.evaluate_conformal_prediction(
        y_preds=y_preds,             # Predicted labels or outcomes
        y_pss=y_pss,                 # Prediction scores or probabilities associated with predictions
        p_value=p_value,             # p-value threshold for conformal prediction
        all_pre=all_pre,             # Array of all prediction values
        y=y[test_index],             # True labels or outcomes for test data
        significance_level=0.05      # Significance level for conformal prediction intervals
    )
```



## Step 3. Improve Deployment Time Performance

Prom can enhance the performance of deployed ML systems through incremental learning.

#### Retraining the model with incremental learning:

```
# Perform incremental learning by selecting the most valuable instances to add to the training set:
#   - 'seed_value' is used for randomization,
#   - 'test_index' and 'train_index' are updated accordingly.
print("Finding the most valuable instances for incremental learning...")
train_index, test_index = Prom_thread.incremental_learning(
    seed_value, 
    test_index, 
    train_index
)
# Fine-tune the model using the updated training set:
#   - 'X_seq[train_index]' is the updated training data,
#   - 'y_1hot[train_index]' are the corresponding labels in one-hot encoding format.
print(f"Retraining the model on {platform}...")
prom_thread.model.fine_tune(
    cascading_features=X_seq[train_index],
    cascading_y=y_1hot[train_index],
    verbose=True
)

# Test the fine-tuned model and make predictions using the updated model:
#   - 'speed_up_all' stores the predicted speedup values,
#   - 'improved_spp_all' stores improvements in speedup,
#   - 'origin_speedup' refers to the initial speedup before fine-tuning.
retrained_speedup, inproved_speedup, data_distri = make_prediction_ilMa(
    speed_up_all=speed_up_all, 
    platform=platform,
    model=prom_thread.model,
    test_x=X_seq[test_index],
    test_index=test_index, 
    X_cc=X_cc,
    origin_speedup=origin_speedup,
    improved_spp_all=improved_spp_all
)

origin_speedup_all.append(origin_speedup)
speed_up_all.append(retrained_speedup)
improved_spp_all.append(inproved_speedup)

mean_acc = sum(Acc_all) / len(Acc_all)
mean_f1 = sum(F1_all) / len(F1_all)
mean_pre = sum(Pre_all) / len(Pre_all)
mean_rec = sum(Rec_all) / len(Rec_all)

# Calculate the improved mean speed-up (improved_spp_all is a list of improved speed-ups)
meanimproved_speed_up = sum(improved_spp_all) / len(improved_spp_all)

# Output the model's average accuracy, precision, recall, and F1 score, formatted as percentages
print(
    "Detection Performance Metrics:\n"
    f"  - Average Accuracy  : {mean_acc * 100:.2f}%\n"
    f"  - Average Precision : {mean_pre * 100:.2f}%\n"
    f"  - Average Recall    : {mean_rec * 100:.2f}%\n"
    f"  - Average F1 Score  : {mean_f1 * 100:.2f}%"
)


# Output the final improved speed-up as a percentage
print(f"Final improved speed-up percentage: {meanimproved_speed_up * 100:.2f}%")
```



# Demo 2: Experimental Evaluation

Here, we provide a small-sized evaluation to showcase the working mechanism of the Prom on five case studies. A full-scale evaluation, which takes more than a day to run, is provided through the Docker image (with detailed instructions on our project Github).

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

## Full-scale evaluation data

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

## Full-scale evaluation data

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

## Full-scale evaluation data

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

## Full-scale evaluation data

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

## Full-scale evaluation data

We now plot the diagrams using full-scale evaluation data. 
The results correspond to Figure 10 of the submitted manuscript.

```
python ae_plot.py --case compare
```

##  Further Analysis (Optional)

This section presents an analysis of certain parameters, corresponding to Section 7.6.%run ae_plot.py --case tlp

```
# The results correspond to Figure 13(a).
python ae_plot.py --case gaussian
```

```
# The results correspond to Figure 13(b).
bash ae_cd.sh
```
```
%run ae_plot.py --case cd
```