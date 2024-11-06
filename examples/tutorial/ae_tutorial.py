# import subprocess
# subprocess.run(["bash", "ae_tu.sh"], check=True)

# import sys
# print(sys.version)
# print(sys.executable)
# !conda info --env
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

print("_"*30)

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

print("_"*30)

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

print("_"*30)

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

print("_"*30)

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

print("_"*30)
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
