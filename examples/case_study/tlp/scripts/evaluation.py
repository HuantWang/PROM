import yaml
import subprocess
import time
import json
#  JSON 
def is_experiment_completed(result_json):
    result_json = json.loads(result_json, strict=False)
    #  JSON  Python 
    try:
        experiments = json.loads(result_json)
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return False

    # 
    for experiment in experiments:
        status = experiment.get("status", "")
        if status not in ["SUCCEEDED", "FAILED", "STOPPED"]:
            return False  #  False

    return True  # 

#  nnictl create --config /home/huanting/PROM/examples/case_study/Thread/config.yaml --port 8088
yaml_file = "/home/huanting/PROM/examples/case_study/tlp/scripts/config.yaml"

# 
experiments = [
# {"command": "python train_tlp.py --mode train --save_folder='models/train/tlp_i7_base' --under_train_dataset='./data_model/bert_base_train_and_val.pkl' --under_test_dataset='./data_model/bert_base_test.pkl' ", "port": 8088},
# {"command": "python train_tlp.py --mode train --save_folder='models/train/tlp_i7_large' --under_train_dataset='./data_model/bert_large_train_and_val.pkl' --under_test_dataset='./data_model/bert_large_test.pkl' ", "port": 8089},
# {"command": "python train_tlp.py --mode train --save_folder='models/train/tlp_i7_tiny' --under_train_dataset='./data_model/bert_tiny_train_and_val.pkl' --under_test_dataset='./data_model/bert_tiny_test.pkl' ", "port": 8090},
# {"command": "python train_tlp.py --mode train --save_folder='models/train/tlp_i7_med' --under_train_dataset='./data_model/bert_medium_train_and_val.pkl' --under_test_dataset='./data_model/bert_medium_test.pkl' ", "port": 8091},
# {"command": "python train_tlp.py --mode deploy --save_folder='models/il/tlp_i7_tiny' --under_model='./models/train/tlp_i7_base/tlp_model_533_best.pkl' --test_data='./data_model/bert_tiny_test.pkl' --path='((bert_tiny*.task.pkl'", "port": 8092},
{"command": "python train_tlp.py --mode deploy --save_folder='models/il/tlp_i7_med' --under_model='./models/tlp_model_base_best.pkl' --test_data='./data_model/bert_medium_test.pkl' --path='((bert_medium*.task.pkl'", "port": 8093},
{"command": "python train_tlp.py --mode deploy --save_folder='models/il/tlp_i7_large' --under_model='./models/tlp_model_base_best.pkl' --test_data='./data_model/bert_large_test.pkl' --path='((bert_large*.task.pkl'", "port": 8094},

]

subprocess.run(["nnictl", "stop", "-a"], capture_output=True, text=True)
time.sleep(10)

for exp in experiments:
    #  YAML 
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    #  command  port
    config['trial']['command'] = exp['command']

    #  YAML 
    with open(yaml_file, 'w') as file:
        yaml.dump(config, file)

    #  NNI 
    subprocess.Popen(["nnictl", "create", "--config", yaml_file, "--port", str(exp['port'])])



    print(f"Waiting for experiment to complete...")
    time.sleep(60*60)

# subprocess.run(["nnictl", "stop", "-a"], capture_output=True, text=True)
