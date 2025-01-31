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
yaml_file = "/cgo/prom/PROM/examples/case_study/BugD/config.yaml"

# 
experiments = [
    # {"command": "python VD_codebert.py --mode train --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../../benchmark/Bug/train.jsonl     --eval_data_file=../../../benchmark/Bug/valid.jsonl     --test_data_file=../../../benchmark/Bug/test.jsonl --evaluate_during_training", "port": 8089},
    # {"command": "python VD_codebert.py --mode deploy --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../../benchmark/Bug/train.jsonl     --eval_data_file=../../../benchmark/Bug/valid.jsonl     --test_data_file=../../../benchmark/Bug/test.jsonl --evaluate_during_training", "port": 8090},
    {"command": "python VD_vulde.py --mode train --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../../benchmark/Bug/train.jsonl     --eval_data_file=../../../benchmark/Bug/valid.jsonl     --test_data_file=../../../benchmark/Bug/test.jsonl --evaluate_during_training", "port": 8091},
    {"command": "python VD_vulde.py --mode deploy --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../../benchmark/Bug/train.jsonl     --eval_data_file=../../../benchmark/Bug/valid.jsonl     --test_data_file=../../../benchmark/Bug/test.jsonl --evaluate_during_training", "port": 8092},
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
    time.sleep(5*60*60)

# subprocess.run(["nnictl", "stop", "-a"], capture_output=True, text=True)
