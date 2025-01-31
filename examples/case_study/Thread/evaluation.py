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
yaml_file = "/home/huanting/PROM/examples/case_study/Thread/config.yaml"

# 
experiments = [
    # {"command": "python Thread_i2v.py --mode train", "port": 8088},
    {"command": "python Thread_i2v.py --mode deploy", "port": 8089},
    {"command": "python Thread_Deep.py --mode train", "port": 8090},
    {"command": "python Thread_Deep.py --mode deploy", "port": 8091},
    {"command": "python Thread_magni.py --mode train", "port": 8092},
    {"command": "python Thread_magni.py --mode deploy", "port": 8093},
]
subprocess.run(["nnictl", "stop", "-a"], capture_output=True, text=True)

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
    time.sleep(1*60*60)

# subprocess.run(["nnictl", "stop", "-a"], capture_output=True, text=True)
