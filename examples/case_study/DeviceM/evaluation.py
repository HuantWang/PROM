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

#  nnictl create --config /cgo/prom/PROM/examples/case_study/Thread/config.yaml --port 8088
yaml_file = "/cgo/prom/PROM/examples/case_study/DeviceM/config.yml"

# 
experiments = [
    # {"command": "python DevM_Deeptune.py --mode train --method Deeptune", "port": 8094},
    # {"command": "python DevM_Deeptune.py --mode deploy --method Deeptune", "port": 8095},
    # {"command": "python DevM_Programl.py --mode train --method Programl", "port": 8096},
    {"command": "python DevM_Programl.py --mode deploy --method Programl", "port": 8090},
    # {"command": "python DevM_i2v.py --mode train", "port": 8098},
    # {"command": "python DevM_i2v.py --mode deploy", "port": 8099},
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
    time.sleep(5*60*60)

# subprocess.run(["nnictl", "stop", "-a"], capture_output=True, text=True)
