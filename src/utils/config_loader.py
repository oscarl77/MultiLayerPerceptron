import json

def load_config(path="./config.json"):
    with open(path, "r") as f:
        return json.load(f)

def update_config(key, value, path="./config.json"):
    """Adds a new entry into the config file. In the context of this project
    it saves an experiment's weight vector and bias term.
    """
    config = load_config()
    config[key] = value
    with open(path, "w") as f:
        json.dump(config, f, indent=4)