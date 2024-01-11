import yaml


def get_config():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    return config
