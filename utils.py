import yaml


def get_config():
    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    return config
