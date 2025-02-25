from pprint import pprint
import yaml

# Updated import for PascalCase LoaderSetup
from utils.LoaderSetup import join_constructor

with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

pprint(config)
