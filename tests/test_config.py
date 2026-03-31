from src.utils.config import load_yaml_config


def test_load_yaml_config_reads_mapping() -> None:
    config = load_yaml_config("configs/data.yaml")

    assert config["dataset"]["name"] == "bdd100k"
