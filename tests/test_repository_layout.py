import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def require_external_artifact(path, description):
    if not path.exists():
        raise unittest.SkipTest(
            f"{description} is an external artifact and is not available at {path}"
        )


def test_local_runtime_and_generated_roots_are_ignored():
    ignored = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {".venv/", ".deps/", ".pip-cache/", ".pytest_cache/", ".tmp/", ".wandb/", "outputs/"} <= ignored


class RepositoryLayoutTest(unittest.TestCase):
    def test_reference_weather_lives_outside_package_code(self):
        weather_file = ROOT / "data" / "processed" / "amsterdam_reference" / "weather" / "Amsterdam" / "2010.csv"

        require_external_artifact(weather_file, "Amsterdam reference weather dataset")
        self.assertTrue(weather_file.exists())

    def test_default_env_config_uses_reference_dataset_path(self):
        config_path = ROOT / "configs" / "envs" / "GreenLightEnv.yml"

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        weather_dir = config["GreenLightEnv"]["weather_repository_kwargs"]["weather_data_dir"]
        self.assertEqual(weather_dir, "data/processed/amsterdam_reference/weather/")

    def test_chengdu_dataset_template_exists(self):
        dataset_config = ROOT / "configs" / "datasets" / "chengdu_agri_greenhouse_001.yml"
        metadata = ROOT / "data" / "raw" / "chengdu_agri" / "greenhouse_001" / "metadata.yml"
        env_config = ROOT / "configs" / "envs" / "ChengduSingleGreenhouseEnv.yml"

        self.assertTrue(dataset_config.exists())
        self.assertTrue(env_config.exists())
        require_external_artifact(metadata, "Chengdu greenhouse raw metadata")
        self.assertTrue(metadata.exists())

    def test_chengdu_env_config_uses_chengdu_processed_weather(self):
        config_path = ROOT / "configs" / "envs" / "ChengduSingleGreenhouseEnv.yml"

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        weather_dir = config["GreenLightEnv"]["weather_repository_kwargs"]["weather_data_dir"]
        scenario = config["GreenLightEnv"]["weather_scenario_sampler_kwargs"]
        self.assertEqual(weather_dir, "data/processed/chengdu_agri/greenhouse_001/weather/")
        self.assertEqual(scenario["location"], "Chengdu")

    def test_experiment_output_paths_are_dataset_scoped_and_ignored(self):
        config_path = ROOT / "configs" / "datasets" / "amsterdam_reference.yml"

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        ignored = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())

        self.assertEqual(config["outputs"]["results_root"], "results/amsterdam_reference/")
        self.assertEqual(config["outputs"]["train_data_root"], "train_data/amsterdam_reference/")
        self.assertIn("results/", ignored)
        self.assertIn("train_data/", ignored)


if __name__ == "__main__":
    unittest.main()
