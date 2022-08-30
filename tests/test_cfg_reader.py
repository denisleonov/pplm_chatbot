import omegaconf

from src.utils.utils import read_chatbot_configs, read_heads_only_configs

class TestReader:
    def test_read_cfg(self):
        path = 'gpt2_baseline.yaml'
        cfg = read_chatbot_configs(path)
        assert cfg['datasets']['response_max_len'] == 75

    def test_heads_only_reader(self):
        cfg = read_heads_only_configs()

        assert 'sentiment' in cfg
