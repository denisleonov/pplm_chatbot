import pytest
from src.dataset_parsers import Dialogue, Parsers, AttrDatasetParsers
from src.dataset import DialogueDataset

class TestParsers:
    def test_taskmaster_parser(self):
        dialogues, topics = Parsers.get_taskmaster_dialogues(debug=False)
        print(topics)
        print(dialogues[0].raw_dialogue)
        assert (len(dialogues) == 30498) and (len(topics) == 22)

    def test_dailydialog_parser(self):
        dialogues, topics = Parsers.get_dailydialog_dialogues()

        assert (len(dialogues) == 13118) and (len(topics) == 10)

    def test_personachat_parser(self):
        dialogues, topics = Parsers.get_personachat_dataset_dialogues()

        assert len(dialogues) == 18878

    def test_personachat_split(self):
        dialogues, topics = Parsers.get_personachat_dataset_dialogues()
        splitted = []
        for d in dialogues:
            splitted.extend(d.split())

        assert len(splitted) == 240722

    def test_dailydialog_split(self):
        dialogues, topics = Parsers.get_dailydialog_dialogues()
        splitted = []
        for d in dialogues:
            splitted.extend(d.split())

        assert len(splitted) == 89862

    def test_taskmaster_split(self):
        dialogues, topics = Parsers.get_taskmaster_dialogues()
        splitted = []
        for d in dialogues:
            splitted.extend(d.split())

        assert len(splitted) == 614352

    def test_sst_parser(self):
        samples = AttrDatasetParsers.get_sst_dataset()

        assert len(samples) == 11855

    def test_emotions_dialydialog(self):
        dialogues, second_param = Parsers.get_dailydialog_dialogues(use_emotions=True)
        topics, emotions_map = second_param
        splitted = []
        for d in dialogues:
            splitted.extend(d.split())
        for i in range(len(splitted)):
            assert 'emotions' in splitted[i]

        assert len(splitted) == 89862
