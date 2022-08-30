import glob

from typing import List, Tuple, Union

import json

class Dialogue:
    def __init__(
            self,
            raw_dialogue: List[Tuple[str, str]],
            topic: str,
            emotions: Union[List[int], None] = None
        ):
        self.raw_dialogue = self._preprocess(raw_dialogue)
        if emotions is not None:
            assert len(raw_dialogue) == len(emotions), '{} {}'.format(len(raw_dialogue), len(emotions))
        self.emotions = emotions
        self.topic = topic
    
    def _preprocess(self, raw_dialogue):
        return raw_dialogue

    def split(self):
        context_response_pairs = [{
            'context': [self.raw_dialogue[0], ],
            'response': self.raw_dialogue[1],
            'topic': self.topic
        }]
        if self.emotions is not None:
            context_response_pairs[-1]['emotions'] = self.emotions[1]
        for split_num in range(2, len(self.raw_dialogue)):
            context = context_response_pairs[-1]['context']
            context.append(context_response_pairs[-1]['response'])
            response = self.raw_dialogue[split_num]
            sample = {
                'context': context,
                'response': response,
                'topic': self.topic
            }
            if self.emotions is not None:
                sample['emotions'] = self.emotions[split_num]
            context_response_pairs.append(sample)
        return context_response_pairs

class Parsers:

    @staticmethod
    def get_taskmaster_dialogues(dataset_path: str = './data/model_training/Taskmaster/', debug=False, use_emotions=False,) -> Tuple[List[Dialogue], List[str]]:

        speaker_map = {
            'USER': '[ALICE]',
            'ASSISTANT': '[BOB]',
            '': '[BOB]'
        }

        tm_2 = dataset_path + 'TM-2-2020/data/'
        dialog_files = glob.glob(tm_2 + '*.json')

        tm_1 = dataset_path + 'TM-1-2019/'
        tm_1_files = [tm_1 + 'self-dialogs.json', tm_1 + 'woz-dialogs.json']

        dialog_files.extend(tm_1_files)
        if debug is True:
            dialog_files = [dialog_files[0], dialog_files[1]]
        dialogues = []
        topics = set()
        for dialog_file in dialog_files:
            with open(dialog_file, 'r', encoding='utf-8') as f:
                raw_file = json.load(f)
                for d in raw_file:
                    topic = d['instruction_id']#.split('-')
                    #print(topic)
                    if 'movie-tickets' in topic:
                        topic = 'movie-ticket'

                    if topic == 'music-8-new':
                        topic = 'music'
                    
                    topic = topic.split('-')
                    if len(topic) <= 2:
                        topic = topic[0]
                    else:
                        topic = '-'.join(topic[:-1])
                    topics.add(topic)

                    if len(d['utterances']) > 1:
                        raw_dialogue = [
                            (speaker_map[t['speaker']], t['text']) for t in d['utterances']
                        ]
    
                        dialogues.append(Dialogue(raw_dialogue, topic))
        
        return dialogues, list(topics)

    @staticmethod
    def get_dailydialog_dialogues(
            dataset_path: str = './data/model_training/ijcnlp_dailydialog/',
            use_emotions=False,
            debug=False
        ) -> Tuple[List[Dialogue], List[str], List[List[int]]]:
        # TODO: emotions!
        raw_dialogues_file = dataset_path + 'dialogues_text.txt'
        dialogues_topics = dataset_path + 'dialogues_topic.txt'
        dialogues_emotions = dataset_path + 'dialogues_emotion.txt'

        topics_map = {
            1:  'Ordinary Life',
            2:  'School Life',
            3:  'Culture & Education',
            4:  'Attitude & Emotion',
            5:  'Relationship',
            6:  'Tourism',
            7:  'Health',
            8:  'Work',
            9:  'Politics',
            10: 'Finance'
        }
        
        emotions_map = {
            0: 'no emotion',
            1: 'anger',
            2: 'disgust',
            3: 'fear',
            4: 'happiness',
            5: 'sadness',
            6: 'surprise'
        }
        
        raw_dialogues = []
        with open(raw_dialogues_file, 'r', encoding='utf-8') as f:
            for raw_dialogue in f:
                dialogue = []
                for i, response in enumerate(raw_dialogue.split('__eou__')[:-1]):
                    name = '[ALICE]' if i % 2 == 0 else '[BOB]'
                    dialogue.append((name, response))
                raw_dialogues.append(dialogue)
        topics = []
        with open(dialogues_topics, 'r', encoding='utf-8') as f:
            topics = [topics_map[int(topic)] for topic in f]
        assert len(topics) == len(raw_dialogues)
        emotions = None
        if use_emotions is True:
            emotions = []
            with open(dialogues_emotions, 'r', encoding='utf-8') as f:
                for line in f:
                    emotions.append(list(map(int, line.split())))
            assert len(emotions) == len(raw_dialogues)
            for i in range(len(raw_dialogues)):
                if len(raw_dialogues[i]) != len(emotions[i]):
                    emotions[i].append(0)
            raw_dialogues = [Dialogue(dialogue, topic, em) for (dialogue, topic, em) in list(zip(raw_dialogues, topics, emotions))]
        else:
            raw_dialogues = [Dialogue(dialogue, topic) for (dialogue, topic) in list(zip(raw_dialogues, topics))]
        if use_emotions is True:
            output = [raw_dialogues, [list(set(topics)), emotions_map]]
        else:
            output = [raw_dialogues, list(set(topics))]
        return output
    
    '''
    @staticmethod
    def get_personalized_dialog_dataset_dialogues(dataset_path: str = './data/personalized-dialog-dataset/full/', debug=False) -> Tuple[List[Dialogue], List[str]]:
        all_files = glob.glob(dataset_path + '*.txt')
        all_files = [fl for fl in all_files if 'OOV' not in fl]
        if debug is True:
            all_files = [dataset_path + 'personalized-dialog-task3-options-tst.txt', ]
        
        return None
    '''

    @staticmethod
    def get_personachat_dataset_dialogues(dataset_path: str = './data/model_training/personachat_self_original.json', use_emotions=False, debug=False) -> Tuple[List[Dialogue], List[str]]:
        personachat_file = dataset_path
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
        if debug is True:
            dataset = [dataset['valid'], ]
        else:
            dataset = [dataset['train'], dataset['valid']]
        
        dialogues = []
        for data in dataset:
            for idx in range(len(data)):
                raw_dialogue = data[idx]['utterances'][-1]['history']
                raw_dialogue = [('[ALICE]' if i%2==0 else '[BOB]', dialogue) for i, dialogue in enumerate(raw_dialogue)]
                dialogues.append(Dialogue(raw_dialogue, 'no_topic'))
        return dialogues, ['no_topic', ]

class AttrDatasetParsers:
    
    @staticmethod
    def get_sst_dataset(dataset_path: str = './data/discriminator_finetuning/sentiment/', debug=False):
        samples = []
        files = ['sst_dev.txt', 'sst_test.txt', 'sst_train.txt']
        for fl in files:
            with open(dataset_path + fl, 'r', encoding='utf-8') as f:
                for sample in f:
                    sample = list(sample.split('\t'))
                    sample[0] = int(sample[0].strip('__label__'))
                    samples.append((sample[1], sample[0]))
        return samples
