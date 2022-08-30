import random

import transformers

from src.dataset_parsers import Dialogue, Parsers
from src.dataset import DialogueDataset, get_dataloaders

class TestDataset:
    def test_dataset(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'additional_special_tokens': ['[CONTEXT]', '[RESPONSE]', '[ALICE]', '[BOB]']})
        d = DialogueDataset(tokenizer, 'gpt2', debug=False)
        
        assert len(d.dialogues) == 935487

    def test_gpt2_output(self):
        random.seed(42)
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'additional_special_tokens': ['[CONTEXT]', '[RESPONSE]', '[ALICE]', '[BOB]']})

        dataset = DialogueDataset(tokenizer=tokenizer,
                                  model_type='gpt2',
                                  mode=None,
                                  use_topic=False,
                                  use_sequence_bucketing=False,
                                  debug=True)
                                
        sample = dataset[0]
        
        assert set(['real_context_len', 'context_tt_ids', 'response_tt_ids', 'context_ids', 'response_ids']) == set(sample.keys())

    def test_bart_output(self):
        random.seed(42)
        tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-base')
        tokenizer.add_special_tokens({'additional_special_tokens': ['[CONTEXT]', '[RESPONSE]', '[ALICE]', '[BOB]']})

        dataset = DialogueDataset(tokenizer=tokenizer,
                                  model_type='bart',
                                  mode=None,
                                  use_topic=False,
                                  use_sequence_bucketing=False,
                                  debug=True)
                                
        sample = dataset[0]
        print(sample.keys())
        assert set(['context_ids', 'response_ids']) == set(sample.keys())
    
    def test_split(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-base')
        tokenizer.add_special_tokens({'additional_special_tokens': ['[CONTEXT]', '[RESPONSE]', '[ALICE]', '[BOB]']})

        dataset = DialogueDataset(tokenizer=tokenizer,
                                  model_type='bart',
                                  mode='train',
                                  use_topic=False,
                                  use_sequence_bucketing=False,
                                  debug=False)
        ds_len = len(dataset)
        dataset = DialogueDataset(tokenizer=tokenizer,
                                  model_type='bart',
                                  mode='valid',
                                  use_topic=False,
                                  use_sequence_bucketing=False,
                                  debug=False)
        val_len = len(dataset)
        assert (ds_len == 935487) and (val_len == 9449)

    def test_dataloader_gpt2(self):
        random.seed(42)
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'additional_special_tokens': ['[CONTEXT]', '[RESPONSE]', '[ALICE]', '[BOB]']})

        train, val = get_dataloaders(tokenizer, 'gpt2', 6, 3)

        train_batch = next(iter(train))

        print(train_batch.keys())
        for key in train_batch:
            print(train_batch[key].shape)
        
        val_batch = next(iter(val))

        print(val_batch.keys())
        for key in val_batch:
            print(val_batch[key].shape)

        assert set(['context_ids', 'context_pos_ids', 'context_tt_ids', 'real_context_len', 'attention_mask', 'lm_indices', 'cls_head_labels']) == set(train_batch.keys()) and set(['context_ids', 'context_pos_ids', 'context_tt_ids', 'real_context_len', 'attention_mask', 'lm_indices', 'cls_head_labels']) == set(val_batch.keys())
        
    def test_dd_emotions(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-base')
        tokenizer.add_special_tokens({'additional_special_tokens': ['[CONTEXT]', '[RESPONSE]', '[ALICE]', '[BOB]']})

        dataset = DialogueDataset(tokenizer=tokenizer,
                                  model_type='bart',
                                  mode='train',
                                  use_topic=False,
                                  use_sequence_bucketing=False,
                                  debug=False,
                                  use_taskmaster=False,
                                  use_dailydialog=True,
                                  use_personachat=False,
                                  use_dailydialog_emotions=True
                                )
        sample = dataset[0]

        print(sample.keys())

        print(len(dataset))

        assert set(['context_ids', 'response_ids', 'label']) == set(sample.keys()) and len(dataset) == 88964

    def test_topics_without_dailydialog(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-base')
        tokenizer.add_special_tokens({'additional_special_tokens': ['[CONTEXT]', '[RESPONSE]', '[ALICE]', '[BOB]']})

        dataset = DialogueDataset(tokenizer=tokenizer,
                                  model_type='bart',
                                  mode='train',
                                  use_topic=True,
                                  use_sequence_bucketing=False,
                                  debug=False,
                                  use_taskmaster=True,
                                  use_dailydialog=True,
                                  use_personachat=False,
                                  use_dailydialog_emotions=False
                                )
        sample = dataset[0]

        print(sample.keys())

        print(len(dataset))

        assert set(['context_ids', 'response_ids', 'label']) == set(sample.keys()) and len(dataset) == 697172

    