import torch
import transformers
from src.model.custom_gpt2 import CustomMaskGPT2Model

class TestCustomGPT2:
    def test_seq2seq_mask(self):
        model = CustomMaskGPT2Model.from_pretrained('gpt2')
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        data = [tokenizer.encode_plus('i love all cats{}'.format(tokenizer.eos_token), return_tensors='pt'), tokenizer.encode_plus('he love all dogs{}'.format(tokenizer.eos_token), return_tensors='pt')]
        
        data[0]['attention_mask'][:, -1] = 0
        data[1]['attention_mask'][:, -1] = 0
        res = {}
        for s in data:
            for key in s:
                if key in res:
                    res[key] = torch.cat([res[key], s[key]], dim=0)
                else:
                    res[key] = s[key]
        
        res['use_seq2seq_mask'] = torch.LongTensor([1, 0])
        res['segment_len'] = torch.LongTensor([3, 0])

        outputs = model(**res)

        assert len(outputs) == 2 and outputs[0].shape == torch.Size([2, 5, 768])