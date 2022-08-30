import pytorch_lightning as pl
import torch

from torch.nn import functional as F
from pytorch_lightning.metrics.nlp import BLEUScore

from src.model.tools import build_model, get_tools, build_full_pipeline
from src.utils.decode_strategies import TopPSampling
from src.utils.lightning_metrics import Perplexity, RougeMetric


def shift1_and_pad(ids, pad_elem):
    target_ids = torch.clone(ids[:, 1:])
    suffix = torch.empty(len(ids), 1).long().fill_(pad_elem).type_as(ids)
    target_ids = torch.cat([target_ids, suffix], dim=1)
    return target_ids

class LightningDoubleHeadsModel(pl.LightningModule):
    def __init__(self, args):
        super(LightningDoubleHeadsModel, self).__init__()
        self.args = args
        self.model, self.tokenizer, self.special_ids = build_model(args.model, args.use_cls_head)
        self.optimizer, self.scheduler, self.criterions = get_tools(
            self.model, args.mode, args.lr, args.warmup, args.weight_decay,
            args.label_smoothing, args.ul_alpha, self.special_ids.pad)
        
        device = next(self.model.parameters()).device
        # needed for inference during validation
        self.generator = TopPSampling(self.model, self.model.special_ids, device, args['datasets'].response_max_len, top_p=args.top_p)
        
        # metrics
        self.ppl = Perplexity(self.model.special_ids.pad)
        self.bleu_2 = BLEUScore(n_gram=2)
        self.bleu_4 = BLEUScore(n_gram=4)
        self.rouge = RougeMetric()

    def forward(self, batch):
        return self.model(**batch)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def get_target_ids(self, batch):
        if self.model.model_type == 'gpt2':
            return shift1_and_pad(batch['context_ids'], self.model.special_ids.pad)
        
        return shift1_and_pad(batch['response_ids'], self.model.special_ids.pad)

    def processing_before_loss(self, batch, logprobs, target_ids):
        """Select needed logprobs and targets for loss computation"""
        # use only positive samples for language modeling
        logprobs = logprobs.index_select(0, batch['lm_indices'])
        target_ids = target_ids.index_select(0, batch['lm_indices'])
        
        if self.model.model_type == 'gpt2':
            # use only response tokens for training
            tt_ids = batch['context_tt_ids'].index_select(0, batch['lm_indices'])
            response_positions = (tt_ids == self.model.special_ids.response)
            # 1D tensor after masked selection
            return (logprobs[response_positions],
                    target_ids[response_positions],
                    response_positions)
        
        return (logprobs.view(-1, logprobs.size(-1)), 
                target_ids.view(-1),
                None)

   
    def training_step(self, batch, batch_nb):
        total_loss = 0
        logs = {}
        target_ids = self.get_target_ids(batch)
        
        logprobs, cls_logits = self(batch)
        logprobs, target_ids, _ = self.processing_before_loss(batch, logprobs, target_ids)

        lm_loss = [
            criter(logprobs, target_ids) 
            for criter in self.criterions
        ]
        total_loss += sum(lm_loss)

        logs['train_smooth_loss'] = lm_loss[0]
        if len(lm_loss) == 2:
            logs['train_un_loss'] = lm_loss[1]
        
        if cls_logits is not None:
            cls_loss = F.cross_entropy(cls_logits, batch['cls_head_labels'])
            total_loss += cls_loss
        
            logs['train_cls_loss'] = cls_loss
        
        logs['train_total_loss'] = total_loss

        return {
            'loss': total_loss,
            'log': logs
        }

    def validation_step(self, batch, batch_nb):
        total_loss = 0
        logs = {}
        target_ids = self.get_target_ids(batch)

        last_hidden, *_ = self.model.get_hidden_and_past(**batch)
        
        # logits are needed for perplexity calculation
        lm_logits = self.model.lm_head.proj(last_hidden)
        logprobs = self.model.lm_head(lm_logits, is_logits=True)
        logprobs, target_ids, response_positions = self.processing_before_loss(batch, logprobs, target_ids)
        
        lm_loss = [
            criter(logprobs, target_ids)
            for criter in self.criterions
        ]
        total_loss += sum(lm_loss)
        
        logs['val_smooth_loss'] = lm_loss[0]
        if len(lm_loss) == 2:
            logs['val_un_loss'] = lm_loss[1]
        
        if hasattr(self.model, 'classification_head'):
            if self.model.model_type == 'gpt2':
                input_ids = batch['context_ids']
            else:
                input_ids = batch['response_ids']
            cls_logits = self.model.classification_head(last_hidden, input_ids)
            cls_loss = F.cross_entropy(cls_logits, batch['cls_head_labels'])
            total_loss += cls_loss
            
            logs['val_cls_loss'] = cls_loss

        logs['val_total_loss'] = total_loss
        
        # model perplexity
        lm_logits = lm_logits.index_select(0, batch['lm_indices'])[response_positions]
        logs['perplexity'] = self.ppl(
            lm_logits.view(-1, lm_logits.size(-1)),
            target_ids.view(-1)
        )
        
        # calculate BLEU metric
        pred_responses, target_responses = self.inference_on_batch(batch)
        logs['bleu_2'] = self.bleu_2(pred_responses, target_responses)
        logs['bleu_4'] = self.bleu_4(pred_responses, target_responses)
        
        # calculate ROUGE metric
        pred_responses = [self.model.tokenizer.convert_tokens_to_string(pred_)
                          for pred_ in pred_responses]
        target_responses = [self.model.tokenizer.convert_tokens_to_string(target_[0])
                            for target_ in target_responses]
        rouge = self.rouge(pred_responses, target_responses)

        return {
            'val_loss': total_loss,
            **logs,
            **rouge
        }

    def validation_epoch_end(self, outputs):
        logs = {}
        for key in outputs[0].keys():
            logs[key] = torch.stack([x[key] for x in outputs]).mean()

        return {'val_loss': logs['val_loss'], 'log': logs}

    def inference_on_batch(self, batch):
        def mask_and_split(inp, mask, split_len):
            inp = inp.masked_select(mask)
            return inp[:split_len], inp[split_len:]

        pred_responses = []
        target_responses = []
        lm_indices = batch['lm_indices'].tolist()
        for indx in lm_indices:
            cntx_ids = batch['context_ids'][indx]
            attn_mask = batch['attention_mask'][indx].bool()
            if self.model.model_type == 'gpt2':
                cntx_tt_ids = batch['context_tt_ids'][indx]
                cntx_pos_ids = batch['context_pos_ids'][indx]
                cntx_len = batch['real_context_len'][indx].item()
                cntx_ids, response_ids = mask_and_split(cntx_ids, attn_mask, cntx_len)
                cntx_tt_ids, response_tt_ids = mask_and_split(cntx_tt_ids, attn_mask, cntx_len)
                cntx_pos_ids, _ = mask_and_split(cntx_pos_ids, attn_mask, cntx_len)
            else:
                response_ids = batch['response_ids'][indx]
                cntx_tt_ids = cntx_pos_ids = None

            self.generator.initialize(cntx_ids, cntx_tt_ids, cntx_pos_ids)
            out_ids = self.generator.run()
            pred_responses.append(
                self.model.tokenizer.convert_ids_to_tokens(out_ids,
                                                           skip_special_tokens=True)
            )
            target_responses.append(
                [self.model.tokenizer.convert_ids_to_tokens(response_ids,
                                                            skip_special_tokens=True)]
            )

        return pred_responses, target_responses


class LightningAttributeModel(pl.LightningModule):
    def __init__(self, args):
        super(LightningAttributeModel, self).__init__()
        self.args = args
        assert args.model == 'gpt2'
        self.language_model, self.attr_model, self.tokenizer, self.special_ids = build_full_pipeline(args, args.model, args.name)
        self.optimizer, self.scheduler, _ = get_tools(
            self.attr_model, args.mode, args.lr, args.warmup, args.weight_decay,
            args.label_smoothing, 0, self.special_ids.pad)
        

    def forward(self, batch):
        hidden, _, _ = self.language_model.get_hidden_and_past(
            **batch
        )
        outputs = self.attr_model.discriminator[self.args.name](hidden)
        return outputs
    
    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        classes = batch['label']
        batch = {key:val for key,val in batch.items() if key != 'label'}
        outputs = self(batch)
        loss = self.attr_model.discriminator[self.args.name].get_loss(outputs, classes)
        return {
            'loss': loss,
            'log': {
                'train_loss': loss
            }
        }

    #def validation_step(self, batch, batch_nb):
    #    pass

    #def validation_epoch_end(self, outputs):
    #    pass