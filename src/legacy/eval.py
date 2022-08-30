import torch
import numpy as np
from collections import deque
from tqdm.auto import tqdm
from utils.metrics import get_rouges, get_recall, get_scores, get_confusion, get_precision, get_f, apply_threshold, \
    ROUGE_TYPES
from utils.utils import to_numpy
import config as cfg
import wandb
from sklearn.metrics import confusion_matrix

def eval_discriminator(gpt, model, dataloaders, device, best_score=None, epoch=0, eval_type="test", save_head=True):
    print(f'===> Evaluating discriminator on {eval_type} set')
    assert eval_type == 'test' or eval_type == 'train', 'No matching evaluation for this type of dataloading!'
    heads = model.discriminator.keys()
    if best_score is None:
        best_score = {head: 0 for head in heads}
    print(best_score)
    for head in heads:
        print(best_score[head])

        if model.discriminator[head].use == 1:
            model, best_score[head] = eval_head(gpt, model, head, dataloaders[head], device, \
                                                epoch=epoch, eval_type=eval_type, best_score=best_score[head], save_head=save_head)

    return model, best_score

def eval_head(gpt, model, head, dataloader, device, epoch=0, eval_type="test", best_score=None, save_head=True):
    print(f'===> Evaluating discriminator using {eval_type} dataloader | head {head}')

    test_loss = 0
    tp = torch.zeros(model.discriminator[head].config['num_classes'], device=device, dtype=torch.int32)
    fp = torch.zeros(model.discriminator[head].config['num_classes'], device=device, dtype=torch.int32)
    tn = torch.zeros(model.discriminator[head].config['num_classes'], device=device, dtype=torch.int32)
    fn = torch.zeros(model.discriminator[head].config['num_classes'], device=device, dtype=torch.int32)

    epoch_probs = []
    epoch_pred_classes = []
    model.eval()

    def eval_on_batch(ids, tt_ids, mask, pos_ids):
        pred_responses = []
        pred_labels = []
        pred_probs = []
        for id, tt_id, mask_id, pos_id in zip(ids, tt_ids, mask, pos_ids):
            hidden = gpt.get_hidden_state(id.unsqueeze(0), tt_id.unsqueeze(0), mask_id.unsqueeze(0), pos_id.unsqueeze(0))
            responce = model.discriminator[head](hidden)
            pred_responses.append(responce)  # raw output of model last layer
            pred_labels.append(model.discriminator[head].get_labels(responce)) # get probs, then apply treshold
            pred_probs.append(model.discriminator[head].get_probs(responce))
        return torch.cat(pred_responses), torch.cat(pred_probs), torch.cat(pred_labels)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)

    for batch_id, batch in pbar:
        with torch.no_grad():
            if eval_type == 'test':
                ids, tt_ids, classes, mask, position_ids = [
                    [x.to(device) for x in part] for part in batch]
                outputs, probs, preds = eval_on_batch(ids, tt_ids, mask, position_ids)
                classes = torch.cat(classes)
            if eval_type == 'train':
                ids, tt_ids, classes, mask, position_ids = \
                    [x.to(device) for x in batch]
                hidden = gpt.get_hidden_state(ids, tt_ids, mask, position_ids)
                outputs = model.discriminator[head](hidden)
                preds = model.discriminator[head].get_labels(outputs)
                probs = model.discriminator[head].get_probs(outputs)
        #epoch_probs.append(probs)

        if model.discriminator[head].config['classification_type'] == 'multilabel':
            classes = classes.to(torch.float32)
        else:
            classes = classes.to(torch.long)
        current_loss = model.discriminator[head].loss(outputs, classes).item()
        test_loss += current_loss
        pbar.set_description(f'Loss: {current_loss :.5} ')
        #epoch_pred_classes.append(classes)
        #print(preds)
        #print(classes)
        temp_tp, temp_tn, temp_fp, temp_fn = get_confusion(preds, classes, device=device,
                                                           labels=list(model.discriminator[head].config['classes'].values()))
        tp += temp_tp
        #print(temp_tp)
        #print(tp)
        tn += temp_tn
        fp += temp_fp
        fn += temp_fn
        #print(tp, fp, tn, fn)
    test_loss /= len(dataloader.dataset)
    wandb.log({f"Eval loss ({eval_type}) | {head}": test_loss}, step=epoch)
    print(f'Mean test loss: {test_loss}')
    #score = get_scores(torch.cat(epoch_probs), torch.cat(epoch_pred_classes), average=model.discriminator[head].config['average'],
    #                   multi_class=model.discriminator[head].config['multi_class'])
    precision = float(to_numpy(get_precision(tp, tn, fp).mean()))
    f1 = get_f(tp, tn, fp, fn)
    recall = float(to_numpy(get_recall(tp, fn).mean()))
    print(f'F1:{f1}, precision {precision}, recall {recall}')
    #wandb.log({f"Epoch_{eval_type}_score_{head}": score}, step=epoch)
    wandb.log({f"Epoch {eval_type} f1 | {head}": f1}, step=epoch)
    wandb.log({f"Epoch {eval_type} precision | {head}": precision}, step=epoch)
    wandb.log({f"Epoch {eval_type} recall | {head}": recall}, step=epoch)
    if f1 > best_score:
        if save_head is True:
            model.discriminator[head].save()
        best_score = f1
        print(f'Best score ({eval_type}): {best_score}, epoch: {epoch} | head {head}')
    return model, best_score



def evaluale(model, dataloader, device, generator, save_filename, last_rouge=float('-inf'), max_iter=-1):
    print('===> Evaluating generator')
    def eval_on_batch(ids, tt_ids, mask, pos_ids):
        pred_responses = []
        for inputs in zip(ids, tt_ids, mask, pos_ids):
            generator.initialize(*inputs)
            out_ids = generator.run()
            response = model.tokenizer.decode(out_ids, skip_special_tokens=True)
            pred_responses.append(response)
        return pred_responses

    rouge_mean = 0
    model.eval()

    pbar = tqdm(dataloader, total=len(dataloader), leave=False)
    #batch is set of 6 lists with ids, token_type_ids, mask, position_ids, gold_ids, gold_str
    i = 0
    for batch in pbar:
        if i == max_iter:
            break
        ids, token_type_ids, mask, position_ids = [
            [x.to(device) for x in part] for part in batch[:4]]
        gold_ids, gold_str = batch[4:]

        with torch.no_grad():
            # get responce ids
            pred_responses = eval_on_batch(ids, token_type_ids, mask, position_ids)
        
        current_rouges = get_rouges(pred_responses, gold_str)
        rouge_mean += current_rouges[-1]

        pbar.set_description(' '.join(
            [
                f'{rouge_type}:{val/len(ids):.2f} ' for rouge_type, val
                in zip(ROUGE_TYPES, current_rouges)
            ]
        ))
        i += 1

        '''for rouge_type, val in zip(ROUGE_TYPES, current_rouges / len(input_ids)):
            writer.add_scalar(f'Eval/{rouge_type}', val, writer.eval_step)
        writer.eval_step += 1'''   
    
    rouge_mean /= len(dataloader) if i != -1 else max_iter
    print(f'Mean rouge is: {rouge_mean}')
    if rouge_mean > last_rouge:
        last_rouge = rouge_mean
        model.save(save_filename)

    return model, last_rouge

if __name__ == "__main__":
    """
    Some test
    """
    import numpy as np

    pred_responces = ['I have a nice cat', 'language models are trained to generate text', 'cv is dope']
    gold_responces = ['I have a cat', 'language models are trained to generate text seq-to-seq', 'cv models are dope']
    rouges = get_rouges(pred_responces, gold_responces)
    print('We have rouges: ')
    print(rouges)
