from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_value_
import torch
from eval import eval_head


def train_discriminator(gpt, model, dataloaders, optimizer, scheduler,
                                                        device, epoch=0, best_score=None, valid_interval=3):

    print(f'=====> Train epoch {epoch}')
    heads = model.active_heads
    model.train()
    for head in heads:
            model, optimizer, scheduler = train_head(gpt, model, head, dataloaders[head], optimizer, scheduler,
                                                        device, epoch=epoch, best_score=best_score[head], valid_interval=valid_interval)
    return model, optimizer, scheduler

def train_head(gpt, model, head, dataloader, optimizer, scheduler,
                                                        device,  epoch=0, best_score=None, valid_interval=3):

    print(f'=====> Train epoch {epoch} | head {head}')
    loss_val = 0
    losses = []
    loss = 0
    model.train()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for idx_batch, batch in pbar:
        # preprocess batch
        ids, tt_ids, classes, mask_ids, position_ids = \
            [x.to(device) for x in batch]

        # forward pass
        hidden = gpt.get_hidden_state(
            ids, tt_ids, mask_ids, position_ids,
        )
        outputs = model.discriminator[head](hidden)
        #logprobs = logprobs[head]
        loss = model.discriminator[head].get_loss(outputs, classes)
        loss.backward()

        # record a loss value)
        loss_val += loss.item()
        pbar.set_description(f"loss:{loss.item():.2f}")
        #clip_grad_value_(model.parameters(), optimizer.clip_value)
        optimizer.step()
        optimizer.zero_grad()

    print(f"mean loss: {loss_val / len(dataloader.dataset):.4f}")
    wandb.log({f'Train loss | {head} ': loss_val/len(dataloader.dataset)}, step=epoch)
    print('memory allocated: ', torch.cuda.memory_allocated(),
          'memory cached', torch.cuda.memory_cached())
    print('ok')
    if epoch % valid_interval == 0:
        model, score = eval_head(gpt, model, head, dataloader, device, epoch, eval_type='train', save_head=False, best_score=best_score)

    return model, optimizer, scheduler

def train(model, dataloader, optimizer, scheduler, criterions, device, epoch):
    print('===> Training generator')
    loss_val = 0
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for idx_batch, batch in pbar:

        # preprocess batch
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        if model.model_name == 'gpt2':
            target_ids = shift1_and_pad(batch['context_ids'], model.special_ids.pad)
        else:
            target_ids = shift1_and_pad(batch['response_ids'], model.special_ids.pad)
        
        # forward pass
        logprobs = model(
            **batch
        )
        
        # loss
        loss = sum( 
            criter(
                logprobs.view(-1, logprobs.shape[-1]),
                target_ids.view(-1)
            ) for criter in criterions
        )
        loss.backward()
        
        # record a loss value
        loss_val += loss.item() * len(batch['context_ids'])
        pbar.set_description(f"loss:{loss.item():.2f}")

        # make a gradient step
        if (idx_batch + 1) % optimizer.accumulation_interval == 0 or (idx_batch + 1) == len(dataloader):
            clip_grad_value_(model.parameters(), optimizer.clip_value)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

    # overall loss per epoch
    print(f"mean loss: {loss_val / len(dataloader):.4f}")
    print('memory allocated: ', torch.cuda.memory_allocated(), 
          'memory cached', torch.cuda.memory_cached())

    # save model, just in case
    model.save('temp_weight')

    return model, optimizer, scheduler

if __name__ == '__main__':
    pass