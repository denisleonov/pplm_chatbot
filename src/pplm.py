import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.model.tools import build_attrmodel_from_model
from src.utils.bow import get_bag_of_words_indices, build_bows_one_hot_vectors
from src.utils.decode_strategies import TopPSampling
from src.utils.ngram_elim import BatchNgramEliminator, DummyNgramEliminator

SMALL_CONST = 1e-15

# TODO: add top-k sampler and ability to select sampler
class PPLMWithTopP(TopPSampling):
    # TODO: add in-time head (de)activation during conversation, loss initialisation for new heads if needed.
    def __init__(self, attr_args, *args, **kwargs):
        self.args = attr_args
        self.labels = kwargs.pop('labels')
        self.heads = kwargs.pop('heads')
        self.topics = kwargs.pop('topics')

        self.temperature = kwargs.pop('temperature')
        self.window_length = kwargs.pop('window_length')
        self.iterations = kwargs.pop('iterations')
        self.horizon_len = kwargs.pop('horizon_length')
        self.stepsize = kwargs.pop('stepsize')
        self.kl_scale = kwargs.pop('kl_scale')
        self.gamma = kwargs.pop('gamma')
        self.fusion_scale = kwargs.pop('fusion_scale')
        self.load_model = False
        
        super().__init__(*args, **kwargs)
        
        self.attr_model = build_attrmodel_from_model(self.model, self.args, device=self.device)
        self.set_attributes(self.topics, self.heads, self.labels, full_init=True)
        if self.heads:
            if self.load_model:
                self.attr_model.load()
            self.attr_model.eval()

        self.past = None

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        # initialize past (for both models) and encoder outputs (only for BART)
        _, past, self.encoder_outputs = self.model.get_hidden_and_past(
            context_ids=self.context_ids,
            context_tt_ids=self.context_tt_ids,
            context_pos_ids=self.context_pos_ids,
        )
        
        if self.past is None:
            self.past = past
            self.grad_norms = None
        else:
            # save recomputed past for history without last utterance
            # it keeps pplm loss more stable
            prev_seq_len = self.past.size(-2)
            past[..., :prev_seq_len, :] = self.past
            self.past = past

    def set_attributes(self, topics=None, heads=None, labels=None, full_init=False):
        # TODO: check head2class correction before add/ remove heads
        if topics:
            self.topics = topics
            self.attr_model.initialize_bow(topics)
        
        if heads:
            assert labels is not None \
                   and len(heads) == len(labels), \
                   "You must provide on class per head"
            
            self.heads = heads
            self.labels = labels
            self.attr_model.initialize_discriminator(heads, full_init)
            self.head2label = {
                h: l
                for h in self.attr_model.active_heads
                for l in labels
                if l in self.attr_model.discriminator[h].config['classes']
            }
    
    def get_model_probs(self):
        '''
        Override parent (see: SamplingStrategy class) method for specific work with the model during PPLM
        '''
        unpert_last_hidden, unpert_past, _ = \
                self.model.get_hidden_and_past(
                    context_ids=self.context_ids,
                    context_tt_ids=self.context_tt_ids,
                    context_pos_ids=self.context_pos_ids,
                    response_ids=self.response_ids,
                    encoder_outputs=self.encoder_outputs
                )
        unpert_probs = self.model.lm_head.get_probs(unpert_last_hidden[:, -1, :])
        accumulated_hidden = torch.sum(unpert_last_hidden[:, :-1, :], dim=1)
        
        # build embeddings for last token
        embeddings = self.model.transformer.get_input_embeddings()
        inputs_embeds = embeddings(self.response_ids[:, -1:])
        if self.using_gpt2:
            token_type_embeds = embeddings(torch.tensor([[self.spec_ids.response]],
                                                        device=self.device,
                                                        dtype=torch.long))
        else:
            token_type_embeds = 0
        # pisition_embeds will be calculated automatically in the model
        embeds = inputs_embeds + token_type_embeds
        
        # calculate perturbed past
        pert_past = self._perturb_past(embeds, unpert_probs, unpert_past, accumulated_hidden)
        pert_last_hidden, *_ = self.model.get_hidden_and_past(inputs_embeds=embeds,
                                                                   past=pert_past,
                                                                   encoder_outputs=self.encoder_outputs)

        # fuse the perturbed probs and the real one
        pert_probs = self.model.lm_head.get_probs(pert_last_hidden[:, -1, :], temperature=self.temperature)
        pert_probs = (pert_probs ** self.fusion_scale) * \
                        (unpert_probs ** (1 - self.fusion_scale))
        # rescale
        if torch.sum(pert_probs) <= 1.0:
            pert_probs = pert_probs / torch.sum(pert_probs)

        return pert_probs

    def _perturb_past(self, input_embeds, unpert_probs, unpert_past, accumulated_hidden):
        # TODO: add mask for past

        # generate initial perturbed past
        grad_accumulator = torch.zeros(self.past.shape).to(self.device)

        # accumulate perturbations for num_iterations
        for _ in range(self.iterations):
            cur_perturbation = grad_accumulator.detach().clone().requires_grad_(True)
            cur_length = cur_perturbation.shape[-2]

            # compute hidden using perturbed past
            pert_past = self.past + cur_perturbation
            pert_last_hidden, *_ = self.model.get_hidden_and_past(inputs_embeds=input_embeds,
                                                                 past=pert_past,
                                                                 encoder_outputs=self.encoder_outputs)
            new_accumulated_hidden = accumulated_hidden + \
                torch.sum(pert_last_hidden, dim=1)
            probs = self.model.lm_head.get_probs(pert_last_hidden[:, -1, :])

            loss = 0.0
            # use BOWs
            if self.attr_model.ohv:
                for one_hot_bow in self.attr_model.ohv:
                    bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                    bow_loss = -torch.log(torch.sum(bow_logits))
                    print('bow loss: ', bow_loss)
                    loss += bow_loss
            
            # use discriminators
            if self.attr_model.active_heads:
                cur_unpert_past = unpert_past
                # we want shape of embeds to be:
                # (batch_size, seq_len, hid_size) -> (1, 1, hid_size)
                # then probs shape:
                # (batch_size, seq_len, vocab_size) -> (1, 1, vocab_size)
                cur_probs = probs.unsqueeze(1)
                for _ in range(self.horizon_len):
                    # size: (1, 1, hid_size)
                    wte = self.model.transformer.get_input_embeddings().weight.data
                    embeds = torch.matmul(cur_probs, wte)
                    cur_last_hidden, cur_unpert_past, _ = self.model.get_hidden_and_past(
                        inputs_embeds=embeds,
                        past=cur_unpert_past,
                        encoder_outputs=self.encoder_outputs
                    )
                    new_accumulated_hidden += torch.sum(cur_last_hidden, dim=1)
                
                # make predictions for current accumulated hidden
                predictions, _ = self.attr_model(
                    new_accumulated_hidden /
                    (cur_length + 1 + self.horizon_len)
                )
                
                discrim_loss = 0.0
                for head in predictions.keys():
                    label = self.head2label[head]
                    classes = self.attr_model.discriminator[head].map_labels_to_classes([label])
                    discrim_loss += self.attr_model.discriminator[head].loss(predictions[head], classes)
                    print('discrim loss: ', discrim_loss.item())
                
                loss += discrim_loss

            # add KL divergence to reduce the distance between the new distribution and the real one
            kl_loss = 0.0
            if self.kl_scale > 0.0:
                unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(self.device).detach()
                )

                correction = SMALL_CONST * \
                             (probs <= SMALL_CONST).float().to(self.device).detach()
                corrected_probs = probs + correction.detach()

                kl_loss = self.kl_scale * (
                    (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
                )
                loss += kl_loss

            loss.backward(retain_graph=True)

            # calculate gradient norms; one value for one layer -> grad_norms.shape = (n_layers, 1)
            if self.grad_norms is not None:
                # give grads with new view (n_layers, -1) to the torch.norm(x, dim=1)
                # it gives exactly one value per layer in the model
                cur_norm = torch.norm(cur_perturbation.grad.view(self.past.size(0), -1),
                                      dim=1)
                # always keep max norm
                self.grad_norms = torch.max(self.grad_norms,
                                            cur_norm)
            else:
                self.grad_norms = torch.norm(cur_perturbation.grad.view(self.past.size(0), -1),
                                             dim=1) + SMALL_CONST

            # new view for grad norms for broadcasting
            # tensor with shape (n_layers, 1,..., 1) will be repeated along axis with size 1
            norms_view = (-1,) + tuple(1 for _ in self.past.shape[1:])
            # addcdiv_ equals += value * (tensor1 / tensor2)
            grad_accumulator.addcdiv_(cur_perturbation.grad.data,
                                      torch.pow(self.grad_norms, self.gamma).view(norms_view),
                                      value=-self.stepsize)

            # reset gradients, just to make sure
            cur_perturbation.grad.data.zero_()
            
            # removing past and accumulated from the graph
            self.past = self.past.detach()
            new_accumulated_hidden = new_accumulated_hidden.detach()

        # apply the accumulated perturbations to the past
        pert_past = self.past + grad_accumulator.detach()

        return pert_past
