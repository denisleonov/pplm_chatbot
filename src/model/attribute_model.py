import torch
import torch.nn as nn
import yaml

from src.model.model import Generator
from src.utils.bow import get_bag_of_words_indices, build_bows_one_hot_vectors
from src.utils.utils import apply_threshold

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, num_classes, embed_size, size):
        super(ClassificationHead, self).__init__()
        self.size = size
        self.num_classes = num_classes
        self.embed_size = embed_size
        if self.size=='big':
            self.mlp1 = torch.nn.Sequential(torch.nn.Dropout(0.3),
                                            torch.nn.Linear(embed_size, embed_size),
                                            torch.nn.Tanh())
        self.mlp = torch.nn.Linear(embed_size, num_classes)
    

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        if self.size == 'big':
            hidden_state = self.mlp1(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


class Discriminator(Generator):
    """Model with classification head. Each head can be (de)activated in-time.

    Heads are configured in yaml files and have multilabel/multiclass types, big/small configurations.
    Small heads can use uber weights and have the same format. To use them as discriminators for LM, it is
    needed to configue desired label(s) as 'very positive', 'toxic' etc. Labels and their number representations
    are given in yaml files.

    """

    # TODO: add head strength
    # TODO: test head activation/deactivation

    def __init__(self, d_hidden, vocab_size, config):
        super().__init__(d_hidden, vocab_size)
        self.d_hidden = d_hidden
        self.sigmoid = nn.Sigmoid()
        self.config = config
        self.classifier = ClassificationHead(num_classes=self.config['num_classes'],
                                             embed_size=d_hidden, size=self.config['size'])
        self.use = 1
        if self.config['classification_type'] == 'multiclass':
            self.loss = nn.CrossEntropyLoss()
        if self.config['classification_type'] == 'multilabel':
            self.loss = nn.BCEWithLogitsLoss()

    # TODO: rewrite this. we can't just use torch.mean because of pads.
    def avg_representation(self, x):
        avg_hidden = torch.mean(x, dim=1)
        return avg_hidden

    def get_num_classes(self):
        return self.classifier.num_classes

    def get_classifier(self):
        return self.classifier

    def activate(self):
        self.use = 1

    def deactivate(self):
        self.use = 0

    def forward(self, x):
        """Returns changed logprobs"""
        if x.dim() == 3:
            x = self.avg_representation(x)
        return self.classifier(x)

    def get_loss(self, preds, classes):
        if self.config['classification_type'] == 'multilabel':
            classes = classes.to(torch.float)
        else:
            classes = classes.to(torch.long)
        return self.loss(preds, classes)

    def get_probs(self, x):
        '''
        Returns probs of classes depending on model type (multilabel/multiclass classification)
        :param x:
        :return: torch tensor with probabilities of classes
        '''
        if self.config['classification_type'] == 'multiclass' or self.config['classification_type'] == 'binary':
            return self.softmax(x)
        if self.config['classification_type'] == 'multilabel':
            return self.sigmoid(x).float()
        else:
            raise Exception('This type of classification is not known!')

    def get_labels(self, x):
        '''
        Returns binary vector of size num_classes and ones on places of predicted classes

        :param x: probabilities of classes
        :return: class number or binary tensor with classes
        '''
        if self.config['classification_type'] == 'multilabel':
            return apply_threshold(self.get_probs(x), self.config['treshold'])
        else:
            return x.argmax(dim=1, keepdim=True)

    def load(self):
        '''
        Loads head weights according to configured path and model mode (small or big)
        '''
        device = next(self.classifier.parameters()).device
        head_state = torch.load(self.config['path_to_weights'] + '_' + self.config['size']+'.pt', map_location=device)
        self.classifier.load_state_dict(head_state)
        print(f"loaded discriminator state | head {self.config['name']}")

    def save(self):
        '''
        Saves head weights according to configured path and model mode (small or big)
        '''
        state = self.classifier.state_dict()
        torch.save(state, self.config['path_to_weights'] + '_' + self.config['size']+'.pt')
        print(f"Saved discriminator state | head {self.config['name']} at {self.config['path_to_weights'] + '_' + self.config['size'] + '.pt'}")
        #torch.load(self.config['path_to_weights'])

    def map_label_to_class(self, label):
        '''Maps label ('toxic' etc) to class number of this head'''
        try:
            return self.config['classes'][label]
        except:
            raise Exception(f"This label is not applicable to this head! \
                | head -> {self.config['name']} \
                label -> {label}")

    def map_labels_to_classes(self, labels, device='cpu'):
        """ Maps list of class numbers to binary vector. Can be used boh for multiclass and multilabel task"""
        classes = [self.map_label_to_class(label) for label in labels]
        if self.config['classification_type'] == 'multilabel':
            binary_classes = [1 if i in classes else 0 for i in range(self.config['num_classes'])]
            return torch.tensor(binary_classes, device=device, dtype=torch.float32)
        else:
            return torch.tensor(classes, device=device, dtype=torch.long)


class AttributeModel(nn.Module):
    '''
    Model with heads dict and BoW vectors for desired attributes.
    '''
    def __init__(self, embed_size, vocab_size, args, tokenizer=None, device=torch.device('cpu')):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.device = device
        self.train_mode = False
        self.ohv = None
        self.active_heads = None
        self.to(device)
        self.args = args

    def initialize_bow(self, topics):
        if not topics:
            self.ohv = None
            return
        
        bow_indices = get_bag_of_words_indices(topics, self.tokenizer)
        # make set of all word ids in all BoWs, which will be used for colorization
        self.bow_indices = set([i for bow in bow_indices
                                for indices in bow
                                for i in indices])
        self.ohv = build_bows_one_hot_vectors(bow_indices, self.tokenizer, self.device)

    def initialize_discriminator(self, heads, full_init=False):
        if full_init:
            self.discriminator = nn.ModuleDict()
            self.num_heads = 0
            self.heads = set()
            self.active_heads = set()
        if not heads:
            return
        
        heads = set(heads)
        heads_to_activate = heads.difference(self.active_heads)
        heads_to_deactivate = self.active_heads.difference(heads)

        self.activate_heads(heads_to_activate)
        self.deactivate_heads(heads_to_deactivate)

    def save(self):
        for head in self.heads:
            self.discriminator[head].save()
        print('Saved heads')

    def load(self):
        for head in self.heads:
            try:
                self.discriminator[head].load()
            except:
                print(f'Could not load {head} state!')


    def deactivate_heads(self, heads):
        for head in heads:
            self.discriminator[head].deactivate()
            self.active_heads.discard(head)

    def activate_heads(self, heads_to_add):
        for head in heads_to_add:
            if head in self.discriminator.keys():
                self.discriminator[head].activate()
            else:

                self.discriminator[head] = Discriminator(
                    self.embed_size,
                    self.vocab_size,
                    self.args[head]
                ).to(self.device)
                self.num_heads += 1
                self.heads.add(head)
                self.active_heads.add(head)
                self.discriminator[head].train(self.train_mode)

    def get_head_preds(self, hidden_state):
        probs = dict()
        for head in self.active_heads:
            probs[head] = self.discriminator[head](hidden_state).float()
        return probs

    def get_head_labels(self, probs):
        labels = {}
        for head in self.active_heads:
            labels[head] = self.discriminator[head].get_labels(probs[head])
        return labels

    def forward(self, hidden):
        logprobs = self.get_head_preds(hidden)
        labels = self.get_head_labels(logprobs)
        return logprobs, labels

    def eval(self):
        self.train(False)

    def train(self, mode=True):
        for head in self.discriminator.keys():
            self.discriminator[head].train(mode=mode)
        self.train_mode = mode #to add heads later with proper mode
        super().train(mode)
