from model_dqn import CommandScorer
from collections import defaultdict
import re
import numpy as np
from textworld import EnvInfos
from typing import List, Mapping, Any, Optional
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda')

class NeuralAgentDQN:
    """ Simple Neural Agent for playing TextWorld games. """
    MAX_VOCAB_SIZE = 1000
    UPDATE_FREQUENCY = 10
    LOG_FREQUENCY = 1000
    GAMMA = 0.9
    EPS = 0.3
    
    def __init__(self) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        
        self.model = CommandScorer(input_size=self.MAX_VOCAB_SIZE, hidden_size=128).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)
        #self.model_copy = CommandScorer(input_size=self.MAX_VOCAB_SIZE, hidden_size=128)
        self.mode = "test"
        #self.model_copy.load_state_dict(self.model.state_dict())
    
    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        self.model.reset_hidden(1)
        self.last_score = 0
        self.no_train_step = 0
    
    def test(self):
        self.mode = "test"
        self.model.reset_hidden(1)
        
    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True)
    
    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.MAX_VOCAB_SIZE:
                return self.word2id["<UNK>"]
            
            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)
            
        return self.word2id[word]
            
    def _tokenize(self, text):
        # Simple tokenizer: strip out all non-alphabetic characters.
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        word_ids = list(map(self._get_word_id, text.split()))
        return word_ids

    def _process(self, texts):
        texts = list(map(self._tokenize, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(device)
        padded_tensor = padded_tensor.permute(1, 0) # Batch x Seq => Seq x Batch
        return padded_tensor
      
    def get_targets(self, episode):
        targets = []
        maxq = torch.max(episode[-1]['Q'])
        for t in reversed(range(len(episode)-1)):
            target = episode[t]['reward'] + self.GAMMA * maxq
            targets.append(target)
            maxq = torch.max(episode[t]['Q'])
            
        return targets[::-1]

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any], EPS=0.2) -> Optional[str]:
        Q = self.get_Q(obs, infos)
        if self.mode == 'test' or random.random()>EPS:
            action_id = torch.argmax(Q).detach()
        else:
            action_id = random.randint(0, len(infos['admissible_commands'])-1)
        
        action = infos["admissible_commands"][action_id]
        if done:
            self.model.reset_hidden(1)
        return action, action_id, Q.detach()

    def get_Q(self, obs, infos):
        # Build agent's observation: feedback + look + inventory.
        input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])
        
        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._process([input_])
        commands_tensor = self._process(infos["admissible_commands"])
        
        # Get our next action and value prediction.
        Q = self.model(input_tensor, commands_tensor)
        return Q.squeeze()

    def apply_updates(self, replay_buffer, NUM_UPDATES=2, EPS=0.2):
        for _i in range(NUM_UPDATES):
            avg_loss = []
            for episode in replay_buffer:
                self.model.reset_hidden(1)
                targets = self.get_targets(episode)
                
                q_loss = 0
                for step, target in zip(episode[:-1], targets):
                    Q = self.get_Q(step['obs'], step['infos'])
                    qsa = Q[step['command_id']]
                    q_loss += (qsa - target)**2
                self.optimizer.zero_grad()
                avg_loss.append(q_loss.item())
                loss = q_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                self.optimizer.step()
            print('avg_q_loss:{}'.format(sum(avg_loss)/len(avg_loss)))
