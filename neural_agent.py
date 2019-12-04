from model import CommandScorer
from collections import defaultdict
import re
import numpy as np
from textworld import EnvInfos
from typing import List, Mapping, Any, Optional
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda')

class NeuralAgent:
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
      
    def _discount_rewards(self, episode):
        returns, advantages = [], []
        R = episode[-1]['values']
        for t in reversed(range(len(episode)-1)):
            R = episode[t]['reward'] + self.GAMMA * R
            adv = R - episode[t]['values']
            returns.append(R)
            advantages.append(adv)
            
        return returns[::-1], advantages[::-1]

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> Optional[str]:
        
        # Build agent's observation: feedback + look + inventory.
        input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])
        
        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._process([input_])
        commands_tensor = self._process(infos["admissible_commands"])
        
        # Get our next action and value prediction.
        outputs, indexes, values, probs = self.model(input_tensor, commands_tensor)
        action = infos["admissible_commands"][indexes[0]]
        
        if done:
            self.model.reset_hidden(1)
        return action, indexes[0].detach(), probs.squeeze()[indexes[0]].detach(), values.detach()

    def get_probs(self, obs, infos, idx):
        # Build agent's observation: feedback + look + inventory.
        input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])
        
        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._process([input_])
        commands_tensor = self._process(infos["admissible_commands"])
        
        # Get our next action and value prediction.
        outputs, indexes, values, probs = self.model(input_tensor, commands_tensor)
        return probs.squeeze()[idx], values, probs

    def apply_updates(self, replay_buffer, NUM_UPDATES=2, EPS=0.2):
        for _i in range(NUM_UPDATES):
            avg_policy_loss = []
            avg_value_loss = []
            avg_entropy_loss = []
            for episode in replay_buffer:
                self.model.reset_hidden(1)
                returns, advantages = self._discount_rewards(episode)
                
                policy_loss = 0
                value_loss = 0
                entropy_loss = 0
                for step, ret, adv in zip(episode[:-1], returns, advantages):
                    probs, values, all_probs = self.get_probs(step['obs'], step['infos'], step['command_id'])
                    ratio = probs/step['prob']
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * adv
                    policy_surr = -torch.min(surr1, surr2)
                    all_log_probs = torch.log(all_probs)
                    policy_loss += policy_surr#-torch.log(probs)*adv #policy_surr
                    value_loss += (ret - values)**2
                    entropy_loss += (-all_probs * all_log_probs).sum()
                self.optimizer.zero_grad()
                avg_policy_loss.append(policy_loss.item())
                avg_value_loss.append(value_loss.item())
                avg_entropy_loss.append(entropy_loss.item())
                loss = policy_loss + 0.25*value_loss - 0.1*entropy_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                self.optimizer.step()
            print('avg_policy_loss:{}\tavg_value_loss:{}\tavg_entropy_loss:{}'.format(sum(avg_policy_loss)/len(avg_policy_loss), sum(avg_value_loss)/len(avg_value_loss), sum(avg_entropy_loss)/len(avg_entropy_loss)))
