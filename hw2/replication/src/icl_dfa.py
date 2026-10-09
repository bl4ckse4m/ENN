"""RegBench (ICL DFA) data — faithful port of berlino/seq_icl dfa.py.

Protocol (Akyurek et al., 2024, benchmark used in paper Table 1):
  * random DFAs/PFAs (all states accepting) with <= max_num_nodes states,
    alphabet subset of `vocab_size - 2` letters, <= max_outgoing_edges
    outgoing transitions per state; minimized + trimmed;
  * one example = one DFA: 10-20 random words sampled from it, joined by the
    separator "|";  train/val/test split by DFA (held-out PFAs);
  * tokens: letters a.., separator "|", padding noop ".";
  * labels = next-token ids, -100 where the target is "|" or padding.

Metrics (ports of train.py::_get_dfa_accuracy and analyze.py::get_dfa_probs):
  * dfa accuracy: fraction of predicted next-symbols that are valid DFA
    transitions (the paper's "Accuracy" column);
  * TVD: mean L1 distance between the model's next-token distribution and the
    uniform distribution over the DFA's outgoing symbols (x100, as in Table 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pythomata import SimpleDFA
from torch.utils.data import DataLoader, Dataset


class DFA:
    def __init__(self, num_nodes, alphabet, transitions, rng):
        transitions = {i: v for i, v in enumerate(transitions)}
        self.dfa = SimpleDFA(
            states=set(range(num_nodes)),
            alphabet=set(alphabet),
            initial_state=0,
            accepting_states=set(range(num_nodes)),
            transition_function=transitions,
        )
        self.rng = rng

    def _sorted_transitions(self):
        nodes = sorted(self.dfa._transition_function.keys())
        return tuple(
            tuple(sorted(self.dfa._transition_function[n].items(), key=lambda item: item[1])) for n in nodes
        )

    def _minimize(self):
        self.dfa = self.dfa.minimize()
        return self

    def _trim(self):
        self.dfa = self.dfa.trim()
        return self

    def __hash__(self):
        return hash(self._sorted_transitions())

    def __eq__(self, other):
        return isinstance(other, DFA) and self._sorted_transitions() == other._sorted_transitions()

    def __call__(self, word: str) -> bool:
        node = self.dfa._initial_state
        for symbol in word.split():
            if symbol not in self.dfa._transition_function[node]:
                return False
            node = self.dfa._transition_function[node][symbol]
        return True

    def forward(self, word: str):
        node = self.dfa._initial_state
        for symbol in word.split():
            if symbol not in self.dfa._transition_function[node]:
                return None
            node = self.dfa._transition_function[node][symbol]
        return node

    def sample(self, length: int = 1) -> str:
        node = self.dfa._initial_state
        word = ""
        for _ in range(length):
            symbols = list(self.dfa._transition_function[node].keys())
            symbol = self.rng.choice(symbols)
            word += symbol + " "
            node = self.dfa._transition_function[node][symbol]
        return word.rstrip()


class RandomDFASampler:
    def __init__(self, num_nodes, alphabet, max_outgoing_edge, seed=None):
        self.num_nodes, self.alphabet, self.max_outgoing_edge = num_nodes, alphabet, max_outgoing_edge
        self.rng = np.random.default_rng(seed)

    def sample(self) -> DFA:
        transitions = []
        for _ in range(self.num_nodes):
            out = int(self.rng.integers(1, self.max_outgoing_edge + 1))
            symbols = self.rng.choice(list(self.alphabet), size=out, replace=False)
            dests = self.rng.integers(0, self.num_nodes, size=out)
            transitions.append({s: int(d) for s, d in zip(symbols, dests)})
        return DFA(self.num_nodes, self.alphabet, transitions, self.rng)


class Vocab:
    def __init__(self, vocab_size: int, special: dict):
        self.special_vocabs = special
        vocab = [chr(v + 97) for v in range(vocab_size)]
        self.non_special_vocab = sorted(vocab)
        self.vocab = sorted(set(vocab + list(special.values())))
        self.v2id = {v: i for i, v in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    @property
    def separator(self):
        return self.special_vocabs["seperator"]

    @property
    def noop(self):
        return self.special_vocabs["noop"]

    def get_id(self, t):
        return self.v2id[t]

    def get_vocab(self, i):
        return self.vocab[i]

    def __len__(self):
        return len(self.vocab)


def uniform_over(symbols, vocab: Vocab) -> np.ndarray:
    p = np.zeros(vocab.vocab_size)
    ids = [vocab.get_id(s) for s in symbols if s in vocab.v2id]
    if ids:
        p[ids] = 1.0 / len(ids)
    return p


def dfa_target_probs(raw_input: str, dfa: DFA, vocab: Vocab) -> np.ndarray:
    """Port of analyze.get_dfa_probs: uniform over outgoing symbols at every
    character position (same alignment convention as the reference code)."""
    probs = []
    for index, example in enumerate(raw_input.split("|")):
        for t in range(0, len(example) + 1):
            if t == 0 and index == 0:
                continue
            current_word = " ".join(list(example[:t]))
            node = dfa.forward(current_word)
            if node is None:
                possibilities = list(vocab.non_special_vocab)
            else:
                possibilities = list(dfa.dfa._transition_function[node].keys())
            probs.append(uniform_over(possibilities, vocab))
    return np.array(probs)


def dfa_accuracy(logits: torch.Tensor, inputs: torch.Tensor, dfas, vocab: Vocab):
    """Port of train.py::_get_dfa_accuracy: fraction of valid DFA transitions."""
    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    inputs = inputs.detach().cpu().numpy()
    total = correct = 0
    for b in range(preds.shape[0]):
        pred_chars = [vocab.get_vocab(t) for t in preds[b]]
        input_chars = [vocab.get_vocab(t) for t in inputs[b] if vocab.get_vocab(t) != vocab.noop]
        dfa = dfas[b]
        for t in range(len(input_chars)):
            if len(input_chars) > t + 1:
                if input_chars[t + 1] == vocab.separator:
                    continue
                if input_chars[t + 1] == vocab.noop:
                    break
            if len(pred_chars) > t:
                current_word = " ".join(input_chars[: t + 1] + [pred_chars[t]]).split(" | ")[-1]
                label = int(dfa(current_word))
                if current_word:
                    total += 1
                    correct += label
    return correct / max(1, total)


def tvd_metric(logits: torch.Tensor, inputs: torch.Tensor, dfas, vocab: Vocab) -> float:
    """Mean L1 distance x100 between model probs and uniform outgoing dist."""
    model_probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    total, count = 0.0, 0
    for b in range(model_probs.shape[0]):
        input_chars = [vocab.get_vocab(t) for t in inputs[b].tolist() if t != -100]
        input_chars = [c for c in input_chars if c != vocab.noop]
        raw = "".join(input_chars)
        dfa_probs = dfa_target_probs(raw, dfas[b], vocab)
        mp = model_probs[b][: len(dfa_probs)]
        total += np.abs(mp - dfa_probs[: len(mp)]).sum()
        count += dfa_probs.shape[0]
    return 100.0 * total / max(1, count)


@dataclass
class ICLConfig:
    num_examples: int = 1000
    num_test_examples: int = 500
    vocab_size: int = 20
    max_num_nodes: int = 12
    max_num_in_context_examples: int = 20
    min_num_in_context_examples: int = 10
    max_outgoing_edges: int = 4
    max_len_per_example: int = 50
    input_seq_len: int = 4096
    seed: int = 0


class ICLDFADataset(Dataset):
    def __init__(self, inputs, labels, dfas):
        self.inputs, self.labels, self.dfas = inputs, labels, dfas

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.dfas[idx]


def build_icl_dfa(cfg: ICLConfig):
    """Returns (train_ds, val_ds, test_ds, vocab) exactly as seq_icl does."""
    rng = np.random.default_rng(cfg.seed)
    vocab = Vocab(cfg.vocab_size - 2, {"seperator": "|", "noop": "."})

    dfas = set()
    for _ in range(cfg.num_examples * 10):
        num_nodes = int(rng.integers(cfg.max_outgoing_edges, cfg.max_num_nodes + 1))
        num_alphabet = int(rng.integers(cfg.max_outgoing_edges, cfg.vocab_size - 2 + 1))
        alphabet = rng.choice(cfg.vocab_size - 2, size=num_alphabet, replace=False)
        alphabet = tuple(chr(a + 97) for a in alphabet)
        sampler = RandomDFASampler(num_nodes, alphabet, cfg.max_outgoing_edges)
        sampler.rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        dfa = sampler.sample()
        dfa._minimize()._trim()
        dfas.add(dfa)
        if len(dfas) >= cfg.num_examples + cfg.num_test_examples:
            break

    dfas = list(dfas)
    rng.shuffle(dfas)
    n_train, n_test = cfg.num_examples, cfg.num_test_examples
    if len(dfas) < n_train + n_test:
        n_train = (len(dfas) * n_train) // (n_train + n_test)
        n_test = len(dfas) - n_train

    splits = {
        "train": dfas[:n_train],
        "test": dfas[n_train : n_train + n_test // 2],
        "val": dfas[n_train + n_test // 2 : n_train + n_test],
    }

    def gen_example(dfa, num_samples):
        example = ""
        for _ in range(num_samples):
            length = int(rng.integers(1, cfg.max_len_per_example))
            example += dfa.sample(length=length) + " | "
        example = example[:-3]
        if len(example) > cfg.input_seq_len:
            example = example[: cfg.input_seq_len]
        ids = [vocab.get_id(t) for t in example.split()]
        return torch.LongTensor(ids[:-1]), torch.LongTensor(ids[1:])

    out = {}
    pad_id = vocab.get_id(vocab.noop)
    sep_id = vocab.get_id(vocab.separator)
    for split, split_dfas in splits.items():
        xs, ys = [], []
        for dfa in split_dfas:
            n = int(rng.integers(cfg.min_num_in_context_examples, cfg.max_num_in_context_examples))
            x, y = gen_example(dfa, n)
            xs.append(x)
            ys.append(y)
        inputs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
        labels[labels == sep_id] = -100
        out[split] = ICLDFADataset(inputs, labels, split_dfas)

    return out["train"], out["val"], out["test"], vocab


def collate(batch):
    xs, ys, dfas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(dfas)


def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
