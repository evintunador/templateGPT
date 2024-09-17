import pickle
import os
from typing import List

class BPE_Tokenizer:
    def __init__(self, merges):
        self.merges = merges

        self.vocab_len = 256 + len(merges) + 3
        
        self.bos_id: int = 256 + len(merges)
        self.eos_id: int = 256 + len(merges) + 1
        self.pad_id: int = 256 + len(merges) + 2

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text: str, bos: bool = True, eos: bool = False, pad: int = None) -> List[int]:
        """
        Converts a string into a list of token indices.
        
        text: the string to be encoded
        bos: whether to add a beginning of sequence token
        eos: wheter to add an end of sequence token. Should be used during training, not inference
        pad: whether to add padding at the end. Should be used during training, not inference
        """
        assert isinstance(text, str)
        
        # Convert the text to a list of token IDs
        tokens = list(text.encode("utf-8"))

        # Perform merging with the possibility of nested merges
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if pad is not None:
            tokens = tokens + [self.pad_id for _ in range(pad - len(tokens))]
            tokens = tokens[:pad]
            
        return tokens

    def get_stats(self, ids, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]): # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        newids = []
        i = 0
        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
        
    def expand_token(self, token):
        """ Helper function to get string representation of a token """
        # Base case: if the token is a direct mapping, return its character
        if token < 256:
            return self.vocab[token].decode("utf-8", errors="replace")
        # Recursive case: if the token is a merged token, expand its constituents
        elif token in self.merges.values():
            pair = next(key for key, value in self.merges.items() if value == token)
            return ''.join(self.expand_token(t) for t in pair)
        # Fallback for unknown tokens, bos, eos, and padding (which will be a thing if i ever get batched inference working)
        else:
            return ''

    def decode(self, tokens):
        """given tokens (list of integers), return Python string"""
        # first remove the bos, eos, & pad tokens if they're there
        tokens = [i for i in tokens if i not in [self.bos_id, self.eos_id, self.pad_id]]
        # the actual decoding of merges back into bytes
        ids = b"".join(self.vocab[idx] for idx in tokens)
        # conversion of bytes into string
        text = ids.decode("utf-8", errors="replace")
        return text

    def display(self, text: str):
        """
        Display the tokens obtained from the text as strings in a list
        """
        # First, encode the text to get the list of token IDs
        token_ids = self.encode(text, bos=False, eos=False)
        
        # Now convert the token IDs back to strings
        token_strings = [self.expand_token(token) for token in token_ids]
        
        return token_strings

def load_tokenizer_data(path):
    with open(path, 'rb') as f:
        tokenizer_data = pickle.load(f)
    return tokenizer_data

def get_tokenizer(size: int = 2048):
    """
    sizes include 512, 1024, 2048
    """
    path = f'tokenizers/bpe_tinyStories/models/{size-3}.model'
    tokenizer_data = load_tokenizer_data(path)
    return BPE_Tokenizer(tokenizer_data['merges'])