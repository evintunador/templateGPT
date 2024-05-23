import pickle
import os
from typing import List

class BPE_Tokenizer:
    def __init__(self, stoi, merges):
        self.stoi = stoi
        self.merges = merges
        self.itos = {i: s for s, i in stoi.items()}  # Inverse mapping for decoding

        self.vocab_len = len(stoi) + len(merges) + 3
        
        self.bos_id: int = len(stoi) + len(merges)
        self.eos_id: int = len(stoi) + len(merges) + 1
        self.pad_id: int = len(stoi) + len(merges) + 2

    def encode(self, text: str, bos: bool = True, eos: bool = False, pad: int = None) -> List[int]:
        """
        Converts a string into a list of token indices.
        
        text: the string to be encoded
        bos: whether to add a beginning of sequence token
        eos: wheter to add an end of sequence token. Should be used during training, not inference
        pad: whether to add padding at the end. Should be used during training, not inference
        """
        assert isinstance(text, str)
        # Convert the text to a list of token IDs, using space for unknown characters
        tokens = [self.stoi.get(c, self.stoi[' ']) for c in text]

        # Perform merging with the possibility of nested merges
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges:
                # Replace the current pair with its merged token
                merged_token = self.merges[pair]
                tokens[i] = merged_token
                del tokens[i + 1]

                # Move back to handle possible nested merges
                if i > 0:
                    i -= 1
            else:
                i += 1
        
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if pad is not None:
            tokens = tokens + [self.pad_id for _ in range(pad - len(tokens))]
            tokens = tokens[:pad]
            
        return tokens

    def expand_token(self, token):
        """ Helper function to get string representation of a token """
        # Base case: if the token is a direct mapping, return its character
        if token in self.itos:
            return self.itos[token]
        # Recursive case: if the token is a merged token, expand its constituents
        elif token in self.merges.values():
            pair = next(key for key, value in self.merges.items() if value == token)
            return ''.join(self.expand_token(t) for t in pair)
        # Fallback for unknown tokens, bos, eos, and padding (which will be a thing if i ever get batched inference working)
        else:
            return ''

    def decode(self, tokens):
        # Decode each token in the list, handling nested merges recursively
        return ''.join(self.expand_token(token) for token in tokens)

    def display(self, text: str, as_list: bool = False):
        """
        Display the tokens obtained from the text.
        
        text: the string to be tokenized
        as_list: if True, returns the tokens as a list of strings; otherwise, returns a single string with tokens separated by '|'
        """
        # First, encode the text to get the list of token IDs
        token_ids = self.encode(text, bos=False, eos=False)
        
        # Now convert the token IDs back to strings
        token_strings = [self.expand_token(token) for token in token_ids]
        
        if as_list:
            return token_strings
        else:
            return '|'.join(token_strings)

def load_tokenizer_data(path):
    with open(path, 'rb') as f:
        tokenizer_data = pickle.load(f)
    return tokenizer_data

def get_tokenizer(size: int = 8192):
    """
    bpe sizes include 95, 128, 256, 512, 1024, 2048, 4096, 8192
    """
    path = f'tokenizers/bpe_v1/models/{size}.model'
    tokenizer_data = load_tokenizer_data(path)
    loaded_stoi = tokenizer_data['stoi']
    loaded_merges = tokenizer_data['merges']
    return BPE_Tokenizer(loaded_stoi, loaded_merges)