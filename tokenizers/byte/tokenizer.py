import pickle
import os
from typing import List

class ByteLevelTokenizer:
    def __init__(self):
        # Define special tokens
        self.bos_id: int = 256
        self.eos_id: int = 257
        self.pad_id: int = 258
        self.vocab_len: int = 259

    def encode(self, text: str, bos: bool = True, eos: bool = False, pad: int = None) -> List[int]:
        """
        Encodes a string into a list of byte token indices.

        Parameters:
        text (str): The string to be encoded.
        bos (bool): Whether to add a beginning of sequence token.
        eos (bool): Whether to add an end of sequence token.
        pad (int): The total length of the output sequence with padding.

        Returns:
        List[int]: The encoded token indices.
        """
        # Convert text to bytes
        tokens = list(text.encode("utf-8"))

        # Optionally add BOS and EOS tokens
        if bos:
            tokens.insert(0, self.bos_id)
        if eos:
            tokens.append(self.eos_id)

        # Optionally pad the sequence to the desired length
        if pad is not None:
            tokens.extend([self.pad_id] * (pad - len(tokens)))
            tokens = tokens[:pad]  # Ensure sequence is exactly `pad` length

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of token indices back into a string.

        Parameters:
        tokens (List[int]): The list of token indices to decode.

        Returns:
        str: The decoded string.
        """
        # Filter out any special tokens
        tokens = [t for t in tokens if t < 256]

        # Convert byte indices back to bytes
        byte_array = bytes(tokens)

        # Convert bytes to string
        return byte_array.decode("utf-8", errors="ignore")


def get_tokenizer(size = None):
    # size is left as an argument just to ensure compatibility with other code in the repo; it does not get used
    return ByteLevelTokenizer()