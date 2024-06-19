# bpe
a minimal implementation of GPT4's style of byte-pair-encoding algorithm trained on the tinyStories dataset. Some notes:
- because it's trained on tinyStories, i do not recommend using this tokenizer for any other dataset
- all of the saved vocabulary sizes are 3 less than a power of 2. the idea here is that our tokenizer has 3 special tokens (bos, eos, & pad) and we want our nn.embedding to have number of rows equal to a power of 2 for efficiency reasons (it's not a huge difference but why not)