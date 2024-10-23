# templateGPT
## about
**this repo does not currently implement every feature I would like in order for it to be fully able to do what it was meant for. eventually if/when I do get back to it the goal is to get this capable of completely replicating GPT-2 (or a GPT-2 sized version of Llama3.1) on multiple GPUs and then start using it as a template for architecture experiments.**


This is the model I edit whenever I want to test a new transformer architecture idea I have. It's designed to be:
- flexible in that many large changes are tweakable from the config file rather than messing with the code
- easy to read/edit the code since files are cleanly organized & well commented
- well suited for training models in the 1-10m parameter range on a CPU or the 100m-1b parameter range on a GPU without editing any code, just the config file
- easy to visualize/demonstrate what's happening in the progression of tensor shapes for learning & debugging purposes (thanks to our custom `LoggingModule.py` and `test_modules.ipynb`)
- almost as efficient as Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) despite everything we've added
- up to date with the most recent SotA architecture, namely Llama 3.1 (Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) is based on the very old GPT2 and his [nanoLlama31](https://github.com/karpathy/nano-llama31) library is built for fine-tuning the full 8b size rather than pre-training 1m-1b sized models)

Notice that even though some of these models are very small (1 to 10m parameters), they are actually reasonable rough proxies for how well a scaled up version may do on real data thanks to our use of the [TinyStories](https://arxiv.org/abs/2305.07759) dataset. According to the original paper, somewhere in the 1 to 3m parameter range a GPT2-inspired architecture is capable of understanding that the token 'apple' is something that the main character of the tiny story 'Tim' would like to 'eat'; meaning it can actually pick up on the relationships in this text which are an isomorphic subset of the ones that a larger language model would see when training on the entire internet. This basic idea is the backbone behind microsoft's Phi family of models, originally described in the paper [Textbooks Are All You Need](https://arxiv.org/pdf/2306.11644), and how they can perform so well despite being so small. I hope this repo can be of help to anyone who wants to get into designing & building novel architectures but doesn't have the compute to test a larger model on every single idea they have. I'm literally training the 1-5m parameter models on the CPU of a 2019 iMac with 8gb of ram.

Then when it's time to scale up (100m-1b parameters) and use a GPU, all you have to do is go into the config file and switch to the [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) or [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Realistically single old GPUs are cheap enough nowadays (less than $1 per hour) that you could train the 1-10m parameter models on them for cheap and that's what I usually do, but it's still nice to think that someone who's resource constrained can mess around without having to learn how to use & pay for a GPU cloud solution at all, or that someone with a halfway decent CPU/GPU/MPS might find it easier to test locally before switching to a cloud GPU node. 

[![ERROR DISPLAYING IMAGE, CLICK HERE FOR VIDEO](https://img.youtube.com/vi/s9kQvDsWnbc/0.jpg)](https://www.youtube.com/watch?v=s9kQvDsWnbc)

This repo is part of a larger project of mine called [micro-GPT-sandbox](https://github.com/evintunador/micro-GPT-sandbox) that's basically a hub for all the novel model experiments I do, with the goal of facilitating easy comparison between the different models. Basically for each of those experiments I just use this very repo as a template to start editing, and then once I'm happy with the project (or if I've just abandoned it but it's moderately functional) I add it to the sandbox. If you end up using this repo as a template, feel free to contribute your project to the sandbox as well!

## getting started
1. clone the repository
2. `cd` to the folder
3. setup a virtual environment unless you're an agent of chaos. Use python 3.12.4; pytorch doesn't like 3.13
4. `pip install -r requirements.txt`
5. edit values in `config.py` to suit your liking. This might involve a lot of trial and error if you don't know what you're doing, either due to errors from incompatible parameter configurations or from going over your available vRAM amount. Checkout the config files for each already trained model to get an idea of what reasonable values look like
6. Run `python train.py` to train your own version of templateGPT
7. If you ever want to just test out a model you've already made then run the following command. The name of each model is the name of the folder it resides in inside `models/`. The model you run need not match up with the hyperparameters currently in `config.py`, that file is just for setting up training.
```
python inference.py <insert_model_name_here> "prompt"
```
8. If you've trained multiple models, you can compare them in `model_comparison.ipynb` as long as you remember to use the third cell to specify which models you want to compare. It'll look at loss curves over the course of training and teacher-forcing topk accuracy rate
9. This step could really go anywhere, but if you're trying to learn how transformers work then along with reading the code in `modules/` you can use `test_modules.ipynb` to visualize how the tensor shapes change. Each cell shows you in detail how a different module or scenario works in terms of how the tensor shapes change as they move through
10. If/when you become confident to mess with the actual code yourself and test out a novel architecture idea you've got, head on over into `modules/` and get to work. While you're doing this, make sure to use `LoggingModule` instead of `nn.module` and put `@log_io` before every class function you make so that you can use `test_modules.ipynb` for easy visualization/debugging. 
11. If/when you've got a novel transformer architecture edit up and working, send it over to your own template/fork of [micro-GPT-sandbox](https://github.com/evintunador/micro-GPT-sandbox) for easy comparisons against the original templateGPT (micro-GPT-sandbox is currently in an even less finished state than this repo)

## file structure
- `tokenizers/`: a folder where you store your tokenizers
    - `bpe_tinyStories/`: a [byte-pair encoding](https://huggingface.co/learn/nlp-course/chapter6/5) tokenizer trained on the first 10k sequences from the [TinyStoriesV2](https://huggingface.co/datasets/noanabeshima/TinyStoriesV2) dataset, which is a fan-made upgrade over the original [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
        - `build.ipynb`: the notebook where i trained the tokenizer models
        - `tokenizer.py`: an overly-simplistic and annoyingly inefficient tokenizer with bos & eos tokens, post-sequence padding, and a `display` function to help you visualize how a given string is broken down into tokens
        - `models/`
            - `{509, 1021, 2045}.model`: different tokenizer sizes, each a subset of the next. 
    - `bpe_fineweb/`: a yet-to-be trained [byte-pair encoding](https://huggingface.co/learn/nlp-course/chapter6/5) tokenizer of [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
        - ...
    - `bpe_fineweb-edu/`: a [byte-pair encoding](https://huggingface.co/learn/nlp-course/chapter6/5) tokenizer trained on the first 2k sequences from the "sample-350BT" subset of [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). We train the model on the "sample-10BT" subset which means the tokenizer was ~mostly~ trained on data the model won't see during training
        - ...
        - `models/`
            - `{509, 1021, 2045, 4093, 8189, 16381, 32765}.model`: different tokenizer sizes, each a subset of the next.
    - `byte/`: choose this to use bytes instead of tokens
          - ...
- `modules/`: where all of the code for the actual model goes
    - `attention.py`: [multi-query attention](https://arxiv.org/abs/1911.02150) with pre-computed [rotary positional encodings](https://arxiv.org/abs/2104.09864) that knows to automatically use [Flash Attention](https://github.com/Dao-AILab/flash-attention) if you have access to a cuda GPU.
    - `layer.py`: defines each residual connection layer of our GPT
    - `logging.py`: defines the `LoggingModule` class, a wrapper that you should use instead of pytorch's `nn.module` in order to facilitate easy demonstration of how tensor shapes change throughout a given module
    - `mlp.py`: a multi-layer perceptron with an optional gate and either [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [GeLU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html), or [SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) nonlinearities, all configurable in `config.py`. Adding more nonlinearities is also absurdly easy
    - `model.py`: the primary class for our GPT
    - `norm.py`: a norm module with an optional affine layer that allows you to switch between [RMSNorm](https://arxiv.org/abs/1910.07467), [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) and [CosineNorm](https://arxiv.org/pdf/1702.05870) easily using a setting over in `config.py`. Adding new normalization methods is also absurdly easy
- `trained/`
    - `Llama3_1m_atto/`: a 1m parameter model trained for 2k iterations with a batch size of 64 for a total of 128k sequences (tinyStoriesV2 dataset is ~2.76 million sequences, so less than 5\% of the available dataset) and designed to resemble the architecture of [Llama 3](https://ai.meta.com/blog/meta-llama-3/)/[nanoLlama31](https://github.com/karpathy/nano-llama31). However it uses BPE rather than llama's tokenization scheme, and i think i also used more dropout during training
        - `model_config.json`: hyperparameters of the model
        - `model.pth`: weights of the model
        - `train_config.json`: hyperparameters of the training loop used
        - `log_data.csv`: a record of loss and a couple other key metrics over the course of training
    - `GPT2_1m_atto/`: a 1m parameter model trained for 2k iterations with a batch size of 64 for a total of 128k sequences (tinyStoriesV2 dataset is ~2.76 million sequences, so less than 5\% of the available dataset) and designed to resemble the architecture of [GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)/[nanoGPT](https://github.com/karpathy/nanoGPT)
        - `...`
- `config.py`: all of the easily editable model and training settings
- `inference.py`: run with multiple prompts and edit your settings like so:
```
python inference.py "insert_model_name_here" "prompt 1" "prompt 2" "prompt..." --temp=0.7 --min_p=0.05 --top_k=None --top_p=None --max_len=100 --show_tokens
```
- `model_comparison.ipynb`: open this notebook to compare different models against each other. includes loss curve plots and topk teacher-forcing accuracy rate
- `model_comparison.py`: functions for comparing models; used in `model_comparison.ipynb`
- `view_modules.ipynb`: creates easy printouts that allow you to follow the progression of tensor shapes for demonstration & debugging purposes of all the `loggingmodule`s in `modules/`. If you're building new modules for a novel architecture idea you have then this notebook will be of extreme value to you in debugging & visualization. Also includes visualizations of the learning rate scheduler and how a given piece of text is tokenized with your chozen tokenizer
- `tools.py`: A variety of functions & classes that don't fit elsewhere and/or are used by more than one of the jupyter notebooks. I should prolly find a better way to organize these
- `train.py`: first edit `config.py` then run this file to train a model like so:
```
python train.py --device=cuda
```

## definite TODOs
- [ ] decrease reliance on `logging.py` by creating tests for each module
- [ ] add useful stuff from karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
    - [ ] make it distributed data parallelizable on cuda
    - [ ] setup downloaded datasets to optionally download as token indices rather than as strings (makes loading them during training faster)
    - [ ] add the benchmark test
- [ ] train new tokenizers
    - [x] tinystoriesv2
    - [ ] fineweb
    - [x] fineweb-edu
    - [ ] make it possible to start from a tokenizer as a checkpoint to make a larger tokenizer
- [ ] figure out random seed & exact replication versus shuffling a dataset
- [ ] implement kv caching based on the code in [karpathy's nanoLlama31](https://github.com/karpathy/nano-llama31) 
- [ ] add batched inference to `inference.py`
- [ ] setup training batches and attention mask to concatenate more than one sequence back to back when the docs are shorter than the model's maximum context length
- [ ] go back and make sure model checkpointing is working. at one point it was but i've changed so much since then and haven't bothered using it so i'd bet it's broken
- [ ] make dropout at different places optional [display of gpt2 vs llama w/ dropout](https://media.licdn.com/dms/image/v2/D5622AQEBPGmatFddzQ/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1727097580150?e=1730332800&v=beta&t=6GiWyRnpzGyOJB4cmbsgapGioeOZqa0jYCx50o2__24)
- [ ] parallelize that one part of teh attention mechanism that's currently a for loop

### potential future TODOs
- [ ] use https://blog.eleuther.ai/mutransfer to set hyperparametrs?
- [ ] add option to continually train pre-existing models & update its training data/hyperparameters accordingly
- [ ] add automated model comparison analysis by GPT4 like in the [TinyStories](https://arxiv.org/abs/2305.07759) paper into `model_comparison.ipynb`
- [ ] add sparse/local/windowed attention mask options
- [ ] switch to [flexAttention](https://pytorch.org/blog/flexattention/)???
- [ ] take advantage of [torchao](https://pytorch.org/blog/pytorch-native-architecture-optimization/?utm_content=309679619&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024)???
- [ ] switch big dataset from fineweb to [TxT360](https://huggingface.co/datasets/LLM360/TxT360)
      
## how to contribute
Other than the above TODO lists, appreciated contributions include:
- bug fixes
- adding more detailed comment explanations of what the code is doing
- general readability edits
- efficiency edits
- editing the code in `modules/` to take better advantage of the `LoggingModule`. This means splitting up each class into more and tinier functions
- training more models (especially if they're bigger than what's already here!)

Because I'm not super knowledgeable on how collaborating on git projects works and I tend to edit directly on the main branch, please reach out and communicate with me about any edits you plan to make so that I can avoid editing the same files. [Click here to join my discord server](https://discord.gg/hTYQyDPpr9)

## check me out
- guides on how to build miniature versions of popular models from scratch, with a hand-holding walkthrough of every single tensor operation: [minGemma](https://github.com/evintunador/minGemma), [minGrok](https://github.com/evintunador/minGrok), and [minLlama3](https://github.com/evintunador/minLlama3). Future versions of those kinds of guides I make will use this repo as a template
- [my YouTube channel](https://www.youtube.com/@Tunadorable)
- my [other links](https://linktr.ee/tunadorable)
