# templateGPT
## about
This is the model I edit whenever I want to test a new transformer architecture idea I have. It's designed to be flexible with many large changes being tweakable from the config, easy to demonstrate what's happening in the progression of tensor shapes, and easy to read/edit the code. Feel free to toy around and build off of it, this repo is actually a template

Notice that even though these models are very small (1 to 4m parameters), they are actually reasonable rough proxies for how well a scaled up version may do on real data thanks to our use of the [TinyStories](https://arxiv.org/abs/2305.07759) dataset. According to the original paper, somewhere in the 1 to 3m parameter range, a GPT2-inspired architecture is capable of understanding that the token 'apple' is something that the main character of the tiny story 'Tim' would like to 'eat'; meaning it can actually pick up on the relationships in this text which are an isomorphic subset of the ones that a larger language model would see when training on the entire internet. This basic idea is the backbone behind microsoft's Phi family of models, originally described in the paper [Textbooks Are All You Need](https://arxiv.org/pdf/2306.11644), and how they can perform so well despite being so small. In practice the models you see here don't quite get there because 1) we train with a much smaller batch size, 2) we're using an inferior tokenizer, 3) our context length is shorter is shorter (512 tokens) and 4) likely other discrepancies I've not noticed. In the future I plan to continually improve on all of those points. I hope this repo can be of help to anyone who wants to get into designing & building novel architectures but doesn't have the compute to test a larger model on every single idea they have. I'm literally training these on the CPU of a 2019 iMac with 8gb of ram.

[![ERROR DISPLAYING IMAGE, CLICK HERE FOR VIDEO](https://img.youtube.com/vi/s9kQvDsWnbc/0.jpg)](https://www.youtube.com/watch?v=s9kQvDsWnbc)

This repo is part of a larger project of mine called [micro_model_sandbox]() that's basically a hub for all the novel model experiments I do, with the goal of facilitating easy comparison between the different models. Basically for each of those experiments I just use this very repo as a template to start editing, and then once I'm happy with the project (or if I've just abandoned it but it's moderately functional) I add it to the sandbox. If you end up using this repo as a template, feel free to contribute your project to the sandbox as well!

## getting started
1. clone the repository
2. `cd` to the folder
3. setup a virtual environment unless you're an agent of chaos
4. `pip install -r requirements.txt`
5. edit values in `config.py` to suit your liking. This might involve a lot of trial and error if you don't know what you're doing, either due to errors from incompatible parameter configurations or from going over your available vram amount. Checkout the config files for each already trained model to get an idea of what reasonable values look like
6. Hop into `train.py` and run every cell before the final one. There's a cell where if you set the `if` statement to `True` then you'll be able to visualize what the learning rate is going to look like over the course of training (which you determined over in `config.py`)
7. If you like the look of the model you trained, run that final cell to save it. I recommend going into `trained/` and changing the folder name if you didn't already do so when messing with the config since the default is just going to be the date & time that its training begun, which is ugly boring and confusing
8. If you ever want to just test out a model you've already made then hop on over into `inference.ipynb` and run all the cells.
9. If you've trained multiple models, you can compare them in `model_comparison.ipynb` as long as you remember to use the third cell to specify which models you want to compare. It'll look at loss curves over the course of training and teacher-forcing topk accuracy rate
10. This step could really go anywhere, but if you're trying to learn how transformers work then along with reading the code in `modules/` you can use `test_modules.ipynb` to visualize how the tensor shapes change.
11. If/when you become confident to mess with the actual code yourself and test out a novel architecture idea you've got, head on over into `modules/` and get to work. While you're doing this, make sure to use `LoggingModule` instead of `nn.module` and put `@log_io` before every class function you make so that you can use `test_modules.ipynb` for easy visualization/debugging. To see an example of one of my novel architecture edits, check out [Fractal-Head Attention]()
12. If/when you've got a novel transformer architecture edit up and working, send it over to the [micro model sandbox]()!

## file structure
- `modules/`: where all of the code for the actual model goes
    - `layer.py`: defines each residual connection layer of our GPT
    - `logging.py`: defines the `LoggingModule` class, a wrapper that you should use instead of pytorch's `nn.module` in order to facilitate easy demonstration of how tensor shapes change throughout a given module
    - `mlp.py`: a two-layer multi-layer perceptron with an optional gate and either [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [GeLU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html), or [SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) nonlinearities, all configurable in `config.py`. Adding more nonlinearities is also absurdly easy
    - `model.py`: the primary class for our GPT
    - `mqa.py`: [multi-query attention](https://arxiv.org/abs/1911.02150) with pre-computed [rotary positional encodings](https://arxiv.org/abs/2104.09864)
    - `norm.py`: a norm module with an optional affine layer that allows you to switch between [RMSNorm](https://arxiv.org/abs/1910.07467), [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) and [CosineNorm](https://arxiv.org/pdf/1702.05870) easily using a setting over in `config.py`. Adding different normalization methods is also absurdly easy
- `tokenizers/`: a folder where you store your tokenizers
    - `bpe_v1/`: a [byte-pair encoding](https://huggingface.co/learn/nlp-course/chapter6/5) tokenizer except I use the characters that show up in TinyStories instead of bytes. This is the one that gets used in all the models that are currently trained, but if you're training a new model and don't care about comparing it against the pre-existing ones then I recommend using `bpe_v2/`
        - `build.ipynb`: the notebook where i built my bpe tokenizers. My pairing rules could certainly be improved upon
        - `tokenizer.py`: an overly-simplistic and annoyingly inefficient tokenizer with bos & eos tokens, post-sequence padding, and a `display` function to help you visualize how a given string is broken down
        - `models/`
            - `{95, 128, 256, 512, 1024, 2048, 4096, 8192}.model`: different tokenizer sizes, each a subset of the next. the 95 one is character-wise tokenization
    - `bpe_v2`: a slightly updated version of the one above that uses GPT4's regex instead of my own shitty from-scratch rules. Not currently implemented in any models
        - `...`
- `trained/`
    - `customGPT_1m_{short_and_thicc, 5ft11_and_skinnyfat, tall_and_skinny}/`: a series of 1m parameter models designed to be compared against one another in terms of number of layers, MLP width, and number of attention heads. they're not large enough to get intelligible output
        - `model_config.json`: hyperparameters of the model
        - `model.pth`: weights of the model
        - `train_config.json`: hyperparameters of the training loop used
        - `log_data.csv`: a record of loss and a couple other key metrics over the course of training
    - `customGPT_2m_{RMSNorm, LayerNorm, CosineNorm}`: a series of 2m parameter models designed to be compared against one another in terms of their chosen normalization technique. At this size you start to see a little bit of coherence in terms of repeated objects/characters in the story.
        - `...`
    - `customGPT_3m_{GatedMLP, NotGatedMLP}`: a series of 3m parameter models designed to be compared against one another to shed light on whether it's better to use old-fashioned or gated MLPs. I was surprised at how clearly better the gated MLP did
        - `...`
    - `customGPT_4m_{GeGLU, SwiGLU}`: a series of 4m parameter models designed to be compared against one another to shed light on which of the two most common activation functions right now are better. Given that Google uses GeGLU, Meta uses SwiGLU, and the industry as a whole hasn't settled on one or the other, it was unsurprising to see these two perform similarly. At this size you start to occasionally see multiple sentences in a row almost making some sense.
        - `...`
- `inference.ipynb`: open this notebook if you just want to see the output of one of the models
- `model_comparison.ipynb`: open this notebook to compare different models against each other. includes loss curve plots and topk teacher-forcing accuracy rate
- `testing_modules.ipynb`: creates easy printouts that allow you to follow the progression of tensor shapes for demonstration & debugging purposes of all the modules in `model.py`. If you're building new modules for a novel architecture idea you have then this notebook will be of extreme value to you in debugging & visualization
- `train.ipynb`: open this notebook to train a new model
- `config.py`: all of the editable model and training settings
- `inference.py`: functions for performing inference used in multiple `.ipynb` files
- `model_comparison.py`: functions for comparing models used in `model_comparison.ipynb`
- `requirements.txt` - I should probably change this to only include the packages that are actually necessary and not be so strict on versions. The command I used to get this list is `pip freeze | grep -v " @ file://" > requirements.txt` and then I deleted the version numbers, lmk if you know of a better method
- `tools.py`: A variety of functions & classes that don't fit elsewhere and/or are used by more than one of the jupyter notebooks. I should prolly find a better way to organize these
- `train.py`: functions for training a model, used in `train.ipynb`

## definite eventual TODOs
- [ ] fix & enable batched inference
    - [ ] update `model_evaluation.ipynb`'s teacher-forcing topk analysis to get more accurate %'s using batches
- [x] build a better tokenizer
    - [ ] train new models with this better tokenizer in `tokenizers/bpe_v2/`
- [ ] go back and make sure model checkpointing is working. at one point it was but i've changed so much since then and haven't bothered using it so i'd bet it's broken
- [ ] setup training batches and attention mask to concatenate more than one sequence back to back when the stories are shorter than the model's maximum context length
- [ ] switch to comparing models according to their non-embedding parameters instead of total parameters

### potential future TODOs
- [ ] create `hyperparameter_search.ipynb` that knows to cancel a run if it's going over your available vram usage
    - [ ] add a more complicated (regression?) analysis to `model_comparison.ipynb` to help us analyze the hyperparameter search
- [ ] setup .py files to be runnable in terminal rather than in the .ipynb files
- [ ] switch to [TinyGrad](https://github.com/tinygrad/tinygrad)? very tempting bc then I could use apple silicon gpu (pytorch currently doesn't support complex numbers, which are used in [rotary positional encodings](https://arxiv.org/abs/2104.09864), on apple silicon) but I feel like it makes more sense to stick with pytorch for public accessibility reasons
- [ ] add option to continually train pre-existing models & update its training data/hyperparameters accordingly
- [ ] add automated model comparison analysis by GPT4 like in the [TinyStories](https://arxiv.org/abs/2305.07759) paper into `model_comparison.ipynb`
- [ ] add sparse/local attention mask options
- [ ] different architectures/modules to incorporate/add
    - [ ] Mixture of Experts
    - [ ] whatever these new RNN-like transformers have going on such as [Gemma 1.1](https://arxiv.org/abs/2402.19427)
    - [ ] [Mamba](https://arxiv.org/abs/2312.00752) blocks
- [ ] build an easy way to design blocks in residual layers using lists of strings in the config. for example, the parallel MoE from [Snowflake](https://www.snowflake.com/en/) would be
```Python
[
'Norm->MQA->+->Norm->MLP->+',
'Norm->MoE->+'
]
```

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
- guides on how to build miniature versions of popular models from scratch, with a hand-holding walkthrough of every single tensor operation: [minGemma](https://github.com/evintunador/minGemma), [minGrok](https://github.com/evintunador/minGrok), and [minLlama3](https://github.com/evintunador/minLlama3). Future versions of these guides will use this Repo as a template
- [my YouTube channel](https://www.youtube.com/@Tunadorable)
- my [other links](https://linktr.ee/tunadorable)
