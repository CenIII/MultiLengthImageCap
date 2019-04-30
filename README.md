# MultiLengthImageCap

## Installation
We use [DenseCap](https://github.com/jcjohnson/densecap) as our feature extractor, which was implemented in [Torch](http://torch.ch/), and depends on the following packages: [torch/torch7](https://github.com/torch/torch7), [torch/nn](https://github.com/torch/nn), [torch/nngraph](https://github.com/torch/nngraph), [torch/image](https://github.com/torch/image), [lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson), [qassemoquab/stnbhwd](https://github.com/qassemoquab/stnbhwd), [jcjohnson/torch-rnn](https://github.com/jcjohnson/torch-rnn)
After installing torch, you can install / update these dependencies by running the following:

```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson
luarocks install https://raw.githubusercontent.com/qassemoquab/stnbhwd/master/stnbhwd-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/torch-rnn-scm-1.rockspec
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
```

The LSTM encoder, LSTM decoder and language model are implemented in [Pytorch](https://pytorch.org/get-started/locally/). Please follow the install instructions on official website.  

## Pretrained model
### LSTM Encoder
To train the generator (LSTM decoder), you need to pretrain an LSTM encoder. For your convenience you can download a pretrained LSTM encoder in [here](https://drive.google.com/file/d/1OFbXUjr9SNc2mGP2YGFdKbr3jiV5uf1w/view?usp=sharing)

### LSTM Decoder
We also provide a pretrained LSTM decoder [here]()

## Dataset
To prepare the image dataset, please follow the following steps:
1. Download the raw images and region descriptions from [the Visual Genome website](https://visualgenome.org/api/v0/api_home.html)
2. Use the script `densecap/preprocess.py` to generate a single HDF5 file containing the entire dataset (\~135GB). 
   [(details here)](https://github.com/jcjohnson/densecap/blob/master/doc/FLAGS.md#preprocesspy)

## vocabulary data (word dictionary + word embedding)
Our LSTM encoder and decoder used shared vocabulary data containing a word dictionary (mapping from word to index) and a word embedding (mapping from index to embedding). We pack the two into one pickle file. Please download the pickle file from [here](https://drive.google.com/open?id=1Kdn8zhTKcYjlkD_UPXCalCfTOXULh_jH) and save them to `./data` folder. 

## pretrain LSTM encoder


## train LSTM decoder

## run demo
