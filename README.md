# Pokemon Card Image Captioner

This repo contains code for a pet-project of mine for captioning Pokemon Cards.
I started this project for an MLOps course run by Weights and Biases, and will be making extensions to it.
I have included a Gradio demo of the captioner I trained for the MLOps course.

More specifically, the image captioner is a pre-trained [ViT-GPT2 image-to-text generation model](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning).
For more information about [ViT](https://arxiv.org/abs/2010.11929) and [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf), please see their respective papers.

## Installation

The Pokemon Card Image Captioner has been tested using Python `3.9`.
If you are able to get this running on an older version of Python, or the Pokemon Card Image Captioner fails to run on a later version, please open an issue and I will look into it.

You can set up your Python envrionment using any method (e.g., Poetry, pipenv, conda, etc.), but please
make sure you have **all** packages from `requirements.txt` installed. 

**NOTE**: I have tested this with `pipenv` and included the relevant `pipenv` files.
If you have success with other methods, please let me know and I can add instructions here!

Once you have all necessary packages installed, you can train, evaluate, and run a demo of the image captioner.

## Running Demo

I have set up a Gradio demo for those interested in simply running a demo of the image captioner.
To run the demo, make sure you have all relevant Python packages installed and run the following:

```
python app.py
```

This will start the app up at `http://127.0.0.1:7860`.
From there, you can upload images of Pokemon cards and the image captioner will attempt to caption it.

## Training

You can fine-tune the pretrained ViT-GPT2 image-to-text generation model with the following command:
```
python train.py
```
For a list of possible hyperparameters that can be tuned, you can add `-h` as an argument:
```
python train.py -h
```

## Evaluation

```
python eval.py
```

TODO: Write evaluation instructions

## Future Work

- Train on full training split. I was restricted to 256 due to training on a Macbook Air. With more compute power, I should be able to train on the full 1K subset I made for the MLOps course.