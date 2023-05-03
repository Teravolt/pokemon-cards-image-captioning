# Pokemon Card Image Captioner

This repo contains code for a pet-project of mine for captioning Pokemon Cards.
I started this project for an MLOps course run by Weights and Biases, and will be making extensions to it.

## Installation

The Pokemon Card Image Captioner has been tested using Python `3.9`.
If you are able to get this running on an older version of Python, or the Pokemon Card Image Captioner fails to run on a later version, please open an issue and I will look into it.

You can set up your Python envrionment using any method (e.g., Poetry, pipenv, conda, etc.), but please
make sure you have **all** packages from `requirements.txt` installed. 

I have tested this with `pipenv` and included the relevant `pipenv` files.
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

TODO: Add training instructions


## Evaluation

TODO: Add evaluation instructions

