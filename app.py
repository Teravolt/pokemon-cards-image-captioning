"""
Gradio app for captioning Pokemon Cards
"""

from argparse import Namespace

import gradio as gr

from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer
from transformers import AutoFeatureExtractor

import torch

MODEL = VisionEncoderDecoderModel.from_pretrained('./artifacts/pokemon-image-captioning-model:v7')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(DEVICE)

# Define image feature extractor and tokenizer
# NOTE: these are not trained, so we can get them directly from HuggingFace
FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
TOKENIZER = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

SEED = 1

CONFIG = Namespace(
    predict_with_generate=True,
    include_inputs_for_metrics=False,
    report_to='wandb',
    run_name='fine_tuning_eval',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    learning_rate=1e-3,
    push_to_hub=False,
    load_best_model_at_end=True,
    seed=SEED,
    output_dir='eval-output/',
    optim='adamw_torch',
    generation_max_length=256,
    generation_num_beams=1,
    log_preds=False,
    val_limit=0
)

def generate_caption(input_img):
    """
    Predict text given an image
    """
    pixel_values = FEATURE_EXTRACTOR(images=[input_img], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(DEVICE)

    gen_kwargs = {
        "max_length": CONFIG.generation_max_length,
        "num_beams": CONFIG.generation_num_beams
        }

    output_ids = MODEL.generate(pixel_values, **gen_kwargs)

    preds = TOKENIZER.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]

demo = gr.Interface(fn=generate_caption, inputs='image', outputs='text')
demo.launch()
