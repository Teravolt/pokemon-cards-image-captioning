"""
Evaluate Pokemon Cards Image Captioning Model
"""

import argparse
from argparse import Namespace

import pandas as pd

from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer
from transformers import AutoFeatureExtractor

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import EvalPrediction

import evaluate

import torch
from torch.utils.data import Dataset

import wandb

SEED = 1

MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image feature extractor and tokenizer
# NOTE: these are not trained, so we can get them directly from HuggingFace
FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
TOKENIZER = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Define metrics
GOOGLE_BLEU_METRIC = evaluate.load('google_bleu')
# PERPLEXITY = evaluate.load('perplexity', module_type='metric')

# Define validation/testing results table 
FULL_RESULTS_TABLE = wandb.Table(columns=['eval_iter', 'image', 'pred_text', 'gt_text', 'google_bleu'])
EVAL_ITER = 0

EVAL_DF = None

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

def download_data(run):
    """
    Download data from wandb
    """
    
    split_data_loc = run.use_artifact('pokemon_cards_split:latest')
    table = split_data_loc.get(f"pokemon_table_1k_data_split_seed_{SEED}")
    return table

def get_df(table, is_test=False):
    """
    Get dataframe from wandb table
    """
    dataframe = pd.DataFrame(data=table.data, columns=table.columns)

    if is_test:
        test_df = dataframe[dataframe.split == 'test']
        return test_df

    train_val_df = dataframe[dataframe.split != 'test']
    return train_val_df

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch], dim=0),
        'labels': torch.stack([x['labels'] for x in batch], dim=0)
    }

class PokemonCardsDataset(Dataset):

    def __init__(self, images:list, captions: list, config) -> None:

        self.images = []
        for image in images:
            image_ = image.image
            if image_.mode != "RGB":
                image_ = image_.convert(mode="RGB")
            self.images.append(image_)

        self.captions = captions
        self.config = config

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        
        image = self.images[index]
        caption = self.captions[index]

        pixel_values = FEATURE_EXTRACTOR(images=image, return_tensors="pt").pixel_values[0]
        tokenized_caption = TOKENIZER.encode(
            caption, return_tensors='pt', padding='max_length',
            truncation='longest_first',
            max_length=self.config.generation_max_length)[0]

        output = {
            'pixel_values': pixel_values,
            'labels': tokenized_caption
            }

        return output

def compute_metrics(eval_obj: EvalPrediction):
    global EVAL_ITER

    pred_ids = eval_obj.predictions
    gt_ids = eval_obj.label_ids

    pred_texts = TOKENIZER.batch_decode(pred_ids, skip_special_tokens=True)
    pred_texts = [text.strip() for text in pred_texts]

    gt_texts = TOKENIZER.batch_decode(gt_ids, skip_special_tokens=True)
    gt_texts = [[text.strip()] for text in gt_texts]

    avg_google_bleu = []
    for i, (pred_text, gt_text) in enumerate(zip(pred_texts, gt_texts)):
        # Compute Google BLEU metric
        # print(f"Prediction {i}: {pred_text}")
        # print(f"Ground truth {i}: {gt_text}")

        google_bleu_metric = \
            GOOGLE_BLEU_METRIC.compute(predictions=[pred_text], references=[gt_text])

        FULL_RESULTS_TABLE.add_data(EVAL_ITER, EVAL_DF['image'].values[i],
                                    pred_text, gt_text[0],
                                    google_bleu_metric['google_bleu'])

        avg_google_bleu.append(google_bleu_metric['google_bleu'])

    avg_google_bleu = {'avg_google_bleu': sum(avg_google_bleu)/len(avg_google_bleu)}
    EVAL_ITER += 1

    return avg_google_bleu

def get_final_results(final_val_iter: int, full_results_table: wandb.Table):
    """
    Get final results
    """

    final_results_table = wandb.Table(
        columns=['val_iter', 'image', 'pred_text', 'gt_text', 'google_bleu'])

    for result in full_results_table.data:
        if result[0] == final_val_iter:
            final_results_table.add_data(*result)

    return final_results_table

def eval(config):
    """
    Evaluation process
    """
    global EVAL_DF, MODEL

    run = wandb.init(project='pokemon-cards', entity=None, job_type="eval", name=config.run_name)

    artifact = run.use_artifact('pkthunder/model-registry/Model Pokemon Cards Image Captioning:v1', type='model')
    artifact_dir = artifact.download()
    # producer_run = artifact.logged_by()

    MODEL = VisionEncoderDecoderModel.from_pretrained(artifact_dir)
    MODEL.to(DEVICE)

    wandb_table = download_data(run)

    # Use validation dataset to ensure that we are using the correct model.
    train_val_df = get_df(wandb_table)

    EVAL_DF = train_val_df[train_val_df.split == 'valid']

    if config.val_limit > 0:
        EVAL_DF = EVAL_DF.iloc[:config.val_limit, :]

    val_dataset = PokemonCardsDataset(
        EVAL_DF.image.values,
        EVAL_DF.caption.values,
        config)

    # Get evaluation data and run model
    # EVAL_DF = get_df(wandb_table, True)
    # eval_dataset = PokemonCardsDataset(
    #     EVAL_DF.image.values,
    #     EVAL_DF.caption.values,
    #     config)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=config.predict_with_generate,
        include_inputs_for_metrics=config.include_inputs_for_metrics,
        report_to=config.report_to,
        run_name=config.run_name,
        evaluation_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        push_to_hub=config.push_to_hub,
        load_best_model_at_end=config.load_best_model_at_end,
        seed=config.seed,
        output_dir=config.output_dir,
        optim=config.optim,
        generation_max_length=config.generation_max_length,
        generation_num_beams=config.generation_num_beams
        )

    trainer = Seq2SeqTrainer(
        model=MODEL,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        tokenizer=FEATURE_EXTRACTOR,
        )

    eval_results = trainer.evaluate(
        val_dataset,
        max_length=config.generation_max_length,
        num_beams=config.generation_num_beams)

    if config.log_full_results:
        # Save full metrics table to wandb
        run.log({'full_results_table': FULL_RESULTS_TABLE})

    # Save final metrics table to wandb
    final_results_table = get_final_results(EVAL_ITER-1, FULL_RESULTS_TABLE)
    run.log({'final_results_table': final_results_table})

    run.finish()

    return eval_results

def parse_args():
    """
    Parse args
    """
    parser = argparse.ArgumentParser("Evaluate Pokemon Cards Image Captioning Model")
    parser.add_argument('--run_name',
                        type=str, default=CONFIG.run_name,
                        help='Run Name')
    parser.add_argument('--per_device_eval_batch_size',
                        type=int, default=CONFIG.per_device_eval_batch_size,
                        help='Per device eval batch size')
    parser.add_argument('--seed',
                        type=int, default=CONFIG.seed, help='Random seed')
    parser.add_argument('--generation_max_length',
                        type=int, default=CONFIG.generation_max_length,
                        help='Maximum length of generated text')
    parser.add_argument('--generation_num_beams',
                        type=int, default=CONFIG.generation_num_beams,
                        help='Number of beams used in text generation')
    parser.add_argument('--log_full_results', action='store_true',
                        help='Log eval results over all iterations')
    parser.add_argument('--val_limit',
                        type=int, default=CONFIG.val_limit,
                        help='Limit number of validation instances used')
    args = parser.parse_args()

    vars(CONFIG).update(vars(args))
    return

if __name__ == '__main__':
    parse_args()
    eval(CONFIG)