"""
Train Pokemon Cards Image Captioning Model
"""

import argparse
from argparse import Namespace

import pandas as pd

# Package for loading model
from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer
from transformers import AutoFeatureExtractor

# Packages for training & evaluating model
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import EvalPrediction

# Evaluation metrics library
import evaluate

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import wandb

SEED = 1

# Define model
MODEL = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(DEVICE)

# Define image feature extractor and tokenizer
FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
TOKENIZER = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Define metrics
GOOGLE_BLEU_METRIC = evaluate.load('google_bleu')
BLEU_METRIC = evaluate.load('sacrebleu')
BERTSCORE_METRIC = evaluate.load('bertscore')

# Define validation/testing results table 
FULL_RESULTS_TABLE = wandb.Table(columns=['eval_iter', 'image', 'pred_text', 'gt_text', 'google_bleu'])
EVAL_ITER = 0

VAL_DF = None

CONFIG = Namespace(
    predict_with_generate=True,
    include_inputs_for_metrics=False,
    report_to='wandb',
    run_name='fine_tuning',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    learning_rate=1e-3,
    push_to_hub=False,
    load_best_model_at_end=True,
    seed=SEED,
    output_dir='baseline-ft-model-output/',
    optim='adamw_torch',
    generation_max_length=256,
    generation_num_beams=1,
    train_limit=0,
    val_limit=0,
    contrast_t_prime = 0, 
    contrast_bias = 0
)

def download_data(run):
    """
    Download data from wandb
    """
    
    split_data_loc = run.use_artifact('pokemon_cards_split_full:v0')
    table = split_data_loc.get(f"pokemon_table_full_data_split_seed_{SEED}")
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

        # self.images = []
        # for image in images:
        #     image_ = image.image
        #     if image_.mode != "RGB":
        #         image_ = image_.convert(mode="RGB")
        #     self.images.append(image_)
        self.images = images
        self.captions = captions
        self.config = config

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        
        image_ = self.images[index]
        image = image_.image
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
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

        FULL_RESULTS_TABLE.add_data(EVAL_ITER, VAL_DF['image'].values[i],
                                    pred_text, gt_text[0],
                                    google_bleu_metric['google_bleu'])

        avg_google_bleu.append(google_bleu_metric['google_bleu'])

    bleu_metric = \
        BLEU_METRIC.compute(predictions=pred_texts, references=gt_texts)

    metrics = {
        'avg_google_bleu': sum(avg_google_bleu)/len(avg_google_bleu),
        'bleu_metric': bleu_metric['score']}

    EVAL_ITER += 1

    return metrics

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


class ContrastLossTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):

        self.t_prime = kwargs['contrast_t_prime']
        self.t = torch.exp(torch.tensor(self.t_prime))
        self.bias = kwargs['contrast_bias']

        del kwargs['contrast_t_prime']
        del kwargs['contrast_bias']

        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):

        output = super().training_step(model, inputs)

        return output

    def compute_loss(self, model, inputs, return_outputs=False):
        """

        NOTE: This is based off the following paper: https://arxiv.org/pdf/2303.15343.pdf
        """

        inputs['output_hidden_states'] = True

        mle_loss, output = \
            super().compute_loss(model, inputs, return_outputs=True)

        # Get average of patch embeddings + embeddings from final layer
        encoder_embed = output.encoder_hidden_states[0] \
            + output.encoder_hidden_states[-1]
        batch_size = encoder_embed.shape[0]

        encoder_embed = encoder_embed.mean(dim=1)
        encoder_embed = F.normalize(encoder_embed, p=2, dim=1)

        # Get average of embeddings from final layer of decoder
        decoder_final_layer = output.decoder_hidden_states[-1]
        decoder_embed = decoder_final_layer.mean(dim=1)
        decoder_embed = F.normalize(decoder_embed, p=2, dim=1)

        # print(f"Encoder embeddings shape: {encoder_embed.shape}")
        # print(f"Decoder embeddings shape: {decoder_embed.shape}")

        sim = self.t*torch.matmul(decoder_embed, encoder_embed.transpose(1, 0)) \
            + self.bias
        # print(f"Similarity shape: {sim.shape}")

        labels = 2*torch.eye(batch_size) - torch.ones(batch_size)
        # print(f"Label shape: {labels.shape}")

        sigmoid_out = -F.logsigmoid(labels * sim).mean()

        # Interestingly, MLE loss looks to learn Pokemon types
        loss = mle_loss + sigmoid_out

        # Sample from the tokenizer and remove the current token
        # Context = embedding from previous tokens

        return (loss, output) if return_outputs else loss

def train(config):
    """
    Training process
    """
    global VAL_DF

    run = wandb.init(project='pokemon-cards', entity=None, job_type="training", name=config.run_name)
    wandb_table = download_data(run)
    train_val_df = get_df(wandb_table)

    train_df = train_val_df[train_val_df.split == 'train']
    VAL_DF = train_val_df[train_val_df.split == 'valid']

    if config.train_limit > 0:
        train_df = train_df.iloc[:config.train_limit, :]
    if config.val_limit > 0:
        VAL_DF = VAL_DF.iloc[:config.val_limit, :]

    train_dataset = PokemonCardsDataset(
        train_df.image.values,
        train_df.caption.values,
        config)

    val_dataset = PokemonCardsDataset(
        VAL_DF.image.values,
        VAL_DF.caption.values,
        config)

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
        metric_for_best_model="avg_google_bleu",
        load_best_model_at_end=config.load_best_model_at_end,
        seed=config.seed,
        output_dir=config.output_dir,
        optim=config.optim,
        generation_max_length=config.generation_max_length,
        generation_num_beams=config.generation_num_beams
        )

    trainer = ContrastLossTrainer(
        model=MODEL,
        contrast_t_prime=CONFIG.contrast_t_prime,
        contrast_bias=CONFIG.contrast_bias,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=FEATURE_EXTRACTOR,
        )

    train_results = trainer.train()

    if config.log_full_results:
        # Save full metrics table to wandb
        run.log({'full_results_table': FULL_RESULTS_TABLE})

    # Save final metrics table to wandb
    final_results_table = get_final_results(EVAL_ITER-1, FULL_RESULTS_TABLE)
    run.log({'final_results_table': final_results_table})

    if config.log_model:
        model_art = wandb.Artifact("pokemon-image-captioning-model", type="model")
        trainer.save_model(f"{config.output_dir}/best_model")
        model_art.add_dir(f"{config.output_dir}/best_model")
        run.log_artifact(model_art)

    run.finish()

    return train_results

def parse_args():
    """
    Parse args
    """
    parser = argparse.ArgumentParser("Train Pokemon Cards Image Captioning Model")
    parser.add_argument('--run_name',
                        type=str, default=CONFIG.run_name,
                        help='Run Name')
    parser.add_argument('--per_device_train_batch_size',
                        type=int, default=CONFIG.per_device_train_batch_size,
                        help='Per device training batch size')
    parser.add_argument('--per_device_eval_batch_size',
                        type=int, default=CONFIG.per_device_eval_batch_size,
                        help='Per device eval batch size')
    parser.add_argument('--num_train_epochs',
                        type=int, default=CONFIG.num_train_epochs,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate',
                        type=float, default=CONFIG.learning_rate,
                        help='Learning rate')
    parser.add_argument('--seed',
                        type=int, default=CONFIG.seed, help='Random seed')
    parser.add_argument('--output_dir',
                        type=str, default=CONFIG.output_dir,
                        help='Model output directory')
    parser.add_argument('--optim',
                        type=str, default=CONFIG.optim,
                        help='Optimizer to use')
    parser.add_argument('--generation_max_length',
                        type=int, default=CONFIG.generation_max_length,
                        help='Maximum length of generated text')
    parser.add_argument('--generation_num_beams',
                        type=int, default=CONFIG.generation_num_beams,
                        help='Number of beams used in text generation')
    parser.add_argument('--log_full_results', action='store_true',
                        help='Log eval results over all iterations')
    parser.add_argument('--log_model', action='store_true',
                        help='Log model to Weights and Biases')
    parser.add_argument('--train_limit',
                        type=int, default=CONFIG.train_limit,
                        help='Limit number of training instances used')
    parser.add_argument('--val_limit',
                        type=int, default=CONFIG.val_limit,
                        help='Limit number of validation instances used')
    args = parser.parse_args()

    vars(CONFIG).update(vars(args))
    return

if __name__ == '__main__':
    parse_args()
    train(CONFIG)