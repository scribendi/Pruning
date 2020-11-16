"""Wrapper of AllenNLP model. Fixes errors based on model predictions"""
import logging
import os
import sys
from time import time

import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util

from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.seq2labels_model import Seq2Labels
from gector.wordpiece_indexer import PretrainedBertIndexer
from utils.helpers import PAD, UNK, get_target_sent_by_edits, START_TOKEN
import numpy as np

import torch.nn.utils.prune as prune

import torch.nn as nn
import copy

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


def get_weights_name(transformer_name, lowercase):
    if transformer_name == 'bert' and lowercase:
        return 'bert-base-uncased'
    if transformer_name == 'bert' and not lowercase:
        return 'bert-base-cased'
    # if transformer_name == 'distilbert':
    #     if not lowercase:
    #         print('Warning! This model was trained only on uncased sentences.')
    #     return 'distilbert-base-uncased'
    if transformer_name == 'distilbert' and lowercase:
        return 'distilbert-base-uncased'
    if transformer_name == 'distilbert' and not lowercase:
        return 'distilbert-base-cased'
    if transformer_name == 'distilgpt2':
        return 'distilgpt2'
    if transformer_name == 'albert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'albert-base-v1'
    if lowercase:
        print('Warning! This model was trained only on cased sentences.')
    if transformer_name == 'roberta':
        return 'roberta-base'
    if transformer_name == 'gpt2':
        return 'gpt2'
    if transformer_name == 'transformerxl':
        return 'transfo-xl-wt103'
    if transformer_name == 'xlnet':
        return 'xlnet-base-cased'

def get_token_embedders(model_name, tune_bert=0, special_tokens_fix=0):
    take_grads = True if tune_bert > 0 else False
    model_name = get_weights_name(model_name, False)
    bert_token_emb = PretrainedBertEmbedder(
        pretrained_model=model_name,
        top_layer_only=True, requires_grad=take_grads,
        special_tokens_fix=special_tokens_fix)
    # exit(0)

    token_embedders = {'bert': bert_token_emb}
    embedder_to_indexer_map = {"bert": ["bert", "bert-offsets"]}

    text_filed_emd = BasicTextFieldEmbedder(token_embedders=token_embedders,
                                            embedder_to_indexer_map=embedder_to_indexer_map,
                                            allow_unmatched_keys=True)
    return text_filed_emd

class GecBERTModel(object):
    def __init__(self, vocab_path=None, model_paths=None,
                 weigths=None,
                 max_len=50,
                 min_len=3,
                 lowercase_tokens=False,
                 log=False,
                 iterations=3,
                 min_probability=0.0,
                 model_name='roberta',
                 special_tokens_fix=1,
                 is_ensemble=True,
                 # is_ensemble=False,
                 min_error_probability=0.0,
                 confidence=0,
                 resolve_cycles=False,
                 prune_amount=0.,
                 num_layers_to_keep=12
                 ):
        # print('here')
        self.model_weights = list(map(float, weigths)) if weigths else [1] * len(model_paths)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.min_len = min_len
        self.lowercase_tokens = lowercase_tokens
        self.min_probability = min_probability
        self.min_error_probability = min_error_probability
        self.vocab = Vocabulary.from_files(vocab_path)
        self.log = log
        self.iterations = iterations
        self.confidence = confidence
        self.resolve_cycles = resolve_cycles
        # set training parameters and operations

        self.indexers = []
        self.models = []

        for model_path in model_paths:
            # print('model_path:', model_path); exit(0)
            if is_ensemble:
                model_name, special_tokens_fix = self._get_model_data(model_path)
            weights_name = get_weights_name(model_name, lowercase_tokens)
            self.indexers.append(self._get_indexer(weights_name, special_tokens_fix))
            # token_embs = get_token_embedders(model_name, tune_bert=1, special_tokens_fix=special_tokens_fix)

            model = Seq2Labels(vocab=self.vocab,
                               text_field_embedder=self._get_embbeder(weights_name, special_tokens_fix),
                               # text_field_embedder= token_embs,
                               confidence=self.confidence
                               ).to(self.device)
            # count number of params
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            print('total params:', pytorch_total_params)

            # print('model:', model)
            print('type:', type(model)); #exit(0)

            def print_size_of_model(model):
                torch.save(model.state_dict(), "temp.p")
                print('Size (MB):', os.path.getsize("temp.p") / 1e6)
                os.remove('temp.p')

            # delete top layers
            def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
                oldModuleList = model.text_field_embedder.token_embedder_bert.bert_model.encoder.layer
                # print('oldModuleList:', oldModuleList)
                # print('oldModuleList:', len(oldModuleList)); exit(0)

                newModuleList = nn.ModuleList()

                # Now iterate over all layers, only keeping only the relevant layers.
                for i in range(0, num_layers_to_keep):
                # for i in range(0, len(num_layers_to_keep)):
                    newModuleList.append(oldModuleList[i])

                # create a copy of the model, modify it with the new list, and return
                copyOfModel = copy.deepcopy(model)
                copyOfModel.text_field_embedder.token_embedder_bert.bert_model.encoder.layer = newModuleList

                return copyOfModel

            print('before model 12:')
            # model = deleteEncodingLayers(model, 12)
            # print('after 12:', model)
            print_size_of_model(model)
            #
            # print('before model:', model)
            # model = deleteEncodingLayers(model, 11)
            # print ('after 11:', model)
            # print_size_of_model(model)

            model = deleteEncodingLayers(model, num_layers_to_keep)
            print('after', num_layers_to_keep,':', model)
            print_size_of_model(model)

            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load(model_path,
                                                 map_location=torch.device('cpu')))
            # print('chk1'); exit(0)
            # get model size


            # print(model)
            print_size_of_model(model); #exit(0)
            print('type:', type(model)); #exit(0)


            model.eval()
            self.models.append(model)



    @staticmethod
    def _get_model_data(model_path):
        model_name = model_path.split('/')[-1]
        tr_model, stf = model_name.split('_')[:2]
        return tr_model, int(stf)

    def _restore_model(self, input_path):
        if os.path.isdir(input_path):
            print("Model could not be restored from directory", file=sys.stderr)
            filenames = []
        else:
            filenames = [input_path]
        for model_path in filenames:
            try:
                if torch.cuda.is_available():
                    loaded_model = torch.load(model_path)
                else:
                    loaded_model = torch.load(model_path,
                                              map_location=lambda storage,
                                                                  loc: storage)
            except:
                print(f"{model_path} is not valid model", file=sys.stderr)
            own_state = self.model.state_dict()
            for name, weights in loaded_model.items():
                if name not in own_state:
                    continue
                try:
                    if len(filenames) == 1:
                        own_state[name].copy_(weights)
                    else:
                        own_state[name] += weights
                except RuntimeError:
                    continue
        print("Model is restored", file=sys.stderr)

    def predict(self, batches):
        t11 = time()
        predictions = []
        # print('batches:', len(batches), 'models:', len(self.models)); exit(0);
        for batch, model in zip(batches, self.models):
            batch = util.move_to_device(batch.as_tensor_dict(), 0 if torch.cuda.is_available() else -1)
            # print('batch:', batch); #exit(0)
            # print('batch:', batch.keys());
            # print('batch:', batch['tokens'].keys())
            # print('bert:', batch['tokens']['bert'].shape, type(batch['tokens']['bert']))
            # print('bert-offsets:', batch['tokens']['bert-offsets'].shape, type(batch['tokens']['bert-offsets']))
            # print('mask:', batch['tokens']['mask'].shape, type(batch['tokens']['mask']))
            # exit(0)
            # def print_size_of_model(model):
            #     torch.save(model.state_dict(), "temp.p")
            #     print('Size (MB):', os.path.getsize("temp.p") / 1e6)
            #     os.remove('temp.p')
            #
            # # print(model)
            # print_size_of_model(model); exit(0);
            with torch.no_grad():
                prediction = model.forward(**batch)
            predictions.append(prediction)

        preds, idx, error_probs = self._convert(predictions)
        t55 = time()
        if self.log:
            print(f"Inference time {t55 - t11}")
        return preds, idx, error_probs

    def get_token_action(self, token, index, prob, sugg_token):
        """Get lost of suggested actions for token."""
        # cases when we don't need to do anything
        if prob < self.min_probability or sugg_token in [UNK, PAD, '$KEEP']:
            return None

        if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token == '$DELETE':
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
            start_pos = index + 1
            end_pos = index + 1

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob

    def _get_embbeder(self, weigths_name, special_tokens_fix):
        embedders = {'bert': PretrainedBertEmbedder(
            pretrained_model=weigths_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=special_tokens_fix)
        }
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders=embedders,
            embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]},
            allow_unmatched_keys=True)
        # print('text_field_embedder:', text_field_embedder.shape); exit(0)
        return text_field_embedder

    def _get_indexer(self, weights_name, special_tokens_fix):
        bert_token_indexer = PretrainedBertIndexer(
            pretrained_model=weights_name,
            do_lowercase=self.lowercase_tokens,
            max_pieces_per_token=5,
            use_starting_offsets=True,
            truncate_long_sequences=True,
            special_tokens_fix=special_tokens_fix,
            is_test=True
        )
        return {'bert': bert_token_indexer}

    def preprocess(self, token_batch):
        seq_lens = [len(sequence) for sequence in token_batch if sequence]
        if not seq_lens:
            return []
        max_len = min(max(seq_lens), self.max_len)
        batches = []
        for indexer in self.indexers:
            batch = []
            for sequence in token_batch:
                tokens = sequence[:max_len]
                tokens = [Token(token) for token in ['$START'] + tokens]
                batch.append(Instance({'tokens': TextField(tokens, indexer)}))
            batch = Batch(batch)
            batch.index_instances(self.vocab)
            batches.append(batch)

        return batches

    def _convert(self, data):
        all_class_probs = torch.zeros_like(data[0]['class_probabilities_labels'])
        error_probs = torch.zeros_like(data[0]['max_error_probability'])
        for output, weight in zip(data, self.model_weights):
            all_class_probs += weight * output['class_probabilities_labels'] / sum(self.model_weights)
            error_probs += weight * output['max_error_probability'] / sum(self.model_weights)

        max_vals = torch.max(all_class_probs, dim=-1)
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
        return probs, idx, error_probs.tolist()

    def update_final_batch(self, final_batch, pred_ids, pred_batch,
                           prev_preds_dict):
        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def postprocess_batch(self, batch, all_probabilities, all_idxs,
                          error_probs,
                          max_len=50):
        all_results = []
        noop_index = self.vocab.get_token_index("$KEEP", "labels")
        for tokens, probabilities, idxs, error_prob in zip(batch,
                                                           all_probabilities,
                                                           all_idxs,
                                                           error_probs):
            length = min(len(tokens), max_len)
            edits = []

            # skip whole sentences if there no errors
            if max(idxs) == 0:
                all_results.append(tokens)
                continue

            # skip whole sentence if probability of correctness is not high
            if error_prob < self.min_error_probability:
                all_results.append(tokens)
                continue

            for i in range(length + 1):
                # because of START token
                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]
                # skip if there is no error
                if idxs[i] == noop_index:
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i],
                                                             namespace='labels')
                action = self.get_token_action(token, i, probabilities[i],
                                               sugg_token)
                if not action:
                    continue

                edits.append(action)
            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results

    def handle_batch(self, full_batch):
        """
        Handle batch of requests.
        """
        # print('full_batch:', full_batch)
        final_batch = full_batch[:]
        # print('final_batch:', final_batch)
        batch_size = len(full_batch)
        # print('batch_size:', batch_size)
        prev_preds_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        # print('prev_preds_dict:', prev_preds_dict)
        short_ids = [i for i in range(len(full_batch))
                     if len(full_batch[i]) < self.min_len]
        # print('short_ids:', short_ids)
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        # print('pred_ids:', pred_ids); exit(0)
        total_updates = 0

        for n_iter in range(self.iterations):
            orig_batch = [final_batch[i] for i in pred_ids]

            sequences = self.preprocess(orig_batch)

            if not sequences:
                break
            probabilities, idxs, error_probs = self.predict(sequences)

            pred_batch = self.postprocess_batch(orig_batch, probabilities,
                                                idxs, error_probs)
            if self.log:
                print(f"Iteration {n_iter + 1}. Predicted {round(100*len(pred_ids)/batch_size, 1)}% of sentences.")

            final_batch, pred_ids, cnt = \
                self.update_final_batch(final_batch, pred_ids, pred_batch,
                                        prev_preds_dict)
            total_updates += cnt

            if not pred_ids:
                break

        return final_batch, total_updates
