import copy
import math
import re
from collections import defaultdict
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from overrides import overrides
from pycocoevalcap.bleu.bleu_scorer import BleuScorer

from news_caption_analytic.modules.criteria import Criterion

#from .decoder_flattened import Decoder
from .decoder_faces_objects import Decoder
from .resnet import resnet152


@Model.register("transformer_faces_objects")
class TransformerFacesObjectModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 decoder: Decoder,
                 criterion: Criterion,
                 evaluate_mode: bool = False,
                 attention_dim: int = 1024,
                 hidden_size: int = 1024,
                 dropout: float = 0.1,
                 vocab_size: int = 50264,
                 model_name: str = 'roberta-base',
                 namespace: str = 'bpe',
                 index: str = 'roberta',
                 padding_value: int = 1,
                 use_context: bool = True,
                 sampling_topk: int = 1,
                 sampling_temp: float = 1.0,
                 weigh_bert: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        self.decoder = decoder
        self.criterion = criterion

        self.index = index
        self.namespace = namespace
        self.resnet = resnet152()
        self.roberta = torch.hub.load(
            'pytorch/fairseq:2f7e3f3323', 'roberta.large')
        self.use_context = use_context
        self.padding_idx = padding_value
        self.evaluate_mode = evaluate_mode
        self.sampling_topk = sampling_topk
        self.sampling_temp = sampling_temp
        self.weigh_bert = weigh_bert
        if weigh_bert:
            self.bert_weight = nn.Parameter(torch.Tensor(25))
            nn.init.uniform_(self.bert_weight)

        self.n_batches = 0
        self.n_samples = 0
        self.sample_history: Dict[str, float] = defaultdict(float)

        initializer(self)

    def forward(self,  # type: ignore
                context: Dict[str, torch.LongTensor],
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                face_embeds: torch.Tensor,
                obj_embeds: torch.Tensor,
                metadata: List[Dict[str, Any]],
                names: Dict[str, torch.LongTensor] = None,
                attn_idx=None) -> Dict[str, torch.Tensor]:

        caption_ids, target_ids, contexts = self._forward(
            context, image, caption, face_embeds, obj_embeds)
        decoder_out = self.decoder(caption, contexts)

        # Assume we're using adaptive loss
        loss, sample_size = self.criterion(
            self.decoder.adaptive_softmax, decoder_out, target_ids)

        loss = loss / math.log(2)

        output_dict = {
            'loss': loss / sample_size,
            'sample_size': sample_size,
        }

        # During evaluation, we will generate a caption and compute BLEU, etc.
        if not self.training and self.evaluate_mode:
            _, gen_ids, attns = self._generate(caption_ids, contexts, attn_idx)
            # We ignore <s> and <pad>
            gen_texts = [self.roberta.decode(x[x > 1]) for x in gen_ids.cpu()]
            captions = [m['caption'] for m in metadata]

            output_dict['captions'] = captions
            output_dict['generations'] = gen_texts
            output_dict['metadata'] = metadata
            output_dict['attns'] = attns
            output_dict['gen_ids'] = gen_ids.cpu().detach().numpy()

            # Remove punctuation
            gen_texts = [re.sub(r'[^\w\s]', '', t) for t in gen_texts]
            captions = [re.sub(r'[^\w\s]', '', t) for t in captions]

            for gen, ref in zip(gen_texts, captions):
                bleu_scorer = BleuScorer(n=4)
                bleu_scorer += (gen, [ref])
                score, _ = bleu_scorer.compute_score(option='closest')
                self.sample_history['bleu-1'] += score[0] * 100
                self.sample_history['bleu-2'] += score[1] * 100
                self.sample_history['bleu-3'] += score[2] * 100
                self.sample_history['bleu-4'] += score[3] * 100

                # rogue_scorer = Rouge()
                # score = rogue_scorer.calc_score([gen], [ref])
                # self.sample_history['rogue'] += score * 100

            if 'rare_tokens' in caption:
                for gen, ref, rare_list in zip(gen_texts, captions, caption['rare_tokens']):
                    bleu_scorer = BleuScorer(n=4)
                    rare_words = ' '.join(rare_list)
                    gen = gen + ' ' + rare_words

                    if rare_words:
                        print(ref)
                        print(gen)
                        print()

                    bleu_scorer += (gen, [ref])
                    score, _ = bleu_scorer.compute_score(option='closest')
                    self.sample_history['bleu-1r'] += score[0] * 100

        self.n_samples += caption_ids.shape[0]
        self.n_batches += 1

        return output_dict
    def generate_caption(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 face_embeds,
                 obj_embeds,
                 metadata: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:

        B = image.shape[0]
        caption = {self.index: context[self.index].new_zeros(B, 2)}
        caption_ids, _, contexts = self._forward(
            context, image, caption, face_embeds, obj_embeds)

        _, gen_ids, attns = self._generate(caption_ids, contexts)
        gen_texts = [self.roberta.decode(x[x > 1]) for x in gen_ids.cpu()]
        gen_ids = gen_ids.cpu().numpy().tolist()

        attns_list: List[List[Dict[str, Any]]] = []

        for i, token_ids in enumerate(gen_ids):
            # Let's process the article text
            article_ids = context[self.index][i]
            article_ids = article_ids[article_ids != self.padding_idx]
            article_ids = article_ids.cpu().numpy()
            # article_ids.shape == [seq_len]

            # remove <s>
            if article_ids[0] == self.roberta.task.source_dictionary.bos():
                article_ids = article_ids[1:]

             # Ignore final </s> token
            if article_ids[-1] == self.roberta.task.source_dictionary.eos():
                article_ids = article_ids[:-1]

            # Sanity check. We plus three because we removed <s>, </s> and
            # the last two attention scores are for no attention and bias
            assert article_ids.shape[0] == attns[0][0]['article'][i][0].shape[0] - 4

            byte_ids = [int(self.roberta.task.source_dictionary[k])
                        for k in article_ids]
            # e.g. [16012, 17163, 447, 247, 82, 4640, 3437]

            byte_strs = [self.roberta.bpe.bpe.decoder.get(token, token)
                         for token in byte_ids]
            # e.g. ['Sun', 'rise', 'âĢ', 'Ļ', 's', 'Ġexecutive', 'Ġdirector']

            merged_article = []
            article_mask = []
            cursor = 0
            a: Dict[str, Any] = {}
            newline = False
            for j, b in enumerate(byte_strs):
                # Start a new word
                if j == 0 or b[0] == 'Ġ' or b[0] == 'Ċ' or newline:
                    if a:
                        byte_text = ''.join(a['tokens'])
                        a['text'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                            'utf-8', errors=self.roberta.bpe.bpe.errors)
                        merged_article.append(a)
                        cursor += 1
                    # Note that
                    #   len(attns) == generation_length
                    #   len(attns[j]) == n_layers
                    #   attns[j][l] is a dictionary
                    #   attns[j][l]['article'].shape == [batch_size, target_len, source_len]
                    #   target_len == 1 since we generate one word at a time
                    a = {'tokens': [b]}
                    article_mask.append(cursor)
                    newline = b[0] == 'Ċ'
                else:
                    a['tokens'].append(b)
                    article_mask.append(cursor)
            byte_text = ''.join(a['tokens'])
            a['text'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                'utf-8', errors=self.roberta.bpe.bpe.errors)
            merged_article.append(a)

            # Next let's process the caption text
            attn_dicts: List[Dict[str, Any]] = []
            # Ignore seed input <s>
            if token_ids[0] == self.roberta.task.source_dictionary.bos():
                token_ids = token_ids[1:]  # remove <s>
            # Now len(token_ids) should be the same of len(attns)

            assert len(attns) == len(token_ids)

            # Ignore final </s> token
            if token_ids[-1] == self.roberta.task.source_dictionary.eos():
                token_ids = token_ids[:-1]
            # Now len(token_ids) should be len(attns) - 1

            byte_ids = [int(self.roberta.task.source_dictionary[k])
                        for k in token_ids]
            # e.g. [16012, 17163, 447, 247, 82, 4640, 3437]

            byte_strs = [self.roberta.bpe.bpe.decoder.get(token, token)
                         for token in byte_ids]
            # e.g. ['Sun', 'rise', 'âĢ', 'Ļ', 's', 'Ġexecutive', 'Ġdirector']

            # Merge by space
            a: Dict[str, Any] = {}
            for j, b in enumerate(byte_strs):
                # Clean up article attention
                article_attns = copy.deepcopy(merged_article)
                start = 0
                for word in article_attns:
                    end = start + len(word['tokens'])
                    layer_attns = []
                    for layer in range(len(attns[j])):
                        layer_attns.append(
                            attns[j][layer]['article'][i][0][start:end].mean())
                    word['attns'] = layer_attns
                    start = end
                    del word['tokens']

                # Start a new word. Ġ is space
                if j == 0 or b[0] == 'Ġ':
                    if a:
                        for l in range(len(a['attns']['image'])):
                            for modal in ['image', 'faces', 'obj']:
                                a['attns'][modal][l] /= len(a['tokens'])
                                a['attns'][modal][l] = a['attns'][modal][l].tolist()
                            for word in a['attns']['article']:
                                word['attns'][l] /= len(a['tokens'])
                                word['attns'][l] = word['attns'][l].tolist()
                        byte_text = ''.join(a['tokens'])
                        a['tokens'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                            'utf-8', errors=self.roberta.bpe.bpe.errors)
                        attn_dicts.append(a)
                    # Note that
                    #   len(attns) == generation_length
                    #   len(attns[j]) == n_layers
                    #   attns[j][l] is a dictionary
                    #   attns[j][l]['article'].shape == [batch_size, target_len, source_len]
                    #   target_len == 1 since we generate one word at a time
                    a = {
                        'tokens': [b],
                        'attns': {
                            'article': article_attns,
                            'image': [attns[j][l]['image'][i][0] for l in range(len(attns[j]))],
                            'faces': [attns[j][l]['faces'][i][0] for l in range(len(attns[j]))],
                            'obj': [attns[j][l]['obj'][i][0] for l in range(len(attns[j]))],
                        }
                    }
                else:
                    a['tokens'].append(b)
                    for l in range(len(a['attns']['image'])):
                        for modal in ['image', 'faces', 'obj']:
                            a['attns'][modal][l] += attns[j][l][modal][i][0]
                        for w, word in enumerate(a['attns']['article']):
                            word['attns'][l] += article_attns[w]['attns'][l]

            for l in range(len(a['attns']['image'])):
                for modal in ['image', 'faces', 'obj']:
                    a['attns'][modal][l] /= len(a['tokens'])
                    a['attns'][modal][l] = a['attns'][modal][l].tolist()
                for word in a['attns']['article']:
                    word['attns'][l] /= len(a['tokens'])
                    word['attns'][l] = word['attns'][l].tolist()
            byte_text = ''.join(a['tokens'])
            a['tokens'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                'utf-8', errors=self.roberta.bpe.bpe.errors)
            attn_dicts.append(a)

            attns_list.append(attn_dicts)

            # gen_texts = [self.roberta.decode(
            #     x[x != self.padding_idx]) for x in gen_ids]

        return attns_list, gen_texts

    def generate(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 face_embeds,
                 obj_embeds,
                 metadata: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:

        B = image.shape[0]
        caption = {self.index: context[self.index].new_zeros(B, 2)}
        caption_ids, _, contexts = self._forward(
            context, image, caption, face_embeds, obj_embeds)

        _, gen_ids, attns = self._generate(caption_ids, contexts)

        gen_ids = gen_ids.cpu().numpy().tolist()
        attns_list: List[List[Dict[str, Any]]] = []

        for i, token_ids in enumerate(gen_ids):
            # Let's process the article text
            article_ids = context[self.index][i]
            article_ids = article_ids[article_ids != self.padding_idx]
            article_ids = article_ids.cpu().numpy()
            # article_ids.shape == [seq_len]

            # remove <s>
            if article_ids[0] == self.roberta.task.source_dictionary.bos():
                article_ids = article_ids[1:]

             # Ignore final </s> token
            if article_ids[-1] == self.roberta.task.source_dictionary.eos():
                article_ids = article_ids[:-1]

            # Sanity check. We plus three because we removed <s>, </s> and
            # the last two attention scores are for no attention and bias
            assert article_ids.shape[0] == attns[0][0]['article'][i][0].shape[0] - 4

            byte_ids = [int(self.roberta.task.source_dictionary[k])
                        for k in article_ids]
            # e.g. [16012, 17163, 447, 247, 82, 4640, 3437]

            byte_strs = [self.roberta.bpe.bpe.decoder.get(token, token)
                         for token in byte_ids]
            # e.g. ['Sun', 'rise', 'âĢ', 'Ļ', 's', 'Ġexecutive', 'Ġdirector']

            merged_article = []
            article_mask = []
            cursor = 0
            a: Dict[str, Any] = {}
            newline = False
            for j, b in enumerate(byte_strs):
                # Start a new word
                if j == 0 or b[0] == 'Ġ' or b[0] == 'Ċ' or newline:
                    if a:
                        byte_text = ''.join(a['tokens'])
                        a['text'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                            'utf-8', errors=self.roberta.bpe.bpe.errors)
                        merged_article.append(a)
                        cursor += 1
                    # Note that
                    #   len(attns) == generation_length
                    #   len(attns[j]) == n_layers
                    #   attns[j][l] is a dictionary
                    #   attns[j][l]['article'].shape == [batch_size, target_len, source_len]
                    #   target_len == 1 since we generate one word at a time
                    a = {'tokens': [b]}
                    article_mask.append(cursor)
                    newline = b[0] == 'Ċ'
                else:
                    a['tokens'].append(b)
                    article_mask.append(cursor)
            byte_text = ''.join(a['tokens'])
            a['text'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                'utf-8', errors=self.roberta.bpe.bpe.errors)
            merged_article.append(a)

            # Next let's process the caption text
            attn_dicts: List[Dict[str, Any]] = []
            # Ignore seed input <s>
            if token_ids[0] == self.roberta.task.source_dictionary.bos():
                token_ids = token_ids[1:]  # remove <s>
            # Now len(token_ids) should be the same of len(attns)

            assert len(attns) == len(token_ids)

            # Ignore final </s> token
            if token_ids[-1] == self.roberta.task.source_dictionary.eos():
                token_ids = token_ids[:-1]
            # Now len(token_ids) should be len(attns) - 1

            byte_ids = [int(self.roberta.task.source_dictionary[k])
                        for k in token_ids]
            # e.g. [16012, 17163, 447, 247, 82, 4640, 3437]

            byte_strs = [self.roberta.bpe.bpe.decoder.get(token, token)
                         for token in byte_ids]
            # e.g. ['Sun', 'rise', 'âĢ', 'Ļ', 's', 'Ġexecutive', 'Ġdirector']

            # Merge by space
            a: Dict[str, Any] = {}
            for j, b in enumerate(byte_strs):
                # Clean up article attention
                article_attns = copy.deepcopy(merged_article)
                start = 0
                for word in article_attns:
                    end = start + len(word['tokens'])
                    layer_attns = []
                    for layer in range(len(attns[j])):
                        layer_attns.append(
                            attns[j][layer]['article'][i][0][start:end].mean())
                    word['attns'] = layer_attns
                    start = end
                    del word['tokens']

                # Start a new word. Ġ is space
                if j == 0 or b[0] == 'Ġ':
                    if a:
                        for l in range(len(a['attns']['image'])):
                            for modal in ['image', 'faces', 'obj']:
                                a['attns'][modal][l] /= len(a['tokens'])
                                a['attns'][modal][l] = a['attns'][modal][l].tolist()
                            for word in a['attns']['article']:
                                word['attns'][l] /= len(a['tokens'])
                                word['attns'][l] = word['attns'][l].tolist()
                        byte_text = ''.join(a['tokens'])
                        a['tokens'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                            'utf-8', errors=self.roberta.bpe.bpe.errors)
                        attn_dicts.append(a)
                    # Note that
                    #   len(attns) == generation_length
                    #   len(attns[j]) == n_layers
                    #   attns[j][l] is a dictionary
                    #   attns[j][l]['article'].shape == [batch_size, target_len, source_len]
                    #   target_len == 1 since we generate one word at a time
                    a = {
                        'tokens': [b],
                        'attns': {
                            'article': article_attns,
                            'image': [attns[j][l]['image'][i][0] for l in range(len(attns[j]))],
                            'faces': [attns[j][l]['faces'][i][0] for l in range(len(attns[j]))],
                            'obj': [attns[j][l]['obj'][i][0] for l in range(len(attns[j]))],
                        }
                    }
                else:
                    a['tokens'].append(b)
                    for l in range(len(a['attns']['image'])):
                        for modal in ['image', 'faces', 'obj']:
                            a['attns'][modal][l] += attns[j][l][modal][i][0]
                        for w, word in enumerate(a['attns']['article']):
                            word['attns'][l] += article_attns[w]['attns'][l]

            for l in range(len(a['attns']['image'])):
                for modal in ['image', 'faces', 'obj']:
                    a['attns'][modal][l] /= len(a['tokens'])
                    a['attns'][modal][l] = a['attns'][modal][l].tolist()
                for word in a['attns']['article']:
                    word['attns'][l] /= len(a['tokens'])
                    word['attns'][l] = word['attns'][l].tolist()
            byte_text = ''.join(a['tokens'])
            a['tokens'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                'utf-8', errors=self.roberta.bpe.bpe.errors)
            attn_dicts.append(a)

            attns_list.append(attn_dicts)

            # gen_texts = [self.roberta.decode(
            #     x[x != self.padding_idx]) for x in gen_ids]

        return attns_list

    def _forward(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 caption: Dict[str, torch.LongTensor],
                 face_embeds,
                 obj_embeds):

        # We assume that the first token in target is the <s> token. We
        # shall use it to seed the decoder. Here decoder_target is simply
        # decoder_input but shifted to the right by one step.
        caption_ids = caption[self.index]
        target_ids = torch.zeros_like(caption_ids)
        target_ids[:, :-1] = caption_ids[:, 1:]

        # The final token is not used as input to the decoder, since otherwise
        # we'll be predicting the <pad> token.
        caption_ids = caption_ids[:, :-1]
        target_ids = target_ids[:, :-1]
        caption[self.index] = caption_ids

        # Embed the image
        X_image = self.resnet(image)
        # X_image.shape == [batch_size, 2048, 7, 7]

        X_image = X_image.permute(0, 2, 3, 1)
        # X_image.shape == [batch_size, 7, 7, 2048]

        # Flatten out the image
        B, H, W, C = X_image.shape
        P = H * W  # number of pixels
        X_image = X_image.view(B, P, C)
        # X_image.shape == [batch_size, 49, 2048]

        article_ids = context[self.index]
        # article_ids.shape == [batch_size, seq_len]

        article_padding_mask = article_ids == self.padding_idx
        # article_padding_mask.shape == [batch_size, seq_len]

        B, S = article_ids.shape

        X_sections_hiddens = self.roberta.extract_features(
            article_ids, return_all_hiddens=True)

        if self.weigh_bert:
            X_article = torch.stack(X_sections_hiddens, dim=2)
            # X_article.shape == [batch_size, seq_len, 13, embed_size]

            weight = F.softmax(self.bert_weight, dim=0)
            weight = weight.unsqueeze(0).unsqueeze(1).unsqueeze(3)
            # weight.shape == [1, 1, 13, 1]

            X_article = (X_article * weight).sum(dim=2)
            # X_article.shape == [batch_size, seq_len, embed_size]

        else:
            X_article = X_sections_hiddens[-1]
            # X_article.shape == [batch_size, seq_len, embed_size]

        # Create padding mask (1 corresponds to the padding index)
        image_padding_mask = X_image.new_zeros(B, P).bool()

        # face_embeds.shape == [batch_size, n_faces, 512]
        face_masks = torch.isnan(face_embeds).any(dim=-1)
        face_embeds[face_masks] = 0

        # obj_embeds.shape == [batch_size, n_objects, 1024]
        obj_masks = torch.isnan(obj_embeds).any(dim=-1)
        obj_embeds[obj_masks] = 0

        # The quirks of dynamic convolution implementation: The context
        # embedding has dimension [seq_len, batch_size], but the mask has
        # dimension [batch_size, seq_len].
        contexts = {
            'image': X_image.transpose(0, 1),
            'image_mask': image_padding_mask,
            'article': X_article.transpose(0, 1),
            'article_mask': article_padding_mask,
            'sections': None,
            'sections_mask': None,
            'faces': face_embeds.transpose(0, 1),
            'faces_mask': face_masks,
            'obj': obj_embeds.transpose(0, 1),
            'obj_mask': obj_masks,
        }

        return caption_ids, target_ids, contexts

    def _generate(self, caption_ids, contexts, attn_idx=None):
        incremental_state: Dict[str, Any] = {}
        seed_input = caption_ids[:, 0:1]
        log_prob_list = []
        index_path_list = [seed_input]
        eos = 2
        active_idx = seed_input[:, -1] != eos
        full_active_idx = active_idx
        gen_len = 100
        B = caption_ids.shape[0]
        attns = []

        for i in range(gen_len):
            if i == 0:
                prev_target = {self.index: seed_input}
            else:
                prev_target = {self.index: seed_input[:, -1:]}

            self.decoder.filter_incremental_state(
                incremental_state, active_idx)

            contexts_i = {
                'image': contexts['image'][:, full_active_idx],
                'image_mask': contexts['image_mask'][full_active_idx],
                'article': contexts['article'][:, full_active_idx],
                'article_mask': contexts['article_mask'][full_active_idx],
                'faces': contexts['faces'][:, full_active_idx],
                'faces_mask': contexts['faces_mask'][full_active_idx],
                'obj': contexts['obj'][:, full_active_idx],
                'obj_mask': contexts['obj_mask'][full_active_idx],
                'sections':  None,
                'sections_mask': None,
            }

            decoder_out = self.decoder(
                prev_target,
                contexts_i,
                incremental_state=incremental_state)

            attns.append(decoder_out[1]['attn'])

            # We're only interested in the current final word
            decoder_out = (decoder_out[0][:, -1:], None)

            lprobs = self.decoder.get_normalized_probs(
                decoder_out, log_probs=True)
            # lprobs.shape == [batch_size, 1, vocab_size]

            lprobs = lprobs.squeeze(1)
            # lprobs.shape == [batch_size, vocab_size]

            topk_lprobs, topk_indices = lprobs.topk(self.sampling_topk)
            topk_lprobs = topk_lprobs.div_(self.sampling_temp)
            # topk_lprobs.shape == [batch_size, topk]

            # Take a random sample from those top k
            topk_probs = topk_lprobs.exp()
            sampled_index = torch.multinomial(topk_probs, num_samples=1)
            # sampled_index.shape == [batch_size, 1]

            selected_lprob = topk_lprobs.gather(
                dim=-1, index=sampled_index)
            # selected_prob.shape == [batch_size, 1]

            selected_index = topk_indices.gather(
                dim=-1, index=sampled_index)
            # selected_index.shape == [batch_size, 1]

            log_prob = selected_lprob.new_zeros(B, 1)
            log_prob[full_active_idx] = selected_lprob

            index_path = selected_index.new_full((B, 1), self.padding_idx)
            index_path[full_active_idx] = selected_index

            log_prob_list.append(log_prob)
            index_path_list.append(index_path)

            seed_input = torch.cat([seed_input, selected_index], dim=-1)

            is_eos = selected_index.squeeze(-1) == eos
            active_idx = ~is_eos

            full_active_idx[full_active_idx.nonzero()[~active_idx]] = 0

            seed_input = seed_input[active_idx]

            if active_idx.sum().item() == 0:
                break

        log_probs = torch.cat(log_prob_list, dim=-1)
        # log_probs.shape == [batch_size * beam_size, generate_len]

        token_ids = torch.cat(index_path_list, dim=-1)
        # token_ids.shape == [batch_size * beam_size, generate_len]

        return log_probs, token_ids, attns

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics['_n_batches'] = self.n_batches
        metrics['_n_samples'] = self.n_samples

        for key, value in self.sample_history.items():
            metrics[key] = value / self.n_samples

        if reset:
            self.n_batches = 0
            self.n_samples = 0
            self.sample_history: Dict[str, float] = defaultdict(float)

        return metrics
