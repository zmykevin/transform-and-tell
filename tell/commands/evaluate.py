import hashlib
import json
import logging
import math
import os
import pickle
import string
from typing import Any, Dict, Iterable
from tqdm import tqdm

import spacy
import textstat
import torch
from allennlp.common.checks import check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util
from allennlp.training.util import HasBeenWarned, datasets_from_params
from allennlp.data.fields import ArrayField, MetadataField, TextField
from nltk.tokenize import word_tokenize
from tell.data.token_indexers import RobertaTokenIndexer
from spacy.tokens import Doc

from .train import yaml_to_params



logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


###Define by Mingyang Zhou#######
# def evaluate_from_purterbated_file(archive_path, model_path, overrides=None, eval_suffix='', device=0):
#     if archive_path.endswith('gz'):
#         archive = load_archive(archive_path, device, overrides)
#         config = archive.config
#         prepare_environment(config)
#         model = archive.model
#         serialization_dir = os.path.dirname(archive_path)
#     elif archive_path.endswith('yaml'):
#         config = yaml_to_params(archive_path, overrides)
#         prepare_environment(config)
#         config_dir = os.path.dirname(archive_path)
#         serialization_dir = os.path.join(config_dir, 'serialization')

#     all_datasets = datasets_from_params(config)

#     vocab = Vocabulary.from_params(config.pop('vocabulary'))
#     model = Model.from_params(vocab=vocab, params=config.pop('model'))

#     if model_path:
#         best_model_state = torch.load(model_path)
#         model.load_state_dict(best_model_state)

#     instances = all_datasets.get('test')
#     iterator = DataIterator.from_params(
#         config.pop("validation_iterator"))

#     iterator.index_with(model.vocab)
#     model.eval().to(device)
#     model.evaluate_mode = True
    
#     with torch.no_grad():
#         eval_iterator = iterator(instances,
#                                      num_epochs=1,
#                                      shuffle=False)
#         logger.info("Iterating over dataset")
#         generator_tqdm = Tqdm.tqdm(
#             eval_iterator, total=iterator.get_num_batches(instances))
#         for batch in generator_tqdm:
#             # print("number of meta_data: {}".format(len(batch['metadata'])))
#             # print("number of context tensors: {}".format(batch['context']['roberta'].size()))

#             #Swap the image, obj_embeds, face_embeds
            
#             permute_index = []
#             for x in range(len(batch['meta_data'])):
#                 if x == 0:
#                     y = len(batch['meta_data'])-1
#                 else:
#                     y = x-1
#                 permute_index.append(y)
#             #generate the perm
#             perm = torch.LongTensor(permute_index)
#             batch["obj_embeds"] = batch["obj_embeds"][perm]
#             batch["face_embeds"] = batch["face_embeds"][perm]
#             batch["image"] = batch["image"][perm]


#             break


#     # metrics = evaluate(model, instances, iterator,
#     #                    device, serialization_dir, eval_suffix, batch_weight_key='')

#     # logger.info("Finished evaluating.")
#     # logger.info("Metrics:")
#     # for key, metric in metrics.items():
#     #     logger.info("%s: %s", key, metric)

#     # output_file = os.path.join(
#     #     serialization_dir, f"evaluate-metrics{eval_suffix}.json")
#     # if output_file:
#     #     with open(output_file, "w") as file:
#     #         json.dump(metrics, file, indent=4)
#     return None

def evaluate_purterbation(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             serialization_dir: str,
             eval_suffix: str,
             batch_weight_key: str) -> Dict[str, Any]:
    check_for_gpu(cuda_device)
    nlp = spacy.load("en_core_web_lg")
    #print(serialization_dir)
    #Add the eval_suffix default_value
    eval_suffix = "_perterbation_train"
    assert not os.path.exists(os.path.join(
        serialization_dir, f'generations{eval_suffix}.jsonl'))

    # caching saves us extra 30 minutes
    if 'goodnews' in serialization_dir:
        cache_path = 'data/goodnews/evaluation_cache.pkl'
    elif 'nytimes' in serialization_dir:
        cache_path = 'data/nytimes/evaluation_cache.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(
            iterator, total=data_iterator.get_num_batches(instances))

        # Number of batches in instances.
        batch_count = 0
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative weighted loss
        total_loss = 0.0
        # Cumulative weight across all batches.
        total_weight = 0.0

        for batch in generator_tqdm:
            batch_count += 1

            #Added by Mingyang Zhou
            #Perturbate the index
            permute_index = []
            for x in range(len(batch['metadata'])):
                if x == 0:
                    y = len(batch['metadata'])-1
                else:
                    y = x-1
                batch['metadata'][x]['purterbation_image_path'] = batch['metadata'][y]['image_path']
                batch['metadata'][x]['purterbation_image_caption'] = batch['metadata'][y]['caption']
                permute_index.append(y)
            #generate the perm
            perm = torch.LongTensor(permute_index)
            batch["obj_embeds"] = batch["obj_embeds"][perm]
            batch["face_embeds"] = batch["face_embeds"][perm]
            batch["image"] = batch["image"][perm]
            #added the purterbation image_caption

            #added the puterbation image_path

            batch = nn_util.move_to_device(batch, cuda_device)
            output_dict = model(**batch)
            #print(output_dict['metadata'][0].keys())
            loss = output_dict.get("loss")

            write_to_json_purterbation(output_dict, serialization_dir,
                          nlp, eval_suffix, cache)

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                if batch_weight_key:
                    weight = output_dict[batch_weight_key].item()
                else:
                    weight = 1.0

                total_weight += weight
                total_loss += loss.item() * weight
                # Report the average loss so far.
                metrics["loss"] = total_loss / total_weight

            if (not HasBeenWarned.tqdm_ignores_underscores and
                    any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                HasBeenWarned.tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            # if loss_count != batch_count:
            #     raise RuntimeError("The model you are trying to evaluate only sometimes " +
            #                        "produced a loss!")
            final_metrics["loss"] = total_loss / total_weight

    if not os.path.exists(cache_path):
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    return final_metrics

def write_to_json_purterbation(output_dict, serialization_dir, nlp, eval_suffix, cache):
    if 'captions' not in output_dict:
        return

    captions = output_dict['captions']
    generations = output_dict['generations']
    metadatas = output_dict['metadata']
    if 'copied_texts' in output_dict:
        copied_texts = output_dict['copied_texts']
    else:
        copied_texts = ['' for _ in range(len(captions))]

    out_path = os.path.join(
        serialization_dir, f'generations{eval_suffix}.jsonl')
    with open(out_path, 'a') as f:
        for i, caption in enumerate(captions):
            m = metadatas[i]
            generation = generations[i]
            caption_doc = spacize(m['caption'], cache, nlp)
            gen_doc = nlp(generation)
            context_doc = spacize(m['context'], cache, nlp)
            obj = {
                'caption': caption,
                'raw_caption': m['caption'],
                'generation': generation,
                'copied_texts': copied_texts[i],
                'web_url': m['web_url'],
                'image_path': m['image_path'],
                'purterbation_image_path': m['purterbation_image_path'],
                'purterbation_image_caption': m['purterbation_image_caption'],
                'context': m['context'],
                'caption_names': get_proper_nouns(caption_doc),
                'generated_names': get_proper_nouns(gen_doc),
                'context_names': get_proper_nouns(context_doc),
                'caption_entities': get_entities(caption_doc),
                'generated_entities': get_entities(gen_doc),
                'context_entities': get_entities(context_doc),
                'caption_readability': get_readability_scores(m['caption']),
                'gen_readability': get_readability_scores(generation),
                'caption_np': get_narrative_productivity(m['caption']),
                'gen_np': get_narrative_productivity(generation),
            }

            if 'copied_texts' in output_dict:
                obj['copied_text'] = output_dict['copied_texts'][i]

            f.write(f'{json.dumps(obj)}\n')
##########################################################################################
def evaluate_from_file(archive_path, model_path, overrides=None, eval_suffix='', device=0, purterbation=False):
    #Added by Mingyang Zhou
    eval_suffix = "_test_nerlocation"
    #######################
    if archive_path.endswith('gz'):
        archive = load_archive(archive_path, device, overrides)
        config = archive.config
        prepare_environment(config)
        model = archive.model
        serialization_dir = os.path.dirname(archive_path)
    elif archive_path.endswith('yaml'):
        config = yaml_to_params(archive_path, overrides)
        prepare_environment(config)
        config_dir = os.path.dirname(archive_path)
        serialization_dir = os.path.join(config_dir, 'serialization')
    #Added by Mingyang to initialize inndexer#
    # print(config.get("dataset_reader"))

    #########################################
    all_datasets = datasets_from_params(config)

    # We want to create the vocab from scratch since it might be of a
    # different type. Vocabulary.from_files will always create the base
    # Vocabulary instance.
    # if os.path.exists(os.path.join(serialization_dir, "vocabulary")):
    #     vocab_path = os.path.join(serialization_dir, "vocabulary")
    #     vocab = Vocabulary.from_files(vocab_path)

    vocab = Vocabulary.from_params(config.pop('vocabulary'))
    model = Model.from_params(vocab=vocab, params=config.pop('model'))

    if model_path:
        best_model_state = torch.load(model_path)
        model.load_state_dict(best_model_state)

    #instances_raw = all_datasets.get('test')
    instances = all_datasets.get('test')
    #instances = all_datasets.get('train')

    #Get the indexer
    #indexer_params = config.get('dataset_reader').get('token_indexers')
    #print(config.pop("dataset_reader"))
    # token_indexers = {}
    # for k, p in indexer_params.items():
    #     class_type = p.pop("type")
    #     #print(class_type)
    #     token_indexers[k] = RobertaTokenIndexer.from_params(p)
    # print("load the test dataset")
    # for x in instances_raw:
    #     print(x.fields['context'])
        #print(x.fields['metadata']['context'])
        # break
    ########Added by Mingyang Zhou, NewsClipp Image Swap#######
    # print("load the swaped subset list")
    # with open("/home/zmykevin/semafor/code/news_clippings_generation/data/goodnews_test_same_semantics_clip_text_text.json", "r") as f:
    #     newsclipping_swap_raw= json.load(f)
    # newsclipping_swap = {x['id']:x['image_id'] for x in newsclipping_swap_raw if x.get('foil', False)}
    # instance_mapping =  {}
    # for x in tqdm(instances_raw):
    #     image_id = x.fields['metadata']['image_path'].split('/')[-1].split('.')[0]
    #     instance_mapping[image_id] = x
    
    # instances = []
    # for x in instances_raw:
    #     article_id = x.fields['metadata']['image_path'].split('/')[-1].split('.')[0]
    #     if newsclipping_swap.get(article_id, None) is not None:
    #         tgt_id = newsclipping_swap.get(article_id, None)
    #         tgt_instance =  instance_mapping.get(tgt_id, None)
    #         if tgt_instance is not None:
    #             #replace the image, face_embeds , object_emebds
    #             x.fields['image'] = tgt_instance.fields['image']
    #             x.fields['face_embeds'] = tgt_instance.fields['face_embeds']
    #             if tgt_instance.fields.get('obj_embeds', None) is not None:
    #                 x.fields['obj_embeds'] = tgt_instance.fields['obj_embeds']
    #             instances.append(x)


    ########Added by Mingyang Zhou, Image Swap##################
    # print("load the swaped subset list")
    # with open("/home/zmykevin/semafor/code/transform-and-tell/data/goodnews/challenging_subset/name_count_matching_swap_test.json", "r") as f:
    #     ner_swap_person = json.load(f)
    # instance_mapping =  {}
    # for x in tqdm(instances_raw):
    #     image_id = x.fields['metadata']['image_path'].split('/')[-1].split('.')[0]
    #     instance_mapping[image_id] = x

    # instances = []
    # for x in instances_raw:
    #     article_id = x.fields['metadata']['image_path'].split('/')[-1].split('.')[0]
    #     #print(article_id)
    #     if ner_swap_person.get(article_id, None) is not None:
    #         tgt_id = ner_swap_person.get(article_id, None)['tgt_id']
    #         tgt_instance =  instance_mapping.get(tgt_id, None)
    #         if tgt_instance is not None:
    #             #replace the image, face_embeds , object_emebds
    #             x.fields['image'] = tgt_instance.fields['image']
    #             x.fields['face_embeds'] = tgt_instance.fields['face_embeds']
    #             if tgt_instance.fields.get('obj_embeds', None) is not None:
    #                 x.fields['obj_embeds'] = tgt_instance.fields['obj_embeds']
    #             instances.append(x)
    #         else:
    #             instances.append(x)
    #         #print(tgt_id)
    #     else:
    #         instances.append(x)

    ################################################
    # #########Added by Mingyang Zhou##################
    # print("load the challenging subset list")
    # with open("/home/zmykevin/semafor/code/transform-and-tell/data/goodnews/challenging_subset/challenge_test.json", "r") as f:
    #     challenge_list = json.load(f)
    # valid_count = 0
    # instances = []
    # for x in tqdm(instances_raw):
    #     img_path = x.fields['metadata']['image_path']
    #     if challenge_list.get(img_path, False):
    #         instances.append(x)
    #         valid_count += 1
        
    # #################################################
    
    iterator = DataIterator.from_params(
        config.pop("validation_iterator"))
    
    iterator.index_with(model.vocab)
    model.eval().to(device)
    model.evaluate_mode = True
    
    if not purterbation:
        metrics = evaluate(model, instances, iterator,
                       device, serialization_dir, eval_suffix, batch_weight_key='')
    else:
        metrics = evaluate_purterbation(model, instances, iterator,
                       device, serialization_dir, eval_suffix, batch_weight_key='')
    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    output_file = os.path.join(
        serialization_dir, f"evaluate-metrics{eval_suffix}.json")
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
    return metrics


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             serialization_dir: str,
             eval_suffix: str,
             batch_weight_key: str) -> Dict[str, Any]:
    check_for_gpu(cuda_device)
    nlp = spacy.load("en_core_web_lg")
    print(serialization_dir)
    assert not os.path.exists(os.path.join(
        serialization_dir, f'generations{eval_suffix}.jsonl'))

    # caching saves us extra 30 minutes
    if 'goodnews' in serialization_dir:
        cache_path = 'data/goodnews/evaluation_cache.pkl'
    elif 'nytimes' in serialization_dir:
        cache_path = 'data/nytimes/evaluation_cache.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(
            iterator, total=data_iterator.get_num_batches(instances))

        # Number of batches in instances.
        batch_count = 0
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative weighted loss
        total_loss = 0.0
        # Cumulative weight across all batches.
        total_weight = 0.0

        for batch in generator_tqdm:
            batch_count += 1
            batch = nn_util.move_to_device(batch, cuda_device)
            output_dict = model(**batch)
            loss = output_dict.get("loss")

            write_to_json(output_dict, serialization_dir,
                          nlp, eval_suffix, cache)

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                if batch_weight_key:
                    weight = output_dict[batch_weight_key].item()
                else:
                    weight = 1.0

                total_weight += weight
                total_loss += loss.item() * weight
                # Report the average loss so far.
                metrics["loss"] = total_loss / total_weight

            if (not HasBeenWarned.tqdm_ignores_underscores and
                    any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                HasBeenWarned.tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            # if loss_count != batch_count:
            #     raise RuntimeError("The model you are trying to evaluate only sometimes " +
            #                        "produced a loss!")
            final_metrics["loss"] = total_loss / total_weight

    if not os.path.exists(cache_path):
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    return final_metrics


def write_to_json(output_dict, serialization_dir, nlp, eval_suffix, cache):
    if 'captions' not in output_dict:
        return

    captions = output_dict['captions']
    generations = output_dict['generations']
    metadatas = output_dict['metadata']
    if 'copied_texts' in output_dict:
        copied_texts = output_dict['copied_texts']
    else:
        copied_texts = ['' for _ in range(len(captions))]

    out_path = os.path.join(
        serialization_dir, f'generations{eval_suffix}.jsonl')
    with open(out_path, 'a') as f:
        for i, caption in enumerate(captions):
            m = metadatas[i]
            generation = generations[i]
            caption_doc = spacize(m['caption'], cache, nlp)
            gen_doc = nlp(generation)
            context_doc = spacize(m['context'], cache, nlp)
            obj = {
                'caption': caption,
                'raw_caption': m['caption'],
                'generation': generation,
                'copied_texts': copied_texts[i],
                'web_url': m['web_url'],
                'image_path': m['image_path'],
                'context': m['context'],
                'caption_names': get_proper_nouns(caption_doc),
                'generated_names': get_proper_nouns(gen_doc),
                'context_names': get_proper_nouns(context_doc),
                'caption_entities': get_entities(caption_doc),
                'generated_entities': get_entities(gen_doc),
                'context_entities': get_entities(context_doc),
                'caption_readability': get_readability_scores(m['caption']),
                'gen_readability': get_readability_scores(generation),
                'caption_np': get_narrative_productivity(m['caption']),
                'gen_np': get_narrative_productivity(generation),
            }

            if 'copied_texts' in output_dict:
                obj['copied_text'] = output_dict['copied_texts'][i]

            f.write(f'{json.dumps(obj)}\n')


def spacize(text, cache, nlp):
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if key not in cache:
        cache[key] = nlp(text).to_bytes()

    return Doc(nlp.vocab).from_bytes(cache[key])


def get_proper_nouns(doc):
    proper_nouns = []
    for token in doc:
        if token.pos_ == 'PROPN':
            proper_nouns.append(token.text)
    return proper_nouns


def get_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'tokens': [{'text': tok.text, 'pos': tok.pos_} for tok in ent],
        })
    return entities


def get_readability_scores(text):
    scores = {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'smog_index': textstat.smog_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'linsear_write_formula': textstat.linsear_write_formula(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'text_standard': textstat.text_standard(text, float_output=True),
        'difficult_words': textstat.difficult_words(text) / len(text.split()),
    }
    return scores


def is_word(tok):
    return tok not in string.punctuation


def get_narrative_productivity(text):
    doc = word_tokenize(text)
    doc = list(filter(is_word, doc))
    n_words = len(doc)
    n_terms = len(set(doc))

    scores = {
        'basic_ttr': basic_ttr(n_terms, n_words),
        'root_ttr': root_ttr(n_terms, n_words),
        'corrected_ttr': corrected_ttr(n_terms, n_words),
        'herdan': herdan(n_terms, n_words),
        'summer': summer(n_terms, n_words),
        'maas': maas(n_terms, n_words),
    }

    return scores


def basic_ttr(n_terms, n_words):
    """ Type-token ratio (TTR) computed as t/w, where t is the number of unique
    terms/vocab, and w is the total number of words.
    (Chotlos 1944, Templin 1957)
    """
    if n_words == 0:
        return 0
    return n_terms / n_words


def root_ttr(n_terms, n_words):
    """ Root TTR (RTTR) computed as t/sqrt(w), where t is the number of unique terms/vocab,
        and w is the total number of words.
        Also known as Guiraud's R and Guiraud's index.
        (Guiraud 1954, 1960)
    """
    if n_words == 0:
        return 0
    return n_terms / math.sqrt(n_words)


def corrected_ttr(n_terms, n_words):
    """ Corrected TTR (CTTR) computed as t/sqrt(2 * w), where t is the number of unique terms/vocab,
        and w is the total number of words.
        (Carrol 1964)
    """
    if n_words == 0:
        return 0
    return n_terms / math.sqrt(2 * n_words)


def herdan(n_terms, n_words):
    """ Computed as log(t)/log(w), where t is the number of unique terms/vocab, and w is the
        total number of words.
        Also known as Herdan's C.
        (Herdan 1960, 1964)
    """
    if n_words <= 1:
        return 0
    return math.log(n_terms) / math.log(n_words)


def summer(n_terms, n_words):
    """ Computed as log(log(t)) / log(log(w)), where t is the number of unique terms/vocab, and
        w is the total number of words.
        (Summer 1966)
    """
    try:
        math.log(math.log(n_terms)) / math.log(math.log(n_words))
    except ValueError:
        return 0


def maas(n_terms, n_words):
    """ Maas's TTR, computed as (log(w) - log(t)) / (log(w) * log(w)), where t is the number of
        unique terms/vocab, and w is the total number of words. Unlike the other measures, lower
        maas measure indicates higher lexical richness.
        (Maas 1972)
    """
    # We cap this score at 0.2
    if n_words <= 1:
        return 0.2
    score = (math.log(n_words) - math.log(n_terms)) / \
        (math.log(n_words) ** 2)
    return min(score, 0.2)
