import os
import unicodedata
from jack.core.data_structures import QASetting
from jack.readers.classification.shared import ClassificationSingleSupportInputModule
from jack.readers.implementations import create_shared_resources
from jack.core.shared_resources import SharedResources
from jack.util.vocab import Vocab
from tqdm import tqdm
import json

import logging
from typing import IO, List, Iterable, Tuple
import warnings

import argparse

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import numpy
import torch

from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.elmo import ElmoEmbedder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def load_parsed_config():
    with open("config_save.db", "r") as f:
        parsed_config = json.load(f)
    return parsed_config


def load_converted_file(filename):
    instances = list()
    with open(filename, "r") as f:
        for line in f:
            instances.append(json.loads(line))
    return instances


def process_qa_setting(qa_settings,
                       original_ids,
                       id2id_dict=dict(),
                       question_dict=dict(),
                       support_dict=dict()):
    def reverse_dict(d, val):
        return [k for k, v in d.items() if v == val]

    annotations = input_module.preprocess(qa_settings)
    questions = list()
    supports = list()
    for original_id, a in tqdm(zip(original_ids, annotations)):
        # id = a.id
        q_id = "Q" + original_id
        s_id = "S" + original_id
        q_sentence = a.question_tokens
        s_sentence = a.support_tokens
        if q_sentence not in question_dict.values():
            question_dict[q_id] = q_sentence
        else:
            id2id_dict[q_id] = reverse_dict(question_dict, q_sentence)[0]

        if s_sentence not in support_dict.values():
            support_dict[s_id] = s_sentence
        else:
            id2id_dict[s_id] = reverse_dict(support_dict, s_sentence)[0]

    return (question_dict, support_dict, id2id_dict)


def my_embed_file(elmo,
                  instances: dict,
                  output_file_path: str,
                  output_format: str = "all",
                  batch_size: int = 64) -> None:
    """
    Computes ELMo embeddings from an input_file where each line contains a sentence tokenized by whitespace.
    The ELMo embeddings are written out in HDF5 format, where each sentences is saved in a dataset.
    Parameters
    ----------
    input_file : ``IO``, required
        A file with one tokenized sentence per line.
    output_file_path : ``str``, required
        A path to the output hdf5 file.
    output_format : ``str``, optional, (default = "all")
        The embeddings to output.  Must be one of "all", "top", or "average".
    batch_size : ``int``, optional, (default = 64)
        The number of sentences to process in ELMo at one time.
    """

    assert output_format in ["all", "top", "average"]

    # Tokenizes the sentences.
    # sentences = [line.strip() for line in input_file if line.strip()]
    # split_sentences = [sentence.split() for sentence in sentences]

    ids = list(instances.keys())
    split_sentences = [instances[id] for id in ids]
    # [{"id": id, "split_sentence": split_sentence}, ....]
    # Uses the sentence as the key.
    # embedded_sentences = zip(sentences,
    #                          self.embed_sentences(split_sentences, batch_size))
    assert len(ids) == len(split_sentences)
    embedded_sentences = zip(ids,
                             elmo.embed_sentences(split_sentences, batch_size))

    logger.info("Processing sentences.")
    with h5py.File(output_file_path, 'w') as fout:
        for key, embeddings in Tqdm.tqdm(embedded_sentences):
            if key in fout.keys():
                logger.warning(
                    f"Key already exists in {output_file_path}, skipping: {key}"
                )
            else:
                if output_format == "all":
                    output = embeddings
                elif output_format == "top":
                    output = embeddings[2]
                elif output_format == "average":
                    output = numpy.average(embeddings, axis=0)

                fout.create_dataset(
                    key, output.shape, dtype='float32', data=output)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("input_file", help="converted jsonl dataset")
    parser.add_argument("question_file", help="json file to resolve sentences")
    parser.add_argument("support_file", help="json file to resolve sentences")
    parser.add_argument(
        "id2id_file",
        help="json file to map id to id ( for overlapping sentences ) ")
    parser.add_argument("output_file", help="hdf5 file that stores embeddings")
    parser.add_argument("--format", choices=["all", "average", "top"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cuda_device", type=int, default=-1)
    args = parser.parse_args()

    if not os.path.exists(args.question_file) or not os.path.exists(
            args.support_file) or not os.path.exists(args.id2id_file):
        vocab_from_embeddings = False
        parsed_config = load_parsed_config()

        embeddings = None
        vocab = Vocab(vocab=embeddings.vocabulary if vocab_from_embeddings
                      and embeddings is not None else None)
        shared_resources = create_shared_resources(
            SharedResources(vocab, parsed_config, embeddings))
        input_module = ClassificationSingleSupportInputModule(shared_resources)

        instances = load_converted_file(args.input_file)
        # "data/FEVER/dataset/converted/dev.sentences.p5.s5.ver20180629.jsonl")
        qa_settings = list()
        logger.info("generating QASettings...")
        original_ids = list()
        for instance in instances:
            id = instance["captionID"]
            premise = instance["sentence1"]
            hypothesis = instance["sentence2"]
            qa_settings.append(QASetting(id=id, question=hypothesis, support=[premise]))
            original_ids.append(id)

        logger.info("process QASettings...")
        q_dict, s_dict, id2id_dict = process_qa_setting(
            qa_settings, original_ids)
        save_json(q_dict, args.question_file)
        save_json(s_dict, args.support_file)
        save_json(id2id_dict, args.id2id_file)

    else:
        q_dict = load_json(args.question_file)
        s_dict = load_json(args.support_file)
        id2id_dict = load_json(args.id2id_file)

    merged_dict = {**q_dict, **s_dict}
    elmo = ElmoEmbedder(cuda_device=args.cuda_device)
    my_embed_file(elmo, merged_dict, args.output_file, args.format, args.batch_size)
