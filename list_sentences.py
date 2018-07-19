from jack.core.data_structures import QASetting
from jack.readers.classification.shared import ClassificationSingleSupportInputModule
from jack.readers.implementations import create_shared_resources
from jack.core.shared_resources import SharedResources
from jack.util.vocab import Vocab
import json


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
                       out_file,
                       question_set=set(),
                       support_set=set()):
    annotations = input_module.preprocess(qa_settings)
    questions = list()
    supports = list()
    for a in annotations:
        q_sentence = " ".join(a.question_tokens)
        s_sentence = " ".join(a.support_tokens)
        if q_sentence not in question_set:
            question_set.add(q_sentence)
            questions.append(q_sentence)
        if s_sentence not in support_set:
            support_set.add(s_sentence)
            supports.append(s_sentence)

    with open(out_file, "a") as f:
        for sentence in questions + supports:
            f.write(sentence + "\n")


if __name__ == "__main__":
    vocab_from_embeddings = False
    parsed_config = load_parsed_config()

    embeddings = None
    vocab = Vocab(vocab=embeddings.vocabulary if vocab_from_embeddings
                  and embeddings is not None else None)
    shared_resources = create_shared_resources(
        SharedResources(vocab, parsed_config, embeddings))
    input_module = ClassificationSingleSupportInputModule(shared_resources)

    instances = load_converted_file("data/FEVER/dataset/converted/dev.sentences.p5.s5.ver20180629.jsonl")
    qa_settings = list()
    print("generating QASettings...")
    for instance in instances:
        q = instance["sentence1"]
        a = instance["sentence2"]
        qa_settings.append(QASetting(question=q, support=[a]))

    print("process QASettings...")
    process_qa_setting(qa_settings, "sentence_list.txt")
