import argparse
from jack import readers
from jack.core import QASetting
from jack.core import Ports
# import random
# import numpy as np
# import tensorflow as tf
# tf.set_random_seed(1)
# np.random.seed(1)
# random.seed(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("read claim/evidence and output verdict")
    parser.add_argument(
        "--saved_reader", help="path to saved reader directory")
    args = parser.parse_args()


    print("loading reader from file:", args.saved_reader)
    dam_reader = readers.reader_from_file(args.saved_reader)

    claim = "A girl is happy"
    evidence = "A girl plays"
    # claim = "edison is genious"
    # evidence = "Edison is stupid"

    print("claim:", claim)
    print("evidence:", evidence)

    setting = QASetting(question=claim, support=[evidence])
    for i in range(20):
        preds = dam_reader([setting])
        print("{}\t{:.4}".format(preds[0][0].text, preds[0][0].score))
    # for pred in preds:
    #     print(pred[0].text, pred[0].score)
