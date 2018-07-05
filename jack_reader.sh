#!/bin/bash
# exit immediately if shell returns non-zero
set -e

START_TIME=$SECONDS

# check the required params
if [ -z $label_pred ]; then
    echo "You need to specify label_pred"
    exit 1
fi
if [ -z $reader ]; then
    echo "You need to specify reader"
    exit 1
fi

_CONDA_EXE=/home/tyoneda/anaconda3/bin/conda

if ! [ -z $prependtitle ]; then
    prependtitle="--prependtitle"
fi
if [ ! -z $prependlinum ]; then
    prependlinum="--prependlinum"
fi
if [ -z $bias1 ]; then
    bias1=0
fi
if [ -z $bias2 ]; then
    bias2=0
fi


# static variables
predicted_evidence=../fever/data/indexed_data/dev.sentences.p5.s5.ver20180629.jsonl
# predicted_evidence=../fever/data/indexed_data/dev.sentences.p5.s5.jsonl

echo "label prediction path (full path): ${label_pred}"

# add current dir to stack
pushd .

# load aliases
shopt -s expand_aliases
. ~/.bash_aliases

# load conda settings
. /home/tyoneda/anaconda3/etc/profile.d/conda.sh

cd ~/jack
echo "moved to `pwd`"
conda deactivate && conda activate jack
# in ~/jack dir

echo CUDA_VISIBLE_DEVICES=0 PYTHONPATH=".:../fever" anaconda-python3-gpu ../fever/jack_reader.py ${predicted_evidence} ${label_pred} --saved_reader ${reader} --bias1 -- ${bias1} --bias2 -- ${bias2} --batch_size 256 ${prependtitle} ${prependlinum}
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=".:../fever" anaconda-python3-gpu ../fever/jack_reader.py ${predicted_evidence} ${label_pred} --saved_reader ${reader} --bias1 -- ${bias1} --bias2 -- ${bias2} --batch_size 256 -${prependtitle} ${prependlinum}


cd ~/fever-baselines
echo "moved to `pwd`"
conda deactivate && conda activate fever-baselines
# in ~/fever-baselines dir
echo PYTHONPATH=src python src/scripts/score.py --predicted_labels ${label_pred} --predicted_evidence ${predicted_evidence} --actual ../fever/data/dev.jsonl
PYTHONPATH=src python src/scripts/score.py --predicted_labels ${label_pred} --predicted_evidence ${predicted_evidence} --actual ../fever/data/dev.jsonl

# go back to the original dir
popd
echo "moved back to `pwd`"

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo elapsed time: $ELAPSED_TIME
