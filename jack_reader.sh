#!/bin/bash
START_TIME=$SECONDS

_CONDA_EXE=/home/tyoneda/anaconda3/bin/conda

cuda=$1
label_pred=$2
reader=$3
prependlinum=$4

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

# if $4 is not set...
if [ -z $4 ]; then
    echo CUDA_VISIBLE_DEVICES=${cuda} PYTHONPATH=".:../fever" anaconda-python3-gpu ../fever/jack_reader.py ${predicted_evidence} ${label_pred} --saved_reader ${reader} --batch_size 256 # --prependlinum
    CUDA_VISIBLE_DEVICES=${cuda} PYTHONPATH=".:../fever" anaconda-python3-gpu ../fever/jack_reader.py ${predicted_evidence} ${label_pred} --saved_reader ${reader} --batch_size 256 #  --prependlinum
else
    cutoff=$4
    echo CUDA_VISIBLE_DEVICES=${cuda} PYTHONPATH=".:../fever" anaconda-python3-gpu ../fever/jack_reader.py ${predicted_evidence} ${label_pred} --saved_reader ${reader} --batch_size 256 --prependtitle # --prependlinum
    CUDA_VISIBLE_DEVICES=${cuda} PYTHONPATH=".:../fever" anaconda-python3-gpu ../fever/jack_reader.py ${predicted_evidence} ${label_pred} --saved_reader ${reader} --batch_size 256 --prependtitle # --prependlinum
fi


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