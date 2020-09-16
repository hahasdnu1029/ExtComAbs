#!/usr/bin/env bash

dataset="$1"
datadir="$2"

if [ ! -n "$dataset" ]; then
    echo "lose the dataset name!"
    exit
fi

if [ ! -n "$datadir" ]; then
    echo "lose the data directory!"
    exit
fi

type=(train val test)

echo -e "\033[34m[Shell] Create Vocabulary! \033[0m"

python ./createVoc.py --dataset $dataset --data_path $datadir/train.label.jsonl

echo -e "\033[34m[Shell] The preprocess of dataset $dataset has finished! \033[0m"


