#!/usr/bin/env bash

allennlp predict logs/main/model.tar.gz data/Bengali_Dataset.txt \
    --output-file data/predict.txt \
    --predictor simple \
    --include-package library \
    --use-dataset-reader
