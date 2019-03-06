#!/usr/bin/env bash

allennlp evaluate logs/main/model.tar.gz data/Bengali_Dataset.txt \
    --include-package library
