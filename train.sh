#!/usr/bin/env bash

allennlp train config/small.json \
    --serialization-dir logs/main \
    --include-package library

#allennlp train config/large.json \
#    --serialization-dir logs/main \
#    --include-package library
