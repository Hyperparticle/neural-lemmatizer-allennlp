# A Dead Simple Neural Lemmatizer using AllenNLP

## Getting Started

Install AllenNLP with the command

```bash
pip install allennlp
```

To train, run `bash.sh`, or

```bash
allennlp train config/small.json \
    --serialization-dir logs/main \
    --include-package library
```

All logs will be put in the `logs/main` directory. If you would like to train again, delete `logs/main`, otherwise
there will be an error.

To predict a trained model, run `predict.sh` or

```bash
allennlp predict logs/main/model.tar.gz data/Bengali_Dataset.txt \
    --output-file data/predict.txt \
    --predictor simple \
    --include-package library \
    --use-dataset-reader
```

To evaluate a trained model, run `evaluate.sh` or

```bash
allennlp evaluate logs/main/model.tar.gz data/Bengali_Dataset.txt \
    --include-package library
```

This will output predictions to `data/predict.txt`.

## Visualize Model Performance in TensorBoard

To view `tensorboard` logs, just run

```bash
tensorboard --logdir logs
```

Make sure `tensorflow` is installed to be able to run the `tensorboard` command.

## Providing Your Own Data or Model

Prepare your data in the format

```
word <tab> lemma
```

with each sentence separated by an empty line. Then split your data into train, validation, and test sets.

Copy a config file in the `config` directory and change `train_data_path`, `validation_data_path`,
and `test_data_path`, to be the paths of your train, validation, and test files respectively.

Then run training with your new config file

```bash
allennlp train path_to_config.json \
    --serialization-dir logs/main \
    --include-package library
```

Finally, output predictions with your new model using

```bash
allennlp predict logs/main/model.tar.gz path_to_input_file.txt \
    --output-file path_to_output_file.txt \
    --predictor simple \
    --include-package library \
    --use-dataset-reader
```
