from typing import Dict, List, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from library.lemma_edit import gen_lemma_rule

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    return line.strip() == ''


@DatasetReader.register("simple")
class SimpleDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, lemmas = fields
                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens_]

                    lemma_rules = [gen_lemma_rule(token, lemma)
                                   for token, lemma in zip(tokens_, lemmas)]

                    yield self.text_to_instance(tokens, lemma_rules)

    def text_to_instance(self, # type: ignore
                         tokens: List[Token],
                         lemma_rules: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {
            "tokens": sequence,
            "tags": SequenceLabelField(lemma_rules, sequence, self.label_namespace),
            "metadata": MetadataField({
                "words": [x.text for x in tokens]
            })
        }

        return Instance(instance_fields)
