"""
Module for the sequence encoding, tokenizing, and word identification.

It uses Hugging Face library for tokenization.
"""
import json
import pathlib
import re
from tokenizers import Tokenizer


def get_package_path():
    """
    Get the parent path.

    Returns
    -------
    str
        Parent path.

    """
    return str(pathlib.Path(__file__).parent.resolve())


package_path = get_package_path()


class HFWordIdentifier:
    """Tokenizingstrings in sub-word token strings.

    Attributes_
        tokenizer : Hugging Face tokenizer using given path.

    """
    
    def __init__(self, path):

        self.tokenizer = Tokenizer.from_file(path)

    def identify_words(self, sequences, padding_len=None, out_type='int'):
        """
        Encode the given batch of inputs.

        Parameters
        ----------
        sequences : list
            A list of single sequences to encode.
        padding_len : int, optional
            If specified, the length at which to pad. 
        out_type : str, optional
            Integer or string as the output type.

        Raises
        ------
        ValueError
            If output type is not integer or string.

        Returns
        -------
        list
            The encoded batch.

        """
        encodings = self.tokenizer.encode_batch(sequences)
        if padding_len is not None:
            for encoding in encodings:
                encoding.pad(padding_len,
                             direction='right',
                             pad_id=0,
                             pad_token='[PAD]')
                encoding.truncate(padding_len)

        if out_type == 'int':
            return [encoding.ids for encoding in encodings]
        elif out_type == 'str':
            return [encoding.tokens for encoding in encodings]
        else:
            raise ValueError('Invalid out_type for word identification')


def tokenize_with_hf(hf_tokenizer_name,
                     sequences,
                     padding_len,
                     out_type='int'):
    """
    Caller for the Hugging Face tokenizer methods.

    Parameters
    ----------
    hf_tokenizer_name : str
        Name of the tokenizer path.
    sequences : list
        List of sequences to encode.
    padding_len : int
        The length at which to pad. 
    out_type : str, optional
        Integer or string as the output type.

    Returns
    -------
    list
        Encoded sequences.

    """
    vocabs_path = f'{package_path}/data/vocabs'
    tokenizer_path = f'{vocabs_path}/{hf_tokenizer_name}.json'
    tokenizer = HFWordIdentifier(tokenizer_path)
    return tokenizer.identify_words(sequences,
                                    padding_len=padding_len,
                                    out_type=out_type)


def smiles_segmenter(sequence):
    """
    Split SMILES sequences using predefined segments.

    Parameters
    ----------
    sequence : str
        SMILES sequence.

    Returns
    -------
    tokens : list
        segmented SMILES sequences.

    """
    pattern = '(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(sequence)]
    return tokens


def encode_smiles(smiles):
    """
    Encode SMILES sequences using given vocabulary.

    Parameters
    ----------
    smiles : sequence
        SMILES sequence.

    Returns
    -------
    str
        Encoded SMILES sequence.

    """
    segments = smiles_segmenter(smiles)
    vocabs_path = f'{package_path}/data/vocabs'
    with open(f'{vocabs_path}/chemical/chembl27_encoding.json') as f:
        encoding_vocab = json.load(f)

    return ''.join([encoding_vocab.get(segment, encoding_vocab['[OOV]']) for segment in segments])