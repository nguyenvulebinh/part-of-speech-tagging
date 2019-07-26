import torch

from fairseq.data import data_utils
import numpy as np
from fairseq.data.language_pair_dataset import FairseqDataset
from tqdm import tqdm


def collate(
        samples, pad_idx, eos_idx, word_max_length, left_pad_source=True, left_pad_target=False,
        input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def collate_char_tokens(values, pad_idx, sentence_length,
                            word_max_length, eos_idx=None, left_pad=False,
                            move_eos_to_beginning=False):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = word_max_length
        res = values[0][0].new(len(values), sentence_length, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel(), "{} != {}".format(dst.numel(), src.numel())
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, line in enumerate(values):
            for j, v in enumerate(line):
                if len(v) > word_max_length:
                    v = v[-word_max_length:]
                copy_tensor(v, res[i][sentence_length - len(line) + j][size - len(v):] if left_pad else
                res[i][sentence_length - len(line) + j][:len(v)])
        return res

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    src_char_tokens = collate_char_tokens(values=[s['source_char'] for s in samples],
                                          pad_idx=pad_idx,
                                          word_max_length=word_max_length,
                                          sentence_length=src_tokens.size(-1),
                                          eos_idx=eos_idx,
                                          left_pad=left_pad_source,
                                          move_eos_to_beginning=False)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    src_char_tokens = src_char_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_char_tokens': src_char_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class SpellCorrectRawDataset(FairseqDataset):
    def __init__(self, src_file_path, tgt_file_path, dict, char_dict, word_max_length, left_pad_source=True,
                 left_pad_target=False,
                 max_source_positions=1024, max_target_positions=1024, input_feeding=True,
                 remove_eos_from_source=False, append_eos_to_target=False, shuffle=True):
        # Read file raw
        with open(src_file_path, 'r', encoding='utf-8') as src_file:
            with open(tgt_file_path, 'r', encoding='utf-8') as tgt_file:
                src_lines = src_file.read().split('\n')
                tgt_lines = tgt_file.read().split('\n')

        print('Done read raw file. Start convert to indices\n')
        # check number sample input == number sample output
        assert len(src_lines) == len(tgt_lines)

        src_indices = []
        tgt_indices = []
        src_char_indices = []

        for line in tqdm(src_lines, desc='Convert source'):
            src_indices.append(dict.encode_line(line, add_if_not_exist=False).long())
            words = line.split() + ['']
            src_char_indices.append([char_dict.encode_line(' '.join(list(word)),
                                                           add_if_not_exist=False).long() for word in words])

        for line in tqdm(tgt_lines, desc='Convert target'):
            tgt_indices.append(dict.encode_line(line, add_if_not_exist=False).long())

        self.src = src_indices
        self.tgt = tgt_indices
        self.src_char = src_char_indices
        self.word_max_length = word_max_length
        self.src_sizes = np.array([line.size(0) for line in src_indices])
        self.tgt_sizes = np.array([line.size(0) for line in tgt_indices])
        # check input size == target size
        assert ((self.src_sizes == self.tgt_sizes).sum()) == len(src_lines)
        self.src_dict = dict
        self.tgt_dict = dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        src_char_item = self.src_char[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
                src_char_item = self.src_char[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'source_char': src_char_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, word_max_length=self.word_max_length
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
