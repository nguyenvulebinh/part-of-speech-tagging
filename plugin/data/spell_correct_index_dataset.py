import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset, LanguagePairDataset


def collate(
        samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True,
):
    if len(samples) == 0:
        return {}

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

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # src_char_tokens = torch.cat([s['source_char'].unsqueeze(0) for s in samples], 0)
    src_char_tokens = collate_char_tokens(values=[s['source_char'] for s in samples],
                                          pad_idx=pad_idx,
                                          word_max_length=15,
                                          sentence_length=src_tokens.size(-1),
                                          eos_idx=eos_idx,
                                          left_pad=True,
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


class SpellCorrectDataset(LanguagePairDataset):

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
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

        if src_item.size(0) != tgt_item.size(0) and src_item.size(0) % tgt_item.size(0) == 0:
            src_words = src_item[:tgt_item.size(0)]
            src_chars = src_item[tgt_item.size(0):].view(tgt_item.size(0), -1)
        else:
            src_words = src_item
            src_chars = None

        return {
            'id': index,
            'source': src_words,
            'source_char': src_chars,
            'target': tgt_item,
        }

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )
