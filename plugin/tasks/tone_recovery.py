import itertools
import os

from fairseq import options, utils
from fairseq.data import (
    data_utils,
    Dictionary
)

from utils import tokenize_line_char, tokenize_line_word

from fairseq.tasks import FairseqTask, register_task
from plugin.data.spell_correct_dataset import SpellCorrectRawDataset


@register_task('tone_recovery')
class ToneRecoveryTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, char_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.char_dict = char_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        char_dict = cls.load_dictionary(os.path.join(paths[0], 'dict_char.txt'))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict, char_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            :param split:
            :param combine:
            :param epoch:
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = SpellCorrectRawDataset(src_file_path=os.path.join(data_path, "{}.{}".format(split, src)),
                                                      tgt_file_path=os.path.join(data_path, "{}.{}".format(split, tgt)),
                                                      dict=self.tgt_dict,
                                                      char_dict=self.char_dict,
                                                      left_pad_source=self.args.left_pad_source,
                                                      left_pad_target=self.args.left_pad_target,
                                                      max_source_positions=self.args.max_source_positions,
                                                      max_target_positions=self.args.max_target_positions,
                                                      word_max_length=self.args.word_max_length)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    @classmethod
    def build_dict(cls, filenames, word_level=False, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename,
                                              d,
                                              tokenize_line_word if word_level else tokenize_line_char,
                                              workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d
