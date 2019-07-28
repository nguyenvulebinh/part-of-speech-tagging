"""
Data pre-processing: build vocabularies and binarize training data.
"""
import utils as postag_utils
from fairseq import options, tasks, utils
import os
from collections import Counter
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer, safe_readline
from multiprocessing import Pool
from utils import tokenize_line_char, tokenize_line_word
import torch


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    lang_part = (
        ".{}-{}.{}".format(args.source_lang, args.target_lang, lang) if lang is not None else ""
    )
    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def str_to_bin(filename, dict, consumer, char_vocab, append_eos=True, reverse_order=False,
               offset=0, end=-1):
    nseq, ntok = 0, 0
    replaced = Counter()

    def replaced_consumer(word, idx):
        if idx == dict.unk_index and word != dict.unk_word:
            replaced.update([word])

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

    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(offset)
        # next(f) breaks f.tell(), hence readline() must be used
        line = safe_readline(f)
        while line:
            if end > 0 and f.tell() > end:
                break
            ids = dict.encode_line(
                line=line,
                line_tokenizer=tokenize_line_word,
                add_if_not_exist=False,
                consumer=replaced_consumer,
                append_eos=append_eos,
                reverse_order=reverse_order,
            )
            if char_vocab is not None:
                ids_char = [char_vocab.encode_line(
                    line=word,
                    line_tokenizer=tokenize_line_char,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                ) for word in line.split() + ['']]

                ids_char_pad = collate_char_tokens(values=[ids_char],
                                                   pad_idx=char_vocab.pad(),
                                                   word_max_length=15,
                                                   sentence_length=len(ids),
                                                   eos_idx=char_vocab.eos(),
                                                   left_pad=True,
                                                   move_eos_to_beginning=False)
                ids = torch.cat([ids, ids_char_pad.view(-1)])
            nseq += 1
            ntok += len(ids)
            consumer(ids)
            line = f.readline()
    return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}


def binarize(args, filename, vocab, output_prefix, lang, offset, end, char_vocab, append_eos=True):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args.dataset_impl, vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    res = str_to_bin(filename, vocab, consumer, char_vocab, append_eos=append_eos,
                     offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def prepare_raw_data(args, word_src_dict, word_tgt_dict, char_dict):
    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers, char_vocab=None):
        print("| [{}] Dictionary: {} types".format(lang, len(vocab) - 1))
        if char_vocab is not None:
            print("| [{}] Char Dictionary: {} types".format(lang, len(char_vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                        char_vocab
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                          impl=args.dataset_impl, vocab_size=len(vocab))
        merge_result(
            str_to_bin(
                input_file, vocab, lambda t: ds.add_item(t), char_vocab,
                offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    # if args.trainpref:
    #     make_binary_dataset(word_src_dict, args.trainpref, "train", args.source_lang, num_workers=args.workers,
    #                         char_vocab=char_dict)
    #     make_binary_dataset(word_tgt_dict, args.trainpref, "train", args.target_lang, num_workers=args.workers)
    if args.validpref:
        make_binary_dataset(word_src_dict, args.validpref, "valid", args.source_lang, num_workers=args.workers,
                            char_vocab=char_dict)
        make_binary_dataset(word_tgt_dict, args.validpref, "valid", args.target_lang, num_workers=args.workers)


def prepare_dict(args):
    utils.import_user_module(args)

    print(args)

    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False, word_level=True):
        assert src ^ tgt
        return task.build_dict(
            filenames,
            word_level=word_level,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    if os.path.exists(dict_path(args.source_lang)) and \
            os.path.exists(dict_path(args.target_lang)) and \
            os.path.exists(os.path.join(args.destdir, 'dict_char.txt')):
        return task.load_dictionary(dict_path(args.source_lang)), \
               task.load_dictionary(dict_path(args.target_lang)), \
               task.load_dictionary(os.path.join(args.destdir, 'dict_char.txt'))

    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True
            )
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], src=True)

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)
        else:
            tgt_dict = None

    src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    char_dict = build_dictionary(
        {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True, word_level=False
    )

    # print(src_dict)
    char_dict.save(os.path.join(args.destdir, 'dict_char.txt'))
    return src_dict, tgt_dict, char_dict


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    src_dict, tgt_dict, char_dict = prepare_dict(args)
    prepare_raw_data(args, src_dict, tgt_dict, char_dict)


if __name__ == "__main__":
    import sys

    postag_utils.import_user_module('./plugin')
    sys.argv += [
        '--source-lang', 'src',
        '--target-lang', 'tgt',
        '--task', 'tone_recovery',
        '--trainpref', 'data-bin/tone_recovery_ecom/raw/train',
        '--validpref', 'data-bin/tone_recovery_ecom/raw/valid',
        '--destdir', 'data-bin/tone_recovery_ecom/preprocessed/',
        '--nwordstgt', '70000',
        '--nwordssrc', '70000',
        '--workers', '20',
        '--joined-dictionary',
    ]
    cli_main()
