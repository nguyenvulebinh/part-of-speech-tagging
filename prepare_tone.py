"""
Data pre-processing: build vocabularies and binarize training data.
"""
import utils as postag_utils
from fairseq import options, tasks, utils
import os


def main(args):
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

    # def build_dictionary(filenames, src=False, tgt=False):
    #     assert src ^ tgt
    #     return task.build_dictionary(
    #         filenames,
    #         workers=args.workers,
    #         threshold=args.thresholdsrc if src else args.thresholdtgt,
    #         nwords=args.nwordssrc if src else args.nwordstgt,
    #         padding_factor=args.padding_factor,
    #     )

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


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    import sys

    postag_utils.import_user_module('./plugin')
    sys.argv += [
        '--source-lang', 'src',
        '--target-lang', 'tgt',
        '--task', 'tone_recovery',
        '--trainpref', 'data-bin/tone_recovery_ecom/raw_dev/train',
        '--validpref', 'data-bin/tone_recovery_ecom/raw_dev/valid',
        '--destdir', 'data-bin/tone_recovery_ecom/raw_dev/',
        '--nwordstgt', '70000',
        '--nwordssrc', '70000',
        '--workers', '10',
        '--joined-dictionary',
    ]
    cli_main()
