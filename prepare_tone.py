"""
Data pre-processing: build vocabularies and binarize training data.
"""
from fairseq_cli import preprocess
import utils as postag_utils
import unidecode


def remove_tone(tgt_file, src_file):
    with open(tgt_file, 'r', encoding='utf-8') as file_text:
        lines = file_text.read().split('\n')
        with open(src_file, 'w', encoding='utf-8') as file_out:
            for line in lines:
                file_out.write('{}\n'.format(unidecode.unidecode(line)))


if __name__ == "__main__":
    import sys

    # remove_tone('data-bin/tone_recovery_ecom/raw/train.tgt', 'data-bin/tone_recovery_ecom/raw/train.src')
    # remove_tone('data-bin/tone_recovery_ecom/raw/valid.tgt', 'data-bin/tone_recovery_ecom/raw/valid.src')

    postag_utils.import_user_module('./plugin')
    sys.argv += [
        '--source-lang', 'src',
        '--target-lang', 'tgt',
        '--task', 'tone_recovery',
        '--trainpref', 'data-bin/tone_recovery_ecom/raw/train',
        '--validpref', 'data-bin/tone_recovery_ecom/raw/valid',
        # '--testpref', 'data-bin/tone_recovery_ecom/raw/test',
        '--destdir', 'data-bin/tone_recovery_ecom/processed_cached/',
        '--workers', '10',
        '--nwordstgt', '70000',
        '--nwordssrc', '70000',
        # '--dataset-impl', 'raw',
        '--joined-dictionary',
    ]
    preprocess.cli_main()
