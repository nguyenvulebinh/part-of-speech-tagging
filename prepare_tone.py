"""
Data pre-processing: build vocabularies and binarize training data.
"""
from fairseq_cli import preprocess
import utils as postag_utils

if __name__ == "__main__":
    import sys

    postag_utils.import_user_module('./plugin')
    sys.argv += [
        '--source-lang', 'src',
        '--target-lang', 'tgt',
        '--task', 'tone_recovery',
        '--trainpref', 'data-bin/tone_recovery_ecom/raw/train',
        '--validpref', 'data-bin/tone_recovery_ecom/raw/valid',
        # '--testpref', 'data-bin/tone_recovery_ecom/raw/test',
        '--destdir', 'data-bin/tone_recovery_ecom/processed_cached/',
        '--workers', '5',
        # '--dataset-impl', 'raw',
        '--joined-dictionary',
    ]
    preprocess.cli_main()
