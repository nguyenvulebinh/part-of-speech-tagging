#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fairseq_cli import train
import utils as postag_utils

if __name__ == '__main__':
    import sys

    postag_utils.import_user_module('./plugin')
    sys.argv += [
        './data-bin/tone_recovery_ecom/processed_cached/',
        '--user-dir', './plugin',
        '--task', 'tone_recovery',
        '-a', 'transformer_tone',
        '--optimizer', 'adam',
        '--lr', '0.0005',
        '-s', 'src', '-t', 'tgt',
        '--label-smoothing', '0.1',
        '--dropout', '0.3',
        '--max-tokens', '4000',
        '--min-lr', '1e-09',
        '--lr-scheduler', 'inverse_sqrt',
        '--weight-decay', '0.0001',
        '--criterion', 'label_smoothed_cross_entropy',
        '--max-update', '50000',
        '--warmup-updates', '4000',
        '--warmup-init-lr', '1e-07',
        '--adam-betas', '(0.9,0.98)',
        '--save-dir', 'checkpoints/transformer',
        # '--dataset-impl', 'raw',
        '--share-all-embeddings',




        # '--encoder-embed-dim', '64',
        # '--encoder-ffn-embed-dim', '128',
        # '--encoder-attention-heads', '2',
        # '--encoder-layers', '2',
        # '--decoder-embed-dim', '64',
        # '--decoder-ffn-embed-dim', '128',
        # '--decoder-attention-heads', '2',
        # '--decoder-layers', '2'
    ]
    train.cli_main()
