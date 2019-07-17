from fairseq import checkpoint_utils, data, options, tasks
import utils as postag_utils
from fairseq import utils
import torch
from tqdm import tqdm
# postag_utils.import_user_module('./plugin')
checkpoint_path = './checkpoints/transformer/checkpoint_best.pt'
# # Parse command-line arguments for generation
# parser = options.get_generation_parser(default_task='tone_recovery')
# args = options.parse_args_and_arch(parser)
#


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
        # '--path', './checkpoints/transformer/checkpoint_best.pt'

        # '--encoder-embed-dim', '64',
        # '--encoder-ffn-embed-dim', '128',
        # '--encoder-attention-heads', '2',
        # '--encoder-layers', '2',
        # '--decoder-embed-dim', '64',
        # '--decoder-ffn-embed-dim', '128',
        # '--decoder-attention-heads', '2',
        # '--decoder-layers', '2'
    ]

    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    # Print args
    print(args)
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)
    # Load model
    print('| loading model from {}'.format(checkpoint_path))
    models, _model_args = checkpoint_utils.load_model_ensemble([checkpoint_path], task=task)
    model = models[0]
    print(model)

    with open('./data-bin/src-testx-w.txt') as file_test:
        lines = file_test.read().split('\n')

    with open('./data-bin/tgt-testx-w.txt', 'w', encoding='utf-8') as file_tgt:
        for sentence in tqdm(lines):
            tokens = task.source_dictionary.encode_line(
                sentence, add_if_not_exist=False,
            )

            # Feed batch to the model and get predictions
            preds = model(tokens.unsqueeze(0).long())

            # Print top 3 predictions and their log-probabilities
            top_scores, top_labels = preds[0].topk(k=3)
            file_tgt.write('{}\n'.format(task.target_dictionary.string(top_labels.squeeze(0).t()[0])))
    # while True:
    #     sentence = input('\nInput: ')
    #
    #     tokens = task.source_dictionary.encode_line(
    #         sentence, add_if_not_exist=False,
    #     )
    #
    #     # Build mini-batch to feed to the model
    #     batch = data.language_pair_dataset.collate(
    #         samples=[{'id': -1, 'source': tokens}],  # bsz = 1
    #         pad_idx=task.source_dictionary.pad(),
    #         eos_idx=task.source_dictionary.eos(),
    #         left_pad_source=False,
    #         input_feeding=False,
    #     )
    #
    #     # Feed batch to the model and get predictions
    #     preds = model(tokens.unsqueeze(0).long())
    #
    #     # Print top 3 predictions and their log-probabilities
    #     top_scores, top_labels = preds[0].topk(k=3)
    #     for idx, labels in enumerate(top_labels.squeeze(0).t()):
    #         print(idx, task.target_dictionary.string(labels))
    #     # for score, label_idx in zip(top_scores.squeeze(0).detach().numpy(), top_labels.squeeze(0).detach().numpy()):
    #     #     label_name = task.target_dictionary.string(label_idx)
    #     #     print('({:.2f})\t{}'.format(score, label_name))
