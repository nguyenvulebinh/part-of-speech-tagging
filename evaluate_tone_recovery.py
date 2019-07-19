from fairseq import checkpoint_utils, data, options, tasks
import utils as postag_utils
from fairseq import utils
import torch
from tqdm import tqdm

checkpoint_path = './checkpoints/transformer/checkpoint24.pt'
adaptive_softmax_cutoff = [5000, 20000]


def infer(task, sentences, use_cuda):
    samples_list = [
        {'id': idx, 'source': task.source_dictionary.encode_line(
            sen, add_if_not_exist=False,
        )}
        for idx, sen in enumerate(sentences)
    ]

    # Build mini-batch to feed to the model
    batch = data.language_pair_dataset.collate(
        samples=samples_list,  # bsz = 1
        pad_idx=task.source_dictionary.eos(),
        eos_idx=task.source_dictionary.eos(),
        left_pad_source=False,
        input_feeding=False,
    )

    # Feed batch to the model and get predictions
    batch = utils.move_to_cuda(batch) if use_cuda else batch
    preds = model(batch['net_input']['src_tokens'].long())
    # Print top k predictions and their log-probabilities
    results = []

    for pred in preds[0]:

        ouput_indx = model.decoder.adaptive_softmax.head(pred).argmax(1)
        print(ouput_indx)
        for idx, tail_model in enumerate(model.decoder.adaptive_softmax.tail):

            tail_input_idx = (model.decoder.adaptive_softmax.head(pred).argmax(1) == adaptive_softmax_cutoff[0] + idx) \
                .nonzero() \
                .view(-1)
            if len(tail_input_idx) > 0:
                tail_input = torch.index_select(pred, 0, tail_input_idx)
                tail_output = tail_model(tail_input).argmax(1) + adaptive_softmax_cutoff[idx]
                ouput_indx[tail_input_idx] = tail_output
        results.append(task.target_dictionary.string(ouput_indx))
    # recovery order as input
    _, results = zip(*sorted(zip(batch['id'].tolist(), results)))
    results = [' '.join(item.split()[:len(input_sen.split())]) for item, input_sen in zip(results, sentences)]
    return results


if __name__ == '__main__':
    import sys

    postag_utils.import_user_module('./plugin')
    sys.argv += [
        './data-bin/tone_recovery_ecom/processed_cached/',
        '--user-dir', './plugin',
        '--task', 'tone_recovery',
        '-a', 'transformer_tone',
        '-s', 'src', '-t', 'tgt',
        '--max-tokens', '4000',
        # '--cpu'
    ]

    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    use_cuda = torch.cuda.is_available() and not args.cpu
    # Print args
    print(args)
    # Setup task, e.g., translation, language modeling, etc.
    task_tone_recovery = tasks.setup_task(args)
    # Load model
    print('| loading model from {}'.format(checkpoint_path))
    models, _model_args = checkpoint_utils.load_model_ensemble([checkpoint_path], task=task_tone_recovery)
    model = models[0]
    model.eval()
    if use_cuda:
        model.cuda()
    print(model)

    # infer batch
    # with open('./data-bin/src-testx-w.txt') as file_test:
    #     lines = file_test.read().split('\n')
    #
    # with open('./data-bin/tgt-testx-w.txt', 'w', encoding='utf-8') as file_tgt:
    #     batch_size = 100
    #
    #
    #     def make_batch(items, group_size):
    #         for i in range(0, len(items), group_size):
    #             yield items[i:i + group_size]
    #
    #
    #     iter_input = make_batch(lines, group_size=batch_size)
    #     for sentences in tqdm(iter_input, total=len(list(make_batch(lines, group_size=batch_size)))):
    #         results = infer(task_tone_recovery, sentences, use_cuda)
    #         for sen_result in results:
    #             file_tgt.write('{}\n'.format(sen_result))

    # infer sentence
    while True:
        sentence = input('\nInput: ')
        print(infer(task_tone_recovery, [sentence], use_cuda))
