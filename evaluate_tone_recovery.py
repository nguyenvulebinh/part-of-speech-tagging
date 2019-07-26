from fairseq import checkpoint_utils, data, options, tasks
import utils as postag_utils
from fairseq import utils
import torch
from plugin.data import spell_correct_dataset
from tqdm import tqdm

checkpoint_path = './checkpoints/transformer/checkpoint_best.pt'


def infer(task, sentences, use_cuda):
    samples_list = [
        {
            'id': idx,
            'source': task.source_dictionary.encode_line(sen, add_if_not_exist=False).long(),
            'source_char': [task.char_dict.encode_line(' '.join(list(word)),
                                                       add_if_not_exist=False).long() for word in sen.split() + ['']]
        } for idx, sen in enumerate(sentences)
    ]

    # Build mini-batch to feed to the model
    batch = spell_correct_dataset.collate(
        samples_list, pad_idx=task.source_dictionary.pad(), eos_idx=task.source_dictionary.eos(),
        left_pad_source=True, left_pad_target=False,
        input_feeding=True, word_max_length=15
    )

    # Feed batch to the model and get predictions
    batch = utils.move_to_cuda(batch) if use_cuda else batch
    preds = model(**batch['net_input'])
    # Print top k predictions and their log-probabilities
    results = []
    for pred in preds[0]:
        top_scores, top_labels = pred.topk(k=1)
        for item in top_labels.squeeze(0).t():
            results.append(task.target_dictionary.string(item))
    # recovery order as input
    _, results = zip(*sorted(zip(batch['id'].tolist(), results)))
    return results


if __name__ == '__main__':
    import sys

    postag_utils.import_user_module('./plugin')
    sys.argv += [
        './data-bin/dict',
        '--user-dir', './plugin',
        '--task', 'tone_recovery',
        # '-a', 'transformer_tone',
        '-s', 'src', '-t', 'tgt',
        '--max-tokens', '4000',
        # '--cpu'
    ]

    parser = options.get_eval_lm_parser()
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
    # model.eval()
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
        if len(sentence) > 0:
            print(infer(task_tone_recovery, [sentence], use_cuda))
