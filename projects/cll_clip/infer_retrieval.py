import os
import time
import glob
import datetime
import argparse
import logging
import configs

from torch.utils.data import DataLoader
from clfm import Seq2Seq, LoggingHandler
from clfm.datasets import CaptionDatasetForRetrieval
from clfm.evaluation import RetrievalEvaluator
from clfm.utils import get_formatted_string
from clfm.models import AdapterEncoder

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

try:
    ROOT = configs.annotation_retrieval_root
except:
    ROOT = configs.annotation_root


def get_loader(args, ann_rpath, mode):
    logger.info(f'Load dataset from {ann_rpath}')

    pickle_path = get_formatted_string(vars(args), 'pickle_path', assigned_kwargs=dict(
        dataset=args.dataset, clip_model_name=model.clip_model_name.replace('/', '_'), mode=mode,
    ))
    
    dataset = CaptionDatasetForRetrieval(
        vision_root=configs.image_video_root[args.dataset],
        ann_rpath=ann_rpath,
        num_frames=args.num_frames,
        lang=args.lang,
        clip_model=model.clip_model,
        pickle_path=pickle_path,
        logger=logger,
        mean_pooling=args.mean_pooling,
    )
    logger.info(f'There are {len(dataset)} vision inputs')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    return loader


def run(args, model, output_path):
    # start evaluation
    start_time = time.time()
    for mode in args.modes:
        ann_rpath = get_formatted_string(vars(args), f"{mode}_file", assigned_keys=['dataset', 'mid_path', 'lang'])
        loader = get_loader(args, ann_rpath, mode)

        fn_prefix = None
        if args.mid_path:
            fn_prefix = 'translated'
        if args.do_fusion:
            fn_prefix = 'fused'
            assert args.mid_path
            mid_path, args.mid_path = args.mid_path, ''
            ann_rpath = get_formatted_string(vars(args), f"{mode}_file", assigned_keys=['dataset', 'mid_path', 'lang'])
            loader = [loader, get_loader(args, ann_rpath, mode)]
            args.mid_path = mid_path

        evaluator = RetrievalEvaluator(
            loader=loader,
            mode=mode,
            logger=logger,
            n_fold=args.n_fold or 1, 
            do_fusion=args.do_fusion,
            fusion_scales=args.fusion_scales,
            fusion_types=args.fusion_types,
        )

        evaluator(model, output_path=output_path, fn_prefix=fn_prefix)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--clip_model_name', type=str)
    parser.add_argument('--skip_if_exists', action='store_true')

    parser.add_argument('--active_adapter', type=str)
    parser.add_argument('--wisely_activate_adapter', action='store_true')
    parser.add_argument('--active_adapter_fusion', type=str)
    parser.add_argument('--wisely_activate_adapter_fusion', action='store_true')

    # Data paths and attributes
    parser.add_argument('--data_root', type=str, default=ROOT)
    parser.add_argument('--dataset', type=str, default='coco', choices=list(configs.image_video_root.keys()))
    parser.add_argument('--val_file', type=str, help='If not specified, use val_file_format')
    parser.add_argument('--test_file', type=str, help='If not specified, use test_file_format')
    parser.add_argument('--pickle_path', type=str, help='If not specified, use pickle_path_format')
    parser.add_argument('--val_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{mid_path}/{lang}/val.json'))
    parser.add_argument('--test_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{mid_path}/{lang}/test.json'))
    parser.add_argument('--pickle_path_format', type=str, default=os.path.join(ROOT, '{dataset}/{clip_model_name}_{mode}.pkl'))
    parser.add_argument('--mid_path', type=str, default='')
    
    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Evaluation settings
    parser.add_argument('--modes', type=str, nargs='+', default=['test'], help='evaluation modes: ["val"], ["test"], ["val", "test"]')
    parser.add_argument('--lang', type=str, default='en', help='which language to be generated?')
    parser.add_argument('--langs', type=str, nargs='+', help='which languages to be generated?')
    parser.add_argument('--num_frames', type=int, default=configs.num_frames)
    parser.add_argument('--mean_pooling', action='store_true')
    parser.add_argument('--n_fold', type=int, help='e.g., you can set n_fold to 5 to get additional 1K test results on MSCOCO')

    parser.add_argument('--do_fusion', action='store_true')
    parser.add_argument('--fusion_types', type=str, nargs='+', default=['feature', 'score'])
    parser.add_argument('--fusion_scales', type=float, nargs='+', default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Output settings
    parser.add_argument('--output_path', type=str, help='If not specified, output_path will be {model}/evaluations_retrieval/{dataset}/{lang}')
    parser.add_argument('--no_suffix_folder', action='store_true', help='If True, the suffix `evaluations_retrieval/{dataset}/{lang}` will not be joined to the output path')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.info(f"{args.model} can not be found in local directory")
        assert args.output_path, "you are trying to load a model from hugginface hub, please specify --output_path"

    assert args.modes in [['val'], ['test'], ['val', 'test']]

    start_time = time.time()

    langs = args.langs or [args.lang]
    logger.info(f'languages: {langs}')

    model = None
    for i, lang in enumerate(langs):
        args.lang = lang

        output_path = args.output_path or args.model
        if not args.no_suffix_folder: 
            output_path = os.path.join(output_path, 'evaluations_retrieval', args.dataset, args.lang)
        else:
            assert len(langs) == 1

        if args.skip_if_exists and all(len(glob.glob(os.path.join(output_path, f'*{mode}_scores.json'))) > 0 for mode in args.modes):
            logger.info(f'Evaluation of {args.model} on {args.dataset} dataset\'s {args.modes} splits in `{args.lang}` language has been done')
            continue

        if model is None:
            logger.info(f'Creating model from {args.model}')
            model = Seq2Seq(args.model, args.clip_model_name)
            encoder = model.multilingual_model._first_module()

        adapter_fusion_name = None
        if isinstance(encoder, AdapterEncoder):
            if (args.active_adapter_fusion is not None \
                or args.wisely_activate_adapter \
                or args.wisely_activate_adapter_fusion) \
                and encoder.train_fusion is not None:
                transformer = encoder.get_transformer()
                transformer.delete_adapter_fusion(encoder.train_fusion)
                encoder.train_fusion = None

            if i == 0:
                if args.active_adapter_fusion is not None:
                    adapter_fusion_name = encoder.load_adapter_fusion_wisely(adapter_fusion_name=args.active_adapter_fusion)
                if adapter_fusion_name: 
                    logger.info(f'activate the adapter fusion `{adapter_fusion_name}` for all languages to be evaluated')
                elif args.active_adapter is not None:
                    flags = encoder.set_active_adapter(args.active_adapter)
                    if any(flags): 
                        logger.info(f'activate the adapter `{args.active_adapter}` for all languages to be evaluated')
            
            if args.wisely_activate_adapter_fusion:
                adapter_fusion_name = encoder.load_adapter_fusion_wisely(identity=lang)
            if adapter_fusion_name: 
                logger.info(f'activate the adapter fusion `{adapter_fusion_name}`')
            elif args.wisely_activate_adapter:
                flags = encoder.set_active_adapter(lang)
                if any(flags): 
                    logger.info(f'activate the adapter `{lang}`')
            
            print(encoder.summary())

        os.makedirs(output_path, exist_ok=True)

        handler = logging.FileHandler(os.path.join(output_path, 'log.txt'), 'w', encoding='utf-8')
        logger.addHandler(handler)
        logger.info(f'output path: {output_path}')
        logger.info(f'model: {args.model}')        
        
        run(args, model, output_path)
        logger.removeHandler(handler)
        print('-' * 100)

        if adapter_fusion_name:
            transformer.delete_adapter_fusion(adapter_fusion_name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total Time {}'.format(total_time_str))
