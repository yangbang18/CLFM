'''This file is responsible for calculating 
    1) the loss with respect to different gaussian noises (we set noise = 0 by default);
    2) the sum of eigenvalues values of the empirical Fisher information matrix;

    Note that we typically select the model at the end of training for evaluation,
    i.e., the model should have learned all tasks.
    
    See reproducibility/run_analysis_converge.sh for a running example
'''
import os
import logging
import argparse
import configs

from clfm import LoggingHandler
from clfm import CLFramework as Framework
from clfm.losses import get_loss_class
from clfm.datasets import CLDataset, get_concat_dataset_and_loader
from clfm.models import AdapterEncoder


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                        help='If not specified, we load model from {model_root}/{arch}_{method}/{order}/{the_last_task}')    
    parser.add_argument('--model_root', type=str, default='output/coco/CL')
    parser.add_argument('--arch', type=str, default='B16', choices=['B16', 'B32', 'L14'])
    parser.add_argument('--method', type=str, default='CLL_CLIP')
    parser.add_argument('--order', type=str, default='order222')

    parser.add_argument('--gaussian_noise_std', type=float, default=0)
    parser.add_argument('--fisher', action='store_true')

    # Data settings
    parser.add_argument('--train_dataset', type=str, default='coco')
    parser.add_argument('--train_corpus_format', type=str, 
                        default="data/corpus/multilingual_{dataset}/36langs/triplet/{dataset}_vision-{source}-{target}.tsv.gz")
    parser.add_argument('--numpy_path_format', type=str, 
                        default='data/corpus/multilingual_{dataset}/36langs/embeddings/{model_name}_sentence_embeddings.npy', 
                        help='Path to a numpy file that stores sentence embeddings')
    parser.add_argument('--vision_numpy_path_format', type=str, 
                        default='data/corpus/multilingual_{dataset}/36langs/embeddings/{model_name}_vision_embeddings.npy', 
                        help='Path to a numpy file that stores vision embeddings')

    parser.add_argument('--source_language', type=str, default='en', choices=['en'], 
                        help='The teacher model accepts English (en) sentences')
    parser.add_argument('--target_languages', type=str, nargs='+', default=configs.xm3600_langs, 
                        help='The languages to be learned by the student model')
    parser.add_argument('--max_sentences', type=int, help='maximun number of sentences per file')
    parser.add_argument('--max_ratio', type=float, help='maximun ratio of training samples')
    parser.add_argument('--weights', type=int, nargs='+', help='If more than one dataset is loaded with load_data: With which frequency should data be sampled from this dataset?')
    parser.add_argument('--num_workers', type=int, default=0, help='# workers to load data for training')

    # Training settings
    parser.add_argument('--crosslingual_loss_scale', type=float, default=1.0, help='Training with bitext pairs')
    parser.add_argument('--crosslingual_loss_type', type=str, default='mse', choices=['mse', 'contrastive'])
    parser.add_argument('--crossmodal_loss_scale', type=float, default=0.01, help='Training with vision-text pairs')
    parser.add_argument('--crossmodal_loss_type', type=str, default='contrastive', choices=['mse', 'contrastive'])
    parser.add_argument('--temperature', type=float, default=0.07)

    parser.add_argument('--use_amp', action='store_true', help='Whether use automatic mixed precision (amp) to speed up training')
    parser.add_argument('--all_parameters', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--train_embeddings', action='store_true')

    # Output settings
    parser.add_argument('--save_root', type=str, default='./')
    parser.add_argument('--save_fn', type=str, default='analysis_results.txt')
    parser.add_argument('--log_every', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    output_path = os.path.join(args.save_root, args.save_fn)

    logger.info(f"Output path: {output_path}")
    logger.info(f"Target languages: {args.target_languages}")

    ######## Model ########
    if not args.model_path:
        tasks = getattr(configs, f'xm3600_{args.order}', None)
        assert tasks is not None, f'xm3600_{args.order} can not be found in configs'
        the_last_task = tasks[-1]
        model_path = os.path.join(args.model_root, f"{args.arch}_{args.method}", args.order, the_last_task)
    else:
        model_path = args.model_path
        assert args.order in model_path, f'The string `{args.order}` can not be found in {model_path}'

    logger.info(f"Create student model from {model_path}")
    model = Framework(model_path, load_sbert_only=True, logger=logger)
    modules = model.get_modules()

    encoder = modules[0]
    assert isinstance(encoder, AdapterEncoder)

    if args.fisher and args.train_embeddings:
        logger.info(f'Make model\'s word embeddings trainable while keep others frozen')
        encoder.freeze_model()
        embeddings = encoder.get_word_embeddings()
        for n, p in embeddings.named_parameters():
            p.requires_grad = True
    
    logger.info(f'Model architecture: \n {model}')
    logger.info(f'\n{encoder.summary()}')
    logger.info("Total Params: {:,}".format(sum(p.numel() for p in model.parameters())))
    if not args.fisher:
        logger.info(f"Gaussian noise std: {args.gaussian_noise_std}")

    ###### Prepare dataset and loader ######
    teacher_model_name = encoder.base_model_name
    train_dataset = args.train_dataset
    CL_datasets = []
    sentence_embedding_cache = vision_embedding_cache = None
    for target_language in args.target_languages:
        annotation_path = args.train_corpus_format.format(
            dataset=train_dataset, 
            source=args.source_language, 
            target=target_language,
        )
        sentence_embedding_numpy_path = args.numpy_path_format.format(
            dataset=train_dataset, 
            model_name=teacher_model_name.replace('/', '_'),
        )
        vision_embedding_numpy_path = args.vision_numpy_path_format.format(
            dataset=train_dataset, 
            model_name=teacher_model_name.replace('/', '_'),
        )
        CL_dataset = CLDataset(
            annotation_path=annotation_path,
            languages=['vision', args.source_language, target_language],
            sentence_embedding_numpy_path=sentence_embedding_numpy_path,
            vision_embedding_numpy_path=vision_embedding_numpy_path,
            max_sentences=args.max_sentences,
            max_ratio=args.max_ratio,
            logger=logger,
            sentence_embedding_cache=sentence_embedding_cache,
            vision_embedding_cache=vision_embedding_cache,
        )
        sentence_embedding_cache = CL_dataset.sentence_embedding_cache
        vision_embedding_cache = CL_dataset.vision_embedding_cache
        CL_datasets.append(CL_dataset)
    
    _, loader = get_concat_dataset_and_loader(
        datasets=CL_datasets, 
        weights=args.weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )
    loader.collate_fn = Framework.smart_batching_collate

    ###### Define Losses ######
    
    LOSS_MODEL_CLASS = get_loss_class(class_name='oEWC' if args.fisher else 'CLLoss')
    
    loss_model = LOSS_MODEL_CLASS(
        model=model,
        crosslingual_loss_scale=args.crosslingual_loss_scale,
        crosslingual_loss_type=args.crosslingual_loss_type,
        crossmodal_loss_scale=args.crossmodal_loss_scale,
        crossmodal_loss_type=args.crossmodal_loss_type,
        temperature=args.temperature,
    )

    if args.fisher:
        if args.use_amp:
            import torch
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        
        fisher = loss_model.end_task(
            train_loader=loader,
            checkpoint_path=None,
            use_amp=args.use_amp,
            scaler=scaler,
            return_fisher_only=True,
        )
        logger.info(f'fisher sum: {fisher.sum()}')
        with open(output_path, 'a') as wf:
            wf.write(f'{args.model_path or args.method}\t{fisher.sum()}\n')
    else:
        info = loss_model.test_perturbation(
            train_loader=loader,
            gaussian_noise_std=args.gaussian_noise_std,
            use_amp=args.use_amp,
            log_every=args.log_every,
            seed=args.seed,
            all_parameters=args.all_parameters,
        )
        logger.info(f"info: {info}")
        with open(output_path, 'a') as wf:
            wf.write(f'{args.model_path or args.method}\t{args.gaussian_noise_std}\t{info["loss"]}\n')
