import os
import sys
import time
import json
import configs
import logging
import argparse
import clfm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from adapters import AdapterConfig, ConfigUnion
from clfm import LoggingHandler
from clfm import CLFramework as Framework
from clfm.datasets import CaptionDatasetForRetrieval
from clfm.evaluation import RetrievalEvaluator
from clfm.losses import get_loss_class
from clfm.Constants import CLIP_MODELS
from clfm.utils import get_formatted_string, load_yaml
from clfm.datasets import CLDataset, MemoryDataset, get_concat_dataset_and_loader, prepare_token_mapping
from clfm.models import AdapterEncoder


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    # Model settings
    parser.add_argument('--teacher_model_name', type=str, 
                        default='openai/clip-vit-base-patch16', choices=CLIP_MODELS, 
                        help='Teacher model that provides visual/text features to be aligned')
    parser.add_argument('--student_model_name', type=str, 
                        help='If specified, load pre-trained models from Hugginface or the local path')
    parser.add_argument('--base_model_name', type=str, 
                        default='openai/clip-vit-base-patch16',
                        help='The pre-trained backbone of the student model')
    parser.add_argument('--embedding_name', type=str,
                        default='openai/clip-vit-base-patch16',
                        help='The pre-trained embedding layer of the student model')
    parser.add_argument('--dim_word', type=int)
    parser.add_argument('--mean_std_word', type=float, nargs='+')

    parser.add_argument('--freeze_model', action='store_true')
    parser.add_argument('--train_word_embs', action='store_true')
    parser.add_argument('--no_shrink', action='store_true')

    parser.add_argument('--regularize', action='store_true')
    parser.add_argument('--regularize_type', type=str, default='grad_decay', choices=['grad_decay', 'grad', 'decay'], 
                        help='grad_decay:   use self-defined AdamW to regularize gradients and weight decay;\
                              grad:         use self-defined AdamW to regularize gradients only \
                              decay:        use self-defined AdamW to regularize weight decay only')
    parser.add_argument('--grad_scale_type', type=str, default='current_reciprocal')
    parser.add_argument('--weight_decay_scale_type', type=str, default='current')
    parser.add_argument('--exclude_special_tokens', action='store_true')

    parser.add_argument('--dynamic_vocab', action='store_true', 
                        help='incrementally add new tokens into the vocabulary, rather than using a shared vocabulary all the time')
    parser.add_argument('--oracle_vocab', action='store_true',
                        help='use a oracle vocabulary learned from the corpus of all languages')
    parser.add_argument('--new_vocab_size', type=int, default=10000)
    parser.add_argument('--new_vocab_mean_std', type=float, nargs='+', default=[0, 0.02])
    parser.add_argument('--directly_use_new_tokenizer', action='store_true')

    parser.add_argument('--adapter_name', type=str)
    parser.add_argument('--adapter_config', type=str, default='')
    parser.add_argument('--embedding_rank', type=int, help='the rank of a embedding LoRA')
    parser.add_argument('--embedding_config', type=type, help='Path to the configuration file of a embedding LoRA')

    ## arguments for oEWC
    parser.add_argument('--fisher_penalty_scale', type=float, default=0)
    parser.add_argument('--fisher_gamma', type=float, default=1.0)
    ## arguments for ER and DER
    parser.add_argument('--experience_replay', action='store_true')
    parser.add_argument('--dark_experience_replay', action='store_true')
    parser.add_argument('--memory_size', type=int, default=8000)
    ## arguments for DualPrompt and CodaPrompt
    parser.add_argument('--with_prompt', action='store_true')
    parser.add_argument('--prompt_type', type=str, default='CodaPrompt', choices=['DualPrompt', 'CodaPrompt'])
    parser.add_argument('--prompt_config', type=str, default='default')
    parser.add_argument('--pull_constraint_coeff', type=float, default=0)

    # Student's encoder settings
    parser.add_argument('--max_seq_length', type=int, default=128, 
                        help='Student model max. lengths for inputs (number of word pieces)')

    # Data & Loader settings
    parser.add_argument('--train_datasets', type=str, nargs='+', default=['coco'])
    parser.add_argument('--train_corpus_format', type=str, 
                        default="%s/multilingual_{dataset}/36langs/triplet/{dataset}_vision-{source}-{target}.tsv.gz" % configs.corpus_root)
    parser.add_argument('--numpy_path_format', type=str, 
                        default='%s/multilingual_{dataset}/36langs/embeddings/{model_name}_sentence_embeddings.npy' % configs.corpus_root, 
                        help='Path to a numpy file that stores sentence embeddings')
    parser.add_argument('--vision_numpy_path_format', type=str, 
                        default='%s/multilingual_{dataset}/36langs/embeddings/{model_name}_vision_embeddings.npy' % configs.corpus_root, 
                        help='Path to a numpy file that stores vision embeddings')
    parser.add_argument('--tokenizer_path', type=str, help="if not specified, use tokenizer_path_format")
    parser.add_argument('--tokenizer_path_format', type=str,
                        default="%s/{dataset}/clip/{tag}/{new_vocab_size}" % configs.tokenizer_root)

    parser.add_argument('--source_language', type=str, default='en', choices=['en'], 
                        help='Our teacher model accepts English (en) sentences')
    parser.add_argument('--target_languages', type=str, nargs='+', default=configs.xm3600_langs, 
                        help='The languages to be learned by the student model')
    parser.add_argument('--max_sentences', type=int, help='maximun number of sentences per file')
    parser.add_argument('--max_ratio', type=float, help='maximun ratio of training samples')
    parser.add_argument('--weights', type=int, nargs='+', help='If more than one dataset is loaded with load_data: With which frequency should data be sampled from this dataset?')
    parser.add_argument('--num_workers', type=int, default=0, help='# workers to load data for training')
    parser.add_argument('--balanced_sampling', action='store_true')
    parser.add_argument('--double_batch_size', action='store_true')

    # Training settings
    parser.add_argument('--crosslingual_loss_scale', type=float, default=1.0, help='Training with bitext pairs')
    parser.add_argument('--crosslingual_loss_type', type=str, default='mse', choices=['mse', 'contrastive'])
    parser.add_argument('--crossmodal_loss_scale', type=float, default=0.0, help='Training with vision-text pairs')
    parser.add_argument('--crossmodal_loss_type', type=str, default='contrastive', choices=['mse', 'contrastive'])
    parser.add_argument('--en_loss_scale', type=float, default=0.0)

    parser.add_argument('--use_amp', action='store_true', help='Whether use automatic mixed precision (amp) to speed up training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--inference_batch_size', type=int, help='Batch size at inference; if not speficied, set to batch_size')
    parser.add_argument('--epochs', type=int, default=3, help='Train for x epochs')
    parser.add_argument('--warmup_steps', type=float, default=0.1, help='Warumup steps')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--betas', type=float, nargs='+', default=[0.9, 0.999])
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--scheduler', type=str, default='warmupconstant', 
                        choices=['constantlr', 'warmupconstant', 'warmuplinear', 'warmupcosine', 'warmupcosinewithhardrestarts'])
    parser.add_argument('--no_decay', type=str, nargs='+', default=[])
    parser.add_argument('--checkpoint_save_total_limit', type=int, default=0)

    # Evaluation settings
    parser.add_argument('--do_evaluation', action='store_true', 
                        help='Whether do evaluation to save the best-performed model during training')
    parser.add_argument('--monitor', type=str, default='r_mean', 
                        help='Which metric (the higher, the better) to be monitored to save the model')
    parser.add_argument('--val_dataset', type=str, default='coco', choices=['coco', 'flickr30k'])
    parser.add_argument('--mode', type=str, default='val', choices=['val'])
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--val_file_format', type=str, 
                        default=os.path.join(configs.annotation_root, '{dataset}/translated/{lang}/val.json'))
    parser.add_argument('--pickle_path', type=str)
    parser.add_argument('--pickle_path_format', type=str, 
                        default=os.path.join(configs.annotation_root, '{dataset}/{clip_model_name}_{mode}.pkl'))
    parser.add_argument('--num_frames', type=int, default=8)

    # Output settings
    parser.add_argument('--output_path', type=str, help='The exact output path to save training info and checkpoints')
    parser.add_argument('--output_root', type=str, default='output/CLL')
    parser.add_argument('--exp_name', type=str, default='debug', help='Experiment name; If `output_path` is not specified, the output path will be {output_root}/{exp_name}')
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--override', action='store_true')
    args = parser.parse_args()

    if args.method:
        load_yaml(args, key=args.method, yaml_path='methods.yaml')
    
    if not args.inference_batch_size:
        args.inference_batch_size = args.batch_size
    
    if args.dynamic_vocab:
        assert args.embedding_name == args.base_model_name

    output_path = args.output_path or os.path.join(args.output_root, args.exp_name)
    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(os.path.join(output_path, 'modules.json')) and not args.override:
        logger.info('The model has been trained! If you want to override it, please pass the argument `--override`')
        sys.exit(0)

    # saving training logs to {output_path}/log.txt
    writter = SummaryWriter(log_dir=os.path.join(output_path, 'logs'))
    logger.addHandler(logging.FileHandler(os.path.join(output_path, 'log.txt'), 'w', encoding='utf-8'))
    logger.info(f"Output path: {output_path}")
    logger.info(f"Target languages: {args.target_languages}")

    ### Define the teacher model
    logger.info(f"Load teacher model: {args.teacher_model_name}")
    teacher_model = Framework(args.teacher_model_name)
    logger.info(f'Teacher model architecture: \n {teacher_model}')
    teacher_model = teacher_model.to(teacher_model.device).eval()
    dim_teacher = teacher_model._last_module().get_sentence_embedding_dimension()
    # freeze teacher model
    for p in teacher_model.parameters():
        p.requires_grad = False

    ### Define the student model
    if not args.student_model_name:
        logger.info(f"Create student model from pre-trained models")

        encoder = AdapterEncoder(
            base_model_name=args.base_model_name,
            max_seq_length=args.max_seq_length,
            clip_model_name=args.teacher_model_name,
            dim_latent_space=dim_teacher,
            embedding_name=args.embedding_name,
            shrink_embeddings=not args.no_shrink,
            dim_word=args.dim_word,
            mean_std_word=args.mean_std_word,
            with_prompt=args.with_prompt,
            prompt_type=args.prompt_type,
            prompt_config=args.prompt_config,
        )

        modules = [encoder]
        student_model = Framework(modules=modules, logger=logger)
    else:
        logger.info(f"Create student model from {args.student_model_name}")
        student_model = Framework(args.student_model_name, load_sbert_only=True, logger=logger)
        modules = student_model.get_modules()

    encoder = modules[0]
    assert isinstance(encoder, AdapterEncoder)

    ### Vocab substitution of the student model
    if args.dynamic_vocab:
        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
        elif args.oracle_vocab:
            tokenizer_path = args.tokenizer_path_format.format(
                dataset='-'.join(args.train_datasets), 
                tag='oracle', new_vocab_size=args.new_vocab_size)
        else:
            assert len(args.target_languages) == 1
            tokenizer_path = args.tokenizer_path_format.format(
                dataset='-'.join(args.train_datasets), 
                tag=args.target_languages[0], new_vocab_size=args.new_vocab_size)

        logger.info(f"Update tokenizer by adding vocab in {tokenizer_path} ...")
        msg = encoder.update_tokenizer(
            tokenizer_path=tokenizer_path,
            save_path=output_path,
            mean=args.new_vocab_mean_std[0],
            std=args.new_vocab_mean_std[1],
            msg_prefix='\t',
            directly_use_new_tokenizer=args.directly_use_new_tokenizer,
            plus_one_before_the_frist_task=True,
        )
        logger.info(msg)
    
    ### Add adapters into the student model
    if not args.adapter_name:
        args.adapter_name = '-'.join(args.target_languages)

    components = []
    if args.adapter_config:
        if args.adapter_config.startswith('repo:'):
            args.adapter_config = os.path.join(os.path.dirname(clfm.__file__), args.adapter_config[5:])
        components.append(AdapterConfig.load(args.adapter_config))

    if len(components) or args.embedding_rank or args.embedding_config:
        if len(components) == 1:
            adapter_config = components[0]
        elif len(components):
            adapter_config = ConfigUnion(*components)
        else:
            adapter_config = None
        emb_adapter_config = args.embedding_rank or args.embedding_config

        encoder.setup_adapter(
            args.adapter_name, 
            adapter_config=adapter_config, 
            emb_adapter_config=emb_adapter_config,
            add=True, 
            set_active=True,
            set_train=True, 
        )

    ### Take care of the trainable components in the student model
    if args.freeze_model and len(components) == 0:
        logger.info(f'Freeze student model')
        encoder.freeze_model()

    if (args.freeze_model or len(components)) and args.train_word_embs:
        logger.info(f'Make student model\'s word embeddings trainable')
        embeddings = encoder.get_word_embeddings()
        for n, p in embeddings.named_parameters():
            logger.info(f'\t {n}')
            p.requires_grad = True
    
    if args.with_prompt:
        logger.info(f'Make student model\'s prompt pool trainable')
        embeddings = encoder.get_embeddings()
        for n, p in embeddings.named_parameters():
            if n.startswith('prompt'):
                logger.info(f'\t {n}')
                p.requires_grad = True
    
    ### Log the overview of the student model
    logger.info(f'Student model architecture: \n {student_model}')
    logger.info(f'\n{encoder.summary()}')
    logger.info("Trainable Params: {:,}".format(sum(p.numel() for p in student_model.parameters() if p.requires_grad)))
    logger.info("Total Params: {:,}".format(sum(p.numel() for p in student_model.parameters())))

    ### Prepare dataset and loader
    concat_datasets, loaders = [], []
    for train_dataset in args.train_datasets:
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
                model_name=args.teacher_model_name.replace('/', '_'),
            )
            vision_embedding_numpy_path = args.vision_numpy_path_format.format(
                dataset=train_dataset, 
                model_name=args.teacher_model_name.replace('/', '_'),
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
    
        if args.experience_replay:
            memory_dataset = MemoryDataset(
                memory_size=args.memory_size,
                read_path=None if not args.student_model_name else args.student_model_name,
                save_path=output_path,
                logger=logger,
                dark_experience_replay=args.dark_experience_replay,
            )
            CL_datasets.append(memory_dataset)
        
        concat_dataset, loader = get_concat_dataset_and_loader(
            datasets=CL_datasets, 
            weights=args.weights,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_train=True,
            balanced_sampling=args.balanced_sampling,
            experience_replay=args.experience_replay,
            double_batch_size=args.double_batch_size,
        )
        concat_datasets.append(concat_dataset)
        loaders.append(loader)
    
    args.steps_per_epoch = min([len(loader) for loader in loaders])
    
    ### Prepare token mapping for the next-task regularization
    if args.regularize:
        logger.info(f'Preparing token mapping for regularization...')
        start_time = time.time()
        token_mapping = prepare_token_mapping(student_model.tokenizer, concat_datasets, args.exclude_special_tokens)
        student_model.call_module_function('set_token_mapping', token_mapping=token_mapping)
        logger.info(f'\ttime: {time.time() - start_time} seconds')
    
        # used in CLFramework._get_optimizer
        setattr(student_model, 'regularize', True)
        setattr(student_model, 'regularize_type', args.regularize_type)
        setattr(student_model, 'grad_scale_type', args.grad_scale_type)
        setattr(student_model, 'weight_decay_scale_type', args.weight_decay_scale_type)
        setattr(student_model, 'exclude_special_tokens', args.exclude_special_tokens)

    ### Define losses and objectives
    loss = get_loss_class(args=args)(
        model=student_model,
        crosslingual_loss_scale=args.crosslingual_loss_scale,
        crosslingual_loss_type=args.crosslingual_loss_type,
        crossmodal_loss_scale=args.crossmodal_loss_scale,
        crossmodal_loss_type=args.crossmodal_loss_type,
        en_loss_scale=args.en_loss_scale,
        pull_constraint_coeff=args.pull_constraint_coeff,
        fisher_penalty_scale=args.fisher_penalty_scale,
        fisher_gamma=args.fisher_gamma,
    )

    train_objectives = [(loader, loss) for loader in loaders]

    ### Define evaluator
    if args.do_evaluation:
        logger.info('Do evaluation during training')
        logger.info(f'\tdataset: {args.val_dataset}')
        logger.info(f'\tmonitor: {args.monitor}')
        logger.info(f'\tmode: {args.mode}')
        logger.info(f'\tnum_frames: {args.num_frames} (only taking effect for a video dataset)')
        logger.info(f'\tvision_root: {configs.image_video_root[args.val_dataset]}')

        vision_root = configs.image_video_root[args.val_dataset]
        num_frames = args.num_frames
        mode = args.mode

        eval_loaders = []
        rpath2emb = None
        for lang in args.target_languages:
            ann_rpath = get_formatted_string(
                kwargs=vars(args), 
                key=f"{args.mode}_file", 
                assigned_kwargs={'dataset': args.val_dataset, 'lang': lang},
            )
            pickle_path = get_formatted_string(
                kwargs=vars(args), 
                key='pickle_path', 
                assigned_kwargs=dict(
                    dataset=args.val_dataset, 
                    clip_model_name=args.teacher_model_name.replace('/', '_'),
                    mode=args.mode
                ),
            )
            eval_data = CaptionDatasetForRetrieval(
                vision_root=configs.image_video_root[args.val_dataset],
                ann_rpath=ann_rpath,
                num_frames=args.num_frames,
                lang=lang,
                clip_model=teacher_model,
                pickle_path=pickle_path,
                rpath2emb=rpath2emb,
            )
            rpath2emb = eval_data.rpath2emb
            eval_loader = DataLoader(
                eval_data,
                batch_size=args.inference_batch_size,
                shuffle=False,
                collate_fn=eval_data.collate_fn,
            )
            eval_loaders.append(eval_loader)

        evaluator = RetrievalEvaluator(
            loader=eval_loaders,
            monitor=args.monitor,
            mode=args.mode,
            logger=logger,
            with_epoch=True,
        )
    else:
        evaluator = None

    ### Start training
    with open(os.path.join(output_path, 'opts.json'), 'w') as wf:
        json.dump(vars(args), wf, indent=4)

    if args.warmup_steps < 1:
        total_steps = int(args.steps_per_epoch * args.epochs)
        args.warmup_steps = int(total_steps * args.warmup_steps)
    else:
        args.warmup_steps = int(args.warmup_steps)
    assert args.warmup_steps >= 0

    # log necessary information
    logger.info(f'dynamic_vocab: {args.dynamic_vocab}')
    logger.info(f'oracle_vocab: {args.oracle_vocab}')
    logger.info(f'crosslingual_loss_scale: {args.crosslingual_loss_scale}')
    logger.info(f'crosslingual_loss_type: {args.crosslingual_loss_type}')
    logger.info(f'crossmodal_loss_scale: {args.crossmodal_loss_scale}')
    logger.info(f'crossmodal_loss_type: {args.crossmodal_loss_type}')
    logger.info(f'en_loss_scale: {args.en_loss_scale}')
    logger.info(f'learning rate: {args.lr}')
    logger.info(f'warmup steps: {args.warmup_steps}')
    logger.info(f'weight decay: {args.weight_decay}')
    logger.info(f'no decay: {args.no_decay}')
    logger.info(f'Use amp for speeding up training: {args.use_amp}')
    logger.info(f'Regularization: {args.regularize}')
    if args.regularize:
        logger.info(f'\tRegularize type: {args.regularize_type}')
        if args.regularize_type in ['grad_decay', 'grad']:
            logger.info(f'\tGrad scale type: {args.grad_scale_type}')
        if args.regularize_type in ['grad_decay', 'decay']:
            logger.info(f'\tWeight decay scale type: {args.weight_decay_scale_type}')
        logger.info(f'\tExclude special tokens: {args.exclude_special_tokens}')
    logger.info(f'Experience replay: {args.experience_replay}')
    if args.experience_replay:
        logger.info(f'\tDark experience replay: {args.dark_experience_replay}')
        logger.info(f'\tMemory size: {args.memory_size}')
    logger.info(f'With prompt: {args.with_prompt}')
    if args.with_prompt:
        logger.info(f'\tPrompt type: {args.prompt_type}')
        logger.info(f'\tPrompt config: {args.prompt_config}')
        logger.info(f'\tPull constraint coeff: {args.pull_constraint_coeff}')    

    student_model.fit(
        train_objectives=train_objectives,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_params= {'lr': args.lr, 'betas': args.betas, 'eps': args.eps},
        weight_decay=args.weight_decay,
        no_decay=args.no_decay,
        output_path=output_path,
        checkpoint_path=output_path,
        checkpoint_save_steps=None, # save checkpoints every epoch, rather than spcific number of steps
        log_every=args.log_every,
        use_amp=args.use_amp,
        scheduler=args.scheduler,
        seed=args.seed,
        checkpoint_save_total_limit=args.checkpoint_save_total_limit,
        writter=writter,
        steps_per_epoch=args.steps_per_epoch,
    )

    if args.experience_replay:
        logger.info(f'{memory_dataset.language_summary()}')
        memory_dataset.save_memory()
