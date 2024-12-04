# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel)

from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import LLaVAModel
from xtuner.model.modeling_llama import CustomLlamaForCausalLM

from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# llm_name_or_path = '/data/kesun/LLaVA-v1.5-7b-1'
llm_name_or_path = '/data/kesun/vicuna-7b-v1.5'
# visual_encoder_name_or_path = 'openai/clip-vit-large-patch14-336'
visual_encoder_name_or_path = '/data/kesun/zzy_weights/aigcllmdetectvisual/image_visual'
# visual_encoder_name_or_path = '/home/kesun/zzy/xtuner/work_dirs/llava_v15_7b_finetune_all/iter_6550_xtuner/visual_encoder'
# visual_encoder_name_or_path = '/home/kesun/zzy/xtuner/work_dirs/llava_v15_7b_finetune_all/iter_6550_xtuner/visual_encoder/pytorch_model.bin'
# visual_encoder_name_or_path = '/home/kesun/zzy/xtuner/work_dirs/llava_v15_7b_finetune_all'
pretrained_pth = '/home/kesun/zzy/xtuner/epoch_1.pth'
# pretrained_pth = '/data/kesun/work_dirs/llava_v15_7b_finetune_AIGC_4_cls_pretrain_1029/iter_500.pth'
# pretrained_pth = '/data/kesun/work_dirs/ll  ava_v15_7b_finetune_AIGC_4_cls_sd_1030/iter_500.pth'

# Data
# "_name_or_path": "openai/clip-vit-large-patch14-336",
# data_root = '/home/kesun/zzy/LLaVA/playground/data/'
# data_path = data_root + 'LLaVA-Pretrain/ffpp0608_wotest.json'
# image_folder = "/data/kesun/kesun/Dataset/Dataset/Deepfake/ffpp"
data_root = "/home/kesun/kesun/kesun/FFAA/data1/"
# data_path = data_root + 'LLaVA-Pretrain/ffpp0608_wotest.json'
# data_path = data_root + 'aigcdetect_progan_all_4cls.json'
data_path = data_root + 'aigcdetect_progan_id.json'
# data_path = data_root + 'aigcdetect_progan_debug.json'
# image_folder = "/data/kesun/kesun/aigcdatasets/progan_train/"
image_folder = "/data/kesun/kesun/"
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = int(4096 - (336 / 14)**2)

# Scheduler & Optimizer
batch_size = 32  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 5
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 22  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''
evaluation_images = ['/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/ADM/0_real/ILSVRC2012_val_00019159.JPEG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/ADM/0_real/ILSVRC2012_val_00041074.JPEG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/ADM/0_real/ILSVRC2012_val_00009110.JPEG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/ADM/0_real/ILSVRC2012_val_00041569.JPEG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/ADM/0_real/ILSVRC2012_val_00020178.JPEG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/ADM/1_fake/624_adm_34.PNG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/ADM/1_fake/188_adm_7.PNG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/ADM/1_fake/440_adm_91.PNG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/ADM/1_fake/988_adm_34.PNG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/ADM/1_fake/265_adm_7.PNG'] + [
"/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/progan/horse/1_fake/08995.png",
"/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/progan/horse/1_fake/00728.png",
"/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/progan/horse/1_fake/02135.png",
"/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/progan/horse/1_fake/10641.png",
"/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/progan/horse/1_fake/19449.png",
"/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/progan/horse/0_real/17566.png",
"/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/progan/horse/0_real/07552.png",
"/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/progan/horse/0_real/00253.png",
"/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/progan/horse/0_real/02938.png",
"/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/progan/horse/0_real/15994.png",
] + ['/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/stable_diffusion_v_1_4/1_fake/885_sdv4_00143.png',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/stable_diffusion_v_1_4/1_fake/282_sdv4_00074.png',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/stable_diffusion_v_1_4/1_fake/592_sdv4_00020.png',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/stable_diffusion_v_1_4/1_fake/035_sdv4_00020.png',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/stable_diffusion_v_1_4/1_fake/792_sdv4_00020.png',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/stable_diffusion_v_1_4/0_real/ILSVRC2012_val_00041159.JPEG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/stable_diffusion_v_1_4/0_real/ILSVRC2012_val_00033883.JPEG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/stable_diffusion_v_1_4/0_real/ILSVRC2012_val_00010206.JPEG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/stable_diffusion_v_1_4/0_real/ILSVRC2012_val_00032401.JPEG',
 '/data/kesun/kesun/aigcdatasets/AIGCDetect_testset/stable_diffusion_v_1_4/0_real/ILSVRC2012_val_00033198.JPEG']
evaluation_inputs = [
    "Can you identify if this image was created by AI or if it is real?",
    "Please assess whether this picture is a product of artificial intelligence or an actual creation.",
    "Determine if this image is generated by AI or if it is genuine.",
    "Could you tell if this image is AI-generated or real?",
    "Please evaluate whether this image is an AI creation or something real.",
    "Can you discern if this picture is made by AI or if it is real?",
    "Please analyze whether this image is produced by AI or if it is real.",
    "Determine whether this image is a result of AI generation or something real.",
    "Can you figure out if this image is AI-generated or real?",
    "Please judge if this image is created by AI or if it is real."
] + [
    "Can you identify if this image was created by AI or if it is real?",
    "Please assess whether this picture is a product of artificial intelligence or an actual creation.",
    "Determine if this image is generated by AI or if it is genuine.",
    "Could you tell if this image is AI-generated or real?",
    "Please evaluate whether this image is an AI creation or something real.",
    "Can you discern if this picture is made by AI or if it is real?",
    "Please analyze whether this image is produced by AI or if it is real.",
    "Determine whether this image is a result of AI generation or something real.",
    "Can you figure out if this image is AI-generated or real?",
    "Please judge if this image is created by AI or if it is real."
] + [
    "Can you identify if this image was created by AI or if it is real?",
    "Please assess whether this picture is a product of artificial intelligence or an actual creation.",
    "Determine if this image is generated by AI or if it is genuine.",
    "Could you tell if this image is AI-generated or real?",
    "Please evaluate whether this image is an AI creation or something real.",
    "Can you discern if this picture is made by AI or if it is real?",
    "Please analyze whether this image is produced by AI or if it is real.",
    "Determine whether this image is a result of AI generation or something real.",
    "Can you figure out if this image is AI-generated or real?",
    "Please judge if this image is created by AI or if it is real."
]

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

model = dict(
    type=LLaVAModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    pretrained_pth=pretrained_pth,
    llm=dict(
        # type=AutoModelForCausalLM.from_pretrained,
        type=CustomLlamaForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True),
    llm_lora=dict(
        type=LoraConfig,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path),
    # visual_encoder_lora = dict(
    #     type=LoraConfig, r=64, lora_alpha=128, lora_dropout=0.05, bias='none')
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=llava_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=default_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        image_processor=image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
