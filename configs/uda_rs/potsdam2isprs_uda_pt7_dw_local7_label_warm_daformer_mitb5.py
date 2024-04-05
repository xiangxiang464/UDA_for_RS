_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture with weighted loss
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # Potsdam->Vaihingen Data Loading
    '../_base_/datasets/uda_potsdam_to_vaihingen.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    # '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Model Configuration
model = dict(
    decode_head=dict(
        num_classes=6,
    )
)
# Modifications to batch_size
data = dict(samples_per_gpu=4)
# Modifications to Basic UDA
uda = dict(
    alpha=0.9, #指数移动平均线（EMA）的超参数
    pseudo_threshold=0.7, # 伪标签阈值，预测的结果大于阈值就判断是正确的伪标签
    dynamic_class_weight=True, # 设置是否需要使用渐近类权重
    pseudo_kernal_size=7, # 局部动态质量，观察局部的深度
    local_ps_weight_type='label',
)
# Optimizer Hyperparameters
# lr_config = dict(
#     warmup_iters=1500,
#     )
lr_config = dict(
    policy='CosineAnnealing',  # 指定使用余弦退火学习率调度器
    by_epoch=False,  # 根据你的训练是按照epoch还是iteration，选择True或False
    warmup='linear',  # 线性预热
    warmup_iters=1500,  # 预热迭代次数
    warmup_ratio=0.1,  # 预热期间起始学习率为最大学习率的比例
    min_lr=0  # 训练结束时的最小学习率
)
optimizer_config = None
optimizer = dict(
    lr=1e-04,
    paramwise_cfg=dict( # seems useless
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
# GPU Configuration
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=4000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=4)
evaluation = dict(interval=400, metric='mIoU')
# Meta Information for Result Analysis
name = 'potsdam2isprs_uda_pt7_local7_label_warm_daformer_mitb5'
exp = 'basic'
name_dataset = 'potsdam2vaihingen'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a0.9_pt0.9_mix0.1'
name_opt = 'adamw_1e-04_poly10warm_1x2_4k'