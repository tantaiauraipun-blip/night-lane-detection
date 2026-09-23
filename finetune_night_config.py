"""
finetune_night_config.py
Fine-tuning config for CLRNet, starting from the existing r18_culane
checkpoint (11.pth), on the NIGHT-ONLY, ENHANCED training list built by
prepare_finetune_and_heldout.py (list/train_gt.txt, ~3500 images, repointed
at /night_enhanced/...).

Same architecture as the original training config (unchanged) -- only the
optimizer/schedule/work_dirs are adjusted for a short fine-tune rather than
a from-scratch run:
  - epochs: 12 -> 3      (fine-tune, not full training)
  - lr:     6e-4 -> 6e-5 (1/10th -- avoid wrecking the pretrained weights)
  - total_iter recomputed for the smaller (night-only) dataset size

Run from ~/CLRNet/CLRNet in the clrnet_torch113 env:

    conda activate clrnet_torch113
    unset PYTORCH_CUDA_ALLOC_CONF
    cd ~/CLRNet/CLRNet
    rm -f cache/culane_train.pkl   # IMPORTANT -- see prepare_finetune_and_heldout.py
    python main.py ~/gen_test/finetune_night_config.py \
        --finetune_from work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/ckpt/11.pth \
        --gpus 1
"""

net = dict(type='Detector', )

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=False,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)

heads = dict(type='CLRHead',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=36)

iou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0

work_dirs = "work_dirs/clr/r18_culane_night_finetune"

neck = dict(type='FPN',
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=False)

test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

# ---- fine-tune schedule (differs from the original 12-epoch / 6e-4 run) ----
epochs = 3
batch_size = 4

optimizer = dict(type='AdamW', lr=6e-5)  # 1/10th of the original 6e-4
NIGHT_TRAIN_SIZE = 3500  # approx -- actual count printed by prepare_finetune_and_heldout.py
total_iter = (NIGHT_TRAIN_SIZE // batch_size) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 1
save_ep = 1

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

# NOTE: dataset_path resolves via the existing ./data/CULane symlink -> HGLane.
# split='train' -> CLRNet's CULane loader reads list/train_gt.txt (see
# clrnet/datasets/culane.py: LIST_FILE['train'] = 'list/train_gt.txt'), which
# prepare_finetune_and_heldout.py has already overwritten with the
# night-only, enhanced list. split='val' below is the ordinary (mixed,
# untouched) validation set -- only used for periodic --validate monitoring
# during training, NOT for the real before/after comparison (that's done
# separately afterward with night_eval_clrnet.py on the clean held-out set).
dataset_path = './data/CULane'
dataset_type = 'CULane'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='train',
    processes=train_process,
),
val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='val',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='val',
    processes=val_process,
))

workers = 4
log_interval = 20
num_classes = 4 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
