export CUDA_VISIBLE_DEVICES=4,5,6,7

export WANDB_MODE=offline
export WANDB_DIR=/lpai/output/models/wandb_log

torchrun --nproc_per_node=4 --nnodes=1 main_wds.py \
  --batch-size 256 \
  --train-data "/lpai/dataset/yfcc15m/0-1-6/yfcc15m_webdataset/0{0000..1410}.tar" \
  --train-num-samples 14082031 \
  --imagenet-val '/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val' \
  --model DetailCLIP_VITB16 \
  --workers 10 --wandb \
  --lr 5e-4 --wd 0.5 \
  --mask-ratio 0.5 \
  --epochs 30 \
  --clip_loss_weight 1 --ibot_patch_loss_weight 1 \
  --ibot_cls_loss_weight 1 --reconst_loss_weight 1 --print-freq 1000 \
  --output-dir /lpai/output/models/detailclip_vitb16_yfcc15m_30ep