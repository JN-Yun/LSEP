### Training ###

# GPUs configurations
n_gpu=4            
n_accumulation=1   
bz_size=256        

# Following Tab1 & Tab5 configurations, 
model="XL"             # 0) select model 

uncond_prob=0.9        # 1) Prob for unconditioning
enc_depth=8            # 2) Depth
crop_start=12          # 3) Crop int if used, else 16

weights_type="steps"   # 4) weighting schedule
weight_start=0.0225
weight_end=0.03
n_weights=10          

weight_schedule="$weights_type-$weight_start-to-$weight_end-$n_weights"

add_name=""
echo $model-uncond_$uncond_prob-enc$enc_depth-RC_$crop_start-lambda-$weight_schedule-$add_name

output_dir="OUTPUT/LSEP/exps"
data_dir="DATADIR/IMAGENET"

accelerate launch --num_processes $n_gpu train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --model="SiT-$model/2" \
  --batch-size=$bz_size \
  --gradient-accumulation-steps=$n_accumulation \
  --uncond-prob=$uncond_prob \
  --encoder-depth=$enc_depth \
  --crop-start=$crop_start \
  --weights-type=$weights_type \
  --weight-start=$weight_start \
  --weight-end=$weight_end \
  --n-weights=$n_weights \
  --exp-name="$model-uncond_$uncond_prob-enc$enc_depth-RC_$crop_start-lambda-$weight_schedule-$add_name" \
  --output-dir=$output_dir \
  --data-dir=$data_dir
 
### Generation ###
sample_dir="SAMPLEDIR/samples"
ckpt="0400000"
cfg_scale=1.0
cfg_guidance=1.0
torchrun --nnodes=1 --nproc_per_node=$n_gpu generate.py \
  --model SiT-$model/2 \
  --num-fid-samples 50000 \
  --ckpt ./exps/$model-uncond_$uncond_prob-enc$enc_depth-RC_$crop_start-lambda-$weight_schedule-$add_name/checkpoints/${ckpt}.pt \
  --path-type=linear \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=$cfg_scale \
  --guidance-high=$cfg_guidance \
  --name_add uncond_$uncond_prob-enc$enc_depth-RC_$crop_start-lambda-$weight_schedule-$add_name \
  --sample_dir $sample_dir

### Evaluation ###
cd evaluator
python evaluator.py ./evaluator/VIRTUAL_imagenet256_labeled.npz ${sample_dir}/SiT-$model-2-enc$enc_depth-0400000-size-256-vae-ema-cfg-1.0-seed-0-sde-uncond_$uncond_prob-enc$enc_depth-RC_$crop_start-lambda-$weight_schedule-$add_name.npz
