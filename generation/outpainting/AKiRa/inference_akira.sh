OUTPUT_PATH=tba

SVD_PATH=/svd/stable-video-diffusion-img2vid
# CAMERACTRL_CKPT=/users/xwang/Work/CameraCtrl/model/CameraCtrl_svd.ckpt
# CAMERACTRL_CKPT=model/ours/sigmoid_cctrl_delta/checkpoint-step-70000.ckpt
CKPT=tba

#assets/svd_prompts_my_test.json \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM inference_ours.py \
      --out_root ${OUTPUT_PATH} \
      --num_frames 14 \
      --ori_model_path ${SVD_PATH} \
      --pose_adaptor_ckpt ${CKPT} \
      --prompt_file assets/svd_prompts_my_zoom.json \
      --trajectory_file assets/pose_files/4a2d6753676df096_svd.txt \
      --model_config configs/train_cameractrl/svd_320_576_cameractrl_ours_cctrl.yaml \
      --n_procs 1 \
      --bokeh_position_x 0 \
      --bokeh_position_y 0 \
      --plucker_type cctrl \
      --preprocess \
      --clear_all_pose \
      --bokeh \
      --bokeh_K 50 \