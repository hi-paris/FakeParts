# export CUDA_VISIBLE_DEVICES=0,1,2,3

export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29579 \
    main.py \
    --dataset_path data_example/DM_testdata/ \
    --test_selected_subsets 'dalle' \
    --eval \
    --pretrained_model pretrained/fatformer_4class_ckpt.pth \
    --num_vit_adapter 3 \
    --num_context_embedding 8