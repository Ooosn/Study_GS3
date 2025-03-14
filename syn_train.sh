date="241203"
subtask="syn_scenes"
data_root="/path/to/data"
view_num=2000
                
python train.py -s $data_root/Synthetic/Hotdog \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Hotdog" \
                --use_nerual_phasefunc \
                --eval 

python train.py -s $data_root/Synthetic/Lego \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 28000 \
                --spcular_freeze_step 18000 \
                --fit_linear_step 8000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Lego" \
                --use_nerual_phasefunc \
                --eval 
                
python train.py -s $data_root/Synthetic/FurBall \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/FurBall" \
                --use_nerual_phasefunc \
                --eval 
                
python train.py -s $data_root/Synthetic/AnisoMetal \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/AnisoMetal" \
                --use_nerual_phasefunc \
                --eval 
                
python train.py -s $data_root/Synthetic/Drums \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 30000 \
                --asg_lr_max_steps 70000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0008 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                --densify_grad_threshold 0.00013 \
                -m "./output/$date/$subtask/Drums" \
                --use_nerual_phasefunc \
                --eval 
                
python train.py -s $data_root/Synthetic/Translucent \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Translucent" \
                --use_nerual_phasefunc \
                --eval 
                
python train.py -s $data_root/RenderCapture/Tower \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Tower" \
                --use_nerual_phasefunc \
                --eval 
                
python train.py -s $data_root/RenderCapture/MaterialBalls \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/MaterialBalls" \
                --use_nerual_phasefunc \
                --eval 

python train.py -s $data_root/RenderCapture/Egg \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Egg" \
                --use_nerual_phasefunc \
                --eval 
                
python train.py -s $data_root/RenderCapture/Fabric \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Fabric" \
                --use_nerual_phasefunc \
                --eval 
                
python train.py -s $data_root/RenderCapture/Cup \
                --hdr \
                --white_background \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Cup" \
                --use_nerual_phasefunc \
                --eval 
