
device=1

LOG=${save_dir}"res.log"
echo ${LOG}

# ------------------------------------Industrial DATASET------------------------------------#

# Train on VisA dataset and test on MVTec dataset
n_ctx=(13)   # 13 12
ab_ctx=(13)  # 13 12
depth=(9)    # 9
t_n_ctx=(4)  # 4
alpha=(1.0 0.0)

for i in "${!depth[@]}"; do
    for j in "${!n_ctx[@]}"; do
        for k in "${!ab_ctx[@]}"; do
            for h in "${!alpha[@]}"; do

                base_dir=${n_ctx[j]}_${ab_ctx[k]}_${depth[i]}_${t_n_ctx[0]}_mvtec
                save_dir=./checkpoints/glocalclip/${base_dir}/alpha_${alpha[h]}/
                
                CUDA_VISIBLE_DEVICES=${device} python train.py --dataset visa --train_data_path ./data/visa \
                --save_path ${save_dir} \
                --features_list 6 12 18 24 --image_size 518 --batch_size 8 --print_freq 1 \
                --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --ab_ctx ${ab_ctx[k]} --alpha ${alpha[h]}
                wait

                result_dir=./results/glocalclip/${base_dir}/alpha_${alpha[h]}/reverse/
                CUDA_VISIBLE_DEVICES=${device} python test.py --dataset mvtec --data_path ./data/mvtec \
                --save_path ${result_dir} --checkpoint_path ${save_dir}epoch_15.pth \
                --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --ab_ctx ${ab_ctx[k]} --sigma 8
                wait
            done
        done
    done
done

# Train on MVTec dataset and test on other datasets
n_ctx=(12)   # 13 12
ab_ctx=(13)  # 10 12
t_n_ctx=(2)  # 4 4
depth=(10)    # 12 9
alpha=(1.0 0.0 0.1 0.01 )
dataset=('visa' 'mpdd' 'btad' 'SDD' 'DTD')

for i in "${!depth[@]}"; do
    for j in "${!n_ctx[@]}"; do
        for k in "${!ab_ctx[@]}"; do
            for h in "${!alpha[@]}"; do

                base_dir=${n_ctx[j]}_${ab_ctx[k]}_${depth[i]}_${t_n_ctx[0]}_indus_others
                save_dir=./checkpoints/glocalclip/${base_dir}/alpha_${alpha[h]}/
                
                CUDA_VISIBLE_DEVICES=${device} python train.py --dataset mvtec --train_data_path ./data/mvtec \
                --save_path ${save_dir} \
                --features_list 6 12 18 24 --image_size 518 --batch_size 8 --print_freq 1 \
                --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --ab_ctx ${ab_ctx[k]} --alpha ${alpha[h]}
                wait

                for z in "${!dataset[@]}"; do

                    result_dir=./results/glocalclip/${base_dir}/alpha_${alpha[h]}/${dataset[z]}/reverse/
                    CUDA_VISIBLE_DEVICES=${device} python test.py --dataset ${dataset[z]} --data_path ./data/${dataset[z]} \
                    --save_path ${result_dir} --checkpoint_path ${save_dir}epoch_15.pth \
                    --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --ab_ctx ${ab_ctx[k]} --sigma 8
                    wait
                done
            done
        done
    done
done

#------------------------------------MEDICAL DATASET------------------------------------#

dataset_image=('headct' 'brain_mri' 'br35h' 'covid19')
dataset_pixel=('ISIC' 'clinicdb' 'colondb' 'kvasir' 'endo' 'tn3k')

# Train on colondb dataset and test on clinicdb dataset
n_ctx=(12)   # 13
ab_ctx=(12)  # 10
depth=(12)   # 12
t_n_ctx=(4)  # 2
alpha=(1.0 0.0)

for i in "${!depth[@]}"; do
    for j in "${!n_ctx[@]}"; do
        for k in "${!ab_ctx[@]}"; do
            for h in "${!alpha[@]}"; do
                base_dir=${n_ctx[j]}_${ab_ctx[k]}_${depth[i]}_${t_n_ctx[0]}_clinicdb
                save_dir=./checkpoints/glocalclip/${base_dir}/alpha_${alpha[h]}/

                CUDA_VISIBLE_DEVICES=${device} python train.py --dataset colondb --train_data_path ./data/colondb \
                --save_path ${save_dir} \
                --features_list 6 12 18 24 --image_size 518 --batch_size 8 --print_freq 1 \
                --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --ab_ctx ${ab_ctx[k]} --alpha ${alpha[h]}
                wait

                result_dir=./results/glocalclip/${base_dir}/alpha_${alpha[h]}/reverse/

                CUDA_VISIBLE_DEVICES=${device} python test.py --dataset clinicdb --data_path ./data/clinicdb --metrics pixel-level \
                --save_path ${result_dir} --checkpoint_path ${save_dir}epoch_15.pth \
                --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --ab_ctx ${ab_ctx[k]} --sigma 8 
                wait
            done
        done
    done
done


# Train on clinicdb dataset and test on other datasets
n_ctx=(12)   # 13
ab_ctx=(12)  # 10
depth=(9)    # 12
t_n_ctx=(4)  # 4
alpha=(1.0 0.0 0.1 0.01)

dataset=('headct' 'brain_mri' 'br35h' 'ISIC' 'colondb' 'kvasir' 'endo' 'tn3k')
for i in "${!depth[@]}"; do
    for j in "${!n_ctx[@]}"; do
        for k in "${!ab_ctx[@]}"; do
            for h in "${!alpha[@]}"; do 
                base_dir=${n_ctx[j]}_${ab_ctx[k]}_${depth[i]}_${t_n_ctx[0]}_medical_others
                save_dir=./checkpoints/glocalclip/${base_dir}/alpha_${alpha[h]}/
                
                CUDA_VISIBLE_DEVICES=${device} python train.py --dataset clinicdb --train_data_path ./data/clinicdb \
                --save_path ${save_dir} \
                --features_list 6 12 18 24 --image_size 518 --batch_size 8 --print_freq 1 \
                --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --ab_ctx ${ab_ctx[k]} --alpha ${alpha[h]}
                wait

                for z in "${!dataset[@]}"; do

                    if echo "${dataset_image[@]}" | grep -qw "${dataset[z]}"; then
                        
                        result_dir=./results/glocalclip/${base_dir}/alpha_${alpha[h]}/${dataset[z]}/reverse/

                        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset ${dataset[z]} --data_path ./data/${dataset[z]} --metrics image-level \
                        --save_path ${result_dir} --checkpoint_path ${save_dir}epoch_15.pth \
                        --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --ab_ctx ${ab_ctx[k]} --sigma 8
                        wait
                    fi

                    if echo "${dataset_pixel[@]}" | grep -qw "${dataset[z]}"; then
                        
                        result_dir=./results/glocalclip/${base_dir}/alpha_${alpha[h]}/${dataset[z]}/reverse/

                        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset ${dataset[z]} --data_path ./data/${dataset[z]} --metrics pixel-level \
                        --save_path ${result_dir} --checkpoint_path ${save_dir}epoch_15.pth \
                        --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --ab_ctx ${ab_ctx[k]} --sigma 8
                        wait
                    fi
                done
            done
        done
    done
done