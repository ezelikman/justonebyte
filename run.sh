IP=$(ip route get 1.2.3.4 | awk '{print $7}') 
START_IP=$(cat start_ip.txt) 


for lr in 1e-4 1e-3 1e-2 1e-1 1 1e-5
do 
    for model_name in gpt2
    do
        echo $model_name

        echo "FT"
        CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 app.py --start_ip $START_IP --self_ip=$IP --normal True --learning_rate $lr --model_name $model_name --gradient_acc_steps 8 > "${model_name}_FT_${IP}" 2>&1
        sleep 60 
        
        echo "FF"
        CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 app.py --start_ip $START_IP --self_ip=$IP --learning_rate $lr --model_name $model_name --gradient_acc_steps 8 > "${model_name}_FF_${IP}" 2>&1
        sleep 60

        echo "TF"
        CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 app.py --start_ip $START_IP --self_ip=$IP --send_full_grad True --learning_rate $lr --model_name $model_name --gradient_acc_steps 8  > "${model_name}_TF_${IP}" 2>&1
        sleep 60

    done


    for model_name in llama-7b-hf
    do
        echo $model_name

        echo "FT"
        CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u app.py --start_ip $START_IP --self_ip=$IP --normal True --learning_rate $lr --model_name $model_name  --max_iterations 100 --batch_size 8 --gradient_acc_steps 16 > "${model_name}_FT_${IP}" 2>&1
        sleep 60

        echo "FF"
        CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u app.py --start_ip $START_IP --self_ip=$IP --learning_rate $lr --model_name $model_name --max_iterations 100 --batch_size 8  --gradient_acc_steps 16  > "${model_name}_FF_${IP}" 2>&1
        sleep 60

        echo "TF"
        CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u app.py --start_ip $START_IP --self_ip=$IP --send_full_grad True --learning_rate $lr --model_name $model_name  --max_iterations 100  --batch_size 8  --gradient_acc_steps 16  > "${model_name}_TF_${IP}" 2>&1
        sleep 60
        
    done

done
