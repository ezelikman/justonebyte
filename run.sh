IP=$(ip route get 1.2.3.4 | awk '{print $7}') 
START_IP=$(cat start_ip.txt) 



for model_name in llama-7b-hf
do
    echo $model_name
    
    echo "FF"
    CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 app.py --start_ip $START_IP --self_ip=$IP --model_name $model_name > "${model_name}_FF_${IP}" 2>&1
    sleep 60

    echo "TF"
    CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 app.py --start_ip $START_IP --self_ip=$IP --send_full_grad True --model_name $model_name  > "${model_name}_TF_${IP}" 2>&1
    sleep 60

    echo "FT"
    CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 app.py --start_ip $START_IP --self_ip=$IP --normal True --learning_rate 1e-4 --model_name $model_name  > "${model_name}_FT_${IP}" 2>&1
    sleep 60
done
