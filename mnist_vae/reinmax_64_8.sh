apt-get update 
apt-get install bc

folder=categorical_64_latent_8

mkdir -p $folder

train_error_sum=0
test_error_sum=0

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python gumbel_softmax_vae.py --lr 5e-4 --temperature 1.5 --method reinmax --optim radam --categorical-dim 64 --latent-dim 8 --seed $SEED | tee ${folder}/reinmax_${SEED}.log
    train_error=`tail -n2 ${folder}/reinmax_${SEED}.log | head -n 1 | awk '{print $6}'`
    train_error_sum=$(echo ${train_error_sum} + ${train_error} | bc) 

    test_error=`tail -n1 ${folder}/reinmax_${SEED}.log | awk '{print $5}'`
    test_error_sum=$(echo ${test_error_sum} + ${test_error} | bc) 
done 

echo "AVG train error: " $(echo "${train_error_sum} / 10" | bc -l)
echo "AVG test error: " $(echo "${test_error_sum} / 10" | bc -l)