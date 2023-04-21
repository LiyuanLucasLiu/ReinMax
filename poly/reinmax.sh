p=${1:-2}

apt-get update 
apt-get install bc

folder=log_p_${p}
echo $folder 
mkdir -p $folder 

loss_sum=0
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python poly_programming.py --method reinmax --pnorm $p --seed ${SEED} | tee ${folder}/reinmax_${SEED}.log
    
    loss=`tail -n1 ${folder}/reinmax_${SEED}.log | awk '{print $6}' | xargs`
    loss_sum=$(echo ${loss_sum} + ${loss} | bc)
done

echo "AVG LOSS: " $(echo "${loss_sum} / 10" | bc -l)