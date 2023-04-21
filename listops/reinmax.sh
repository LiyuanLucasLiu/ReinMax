apt-get update 
apt-get install bc
pip install -r requirements.txt

folder=${1:-"reinmax_cps"}
mkdir -p tmp

valid_loss_sum=0
valid_acc_sum=0
for SEED in 0 1 2 3 4
do
    echo "Evaluating ${folder}/checkpoint_${SEED}.pt"
    python -m nlp.evaluate --word-dim 300 --hidden-dim 300 --clf-hidden-dim 300 \
                        --clf-num-layers 1 --device cuda --leaf-rnn --dropout 0.5 --lower \
                        --method reinmax --task listops --temperature 1.1 --model ${folder}/checkpoint_${SEED}.pt | tee tmp/reinmax_${SEED}.log
    
    valid_loss=`tail -n2 tmp/reinmax_${SEED}.log | awk '{print $2}' | xargs`
    valid_loss=${valid_loss:0:6}
    valid_loss_sum=$(echo ${valid_loss_sum} + ${valid_loss} | bc)


    valid_acc=`tail -n1 tmp/reinmax_${SEED}.log | awk '{print $2}' | xargs`
    valid_acc=${valid_acc:0:6}
    valid_acc_sum=$(echo ${valid_acc_sum} + ${valid_acc} | bc)
done

valid_loss_sum=$(echo "${valid_loss_sum} / 5" | bc -l)
valid_acc_sum=$(echo "${valid_acc_sum} / 5" | bc -l)
echo "Valid loss avg: ${valid_loss_sum}" 
echo "Valid acc avg: ${valid_acc_sum}"
