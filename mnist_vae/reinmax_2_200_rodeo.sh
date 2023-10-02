apt-get update 
apt-get install bc

##################################################################
## categorical_2_latent_200_mnist_k3
################################################################## 
folder=categorical_2_latent_200_mnist_k3
echo $folder
mkdir -p $folder

train_error_sum=0
test_error_sum=0

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python gumbel_softmax_vae_rodeo.py --lr 3e-4 --temperature 1.1 --method reinmax --number-of-samples 3 --seed $SEED | tee ${folder}/reinmax_${SEED}.log
    train_error=`tail -n2 ${folder}/reinmax_${SEED}.log | head -n 1 | awk '{print $6}'`
    train_error_sum=$(echo ${train_error_sum} + ${train_error} | bc) 

    test_error=`tail -n1 ${folder}/reinmax_${SEED}.log | awk '{print $5}'`
    test_error_sum=$(echo ${test_error_sum} + ${test_error} | bc) 
done 

echo "AVG train error: " $(echo "${train_error_sum} / 10" | bc -l)
echo "AVG test error: " $(echo "${test_error_sum} / 10" | bc -l)


##################################################################
## categorical_2_latent_200_omni_k3
################################################################## 
folder=categorical_2_latent_200_omni_k3
echo $folder

if [ ! -f omni_chardata.mat ]
then
	echo "downloading OmniGlot data"
    wget -O omni_chardata.mat https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat
fi

mkdir -p $folder

train_error_sum=0
test_error_sum=0

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python gumbel_softmax_vae_rodeo.py --lr 5e-4 --temperature 1.2 --method reinmax --dataset omniglot --number-of-samples 3 --seed $SEED | tee ${folder}/reinmax_${SEED}.log
    train_error=`tail -n2 ${folder}/reinmax_${SEED}.log | head -n 1 | awk '{print $6}'`
    train_error_sum=$(echo ${train_error_sum} + ${train_error} | bc) 

    test_error=`tail -n1 ${folder}/reinmax_${SEED}.log | awk '{print $5}'`
    test_error_sum=$(echo ${test_error_sum} + ${test_error} | bc) 
done 

echo "AVG train error: " $(echo "${train_error_sum} / 10" | bc -l)
echo "AVG test error: " $(echo "${test_error_sum} / 10" | bc -l)


##################################################################
## categorical_2_latent_200_fashion_k3
################################################################## 
folder=categorical_2_latent_200_fashion_k3
echo $folder

mkdir -p $folder

train_error_sum=0
test_error_sum=0

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python gumbel_softmax_vae_rodeo.py --lr 3e-4 --temperature 1.1 --method reinmax --dataset FashionMNIST --number-of-samples 3 --seed $SEED | tee ${folder}/reinmax_${SEED}.log
    train_error=`tail -n2 ${folder}/reinmax_${SEED}.log | head -n 1 | awk '{print $6}'`
    train_error_sum=$(echo ${train_error_sum} + ${train_error} | bc) 

    test_error=`tail -n1 ${folder}/reinmax_${SEED}.log | awk '{print $5}'`
    test_error_sum=$(echo ${test_error_sum} + ${test_error} | bc) 
done 

echo "AVG train error: " $(echo "${train_error_sum} / 10" | bc -l)
echo "AVG test error: " $(echo "${test_error_sum} / 10" | bc -l)


##################################################################
## categorical_2_latent_200_mnist_k2
################################################################## 
folder=categorical_2_latent_200_mnist_k2
echo $folder
mkdir -p $folder

train_error_sum=0
test_error_sum=0

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python gumbel_softmax_vae_rodeo.py --lr 3e-4 --temperature 1.1 --method reinmax --number-of-samples 2 --seed $SEED | tee ${folder}/reinmax_${SEED}.log
    train_error=`tail -n2 ${folder}/reinmax_${SEED}.log | head -n 1 | awk '{print $6}'`
    train_error_sum=$(echo ${train_error_sum} + ${train_error} | bc) 

    test_error=`tail -n1 ${folder}/reinmax_${SEED}.log | awk '{print $5}'`
    test_error_sum=$(echo ${test_error_sum} + ${test_error} | bc) 
done 

echo "AVG train error: " $(echo "${train_error_sum} / 10" | bc -l)
echo "AVG test error: " $(echo "${test_error_sum} / 10" | bc -l)


##################################################################
## categorical_2_latent_200_omni_k2
################################################################## 
folder=categorical_2_latent_200_omni_k2
echo $folder

mkdir -p $folder

if [ ! -f omni_chardata.mat ]
then
	echo "downloading OmniGlot data"
    wget -O omni_chardata.mat https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat
fi

train_error_sum=0
test_error_sum=0

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python gumbel_softmax_vae_rodeo.py --lr 3e-4 --temperature 1.1 --method reinmax --dataset omniglot --number-of-samples 2 --seed $SEED | tee ${folder}/reinmax_${SEED}.log
    train_error=`tail -n2 ${folder}/reinmax_${SEED}.log | head -n 1 | awk '{print $6}'`
    train_error_sum=$(echo ${train_error_sum} + ${train_error} | bc) 

    test_error=`tail -n1 ${folder}/reinmax_${SEED}.log | awk '{print $5}'`
    test_error_sum=$(echo ${test_error_sum} + ${test_error} | bc) 
done 

echo "AVG train error: " $(echo "${train_error_sum} / 10" | bc -l)
echo "AVG test error: " $(echo "${test_error_sum} / 10" | bc -l)


##################################################################
## categorical_2_latent_200_fashion_k2
################################################################## 
folder=categorical_2_latent_200_fashion_k2
echo $folder

mkdir -p $folder

train_error_sum=0
test_error_sum=0

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python gumbel_softmax_vae_rodeo.py --lr 3e-4 --temperature 1.1 --method reinmax --dataset FashionMNIST --number-of-samples 2 --seed $SEED | tee ${folder}/reinmax_${SEED}.log
    train_error=`tail -n2 ${folder}/reinmax_${SEED}.log | head -n 1 | awk '{print $6}'`
    train_error_sum=$(echo ${train_error_sum} + ${train_error} | bc) 

    test_error=`tail -n1 ${folder}/reinmax_${SEED}.log | awk '{print $5}'`
    test_error_sum=$(echo ${test_error_sum} + ${test_error} | bc) 
done 

echo "AVG train error: " $(echo "${train_error_sum} / 10" | bc -l)
echo "AVG test error: " $(echo "${test_error_sum} / 10" | bc -l)
