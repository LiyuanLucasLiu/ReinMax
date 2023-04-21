# MNIST VAE
The implementation of MNIST VAE is based on:
- https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
- https://github.com/chijames/GST/tree/master


We summarized the searched hyper-parameters for `reinmax` in this repo. Specifically, for MNIST VAE with C categorical dimensions and L latent dimensions, run:
```
bash reinmax_${C}_${L}.sh
```

For example, for MNIST VAE with 8 categorical dimensions and 4 latent dimensions, run:
```
bash reinmax_8_4.sh
```

To estimate the bias of the estimated gradient, run:
```
python gumbel_softmax_vae.py --lr 5e-4 --temperature 1.3 --method reinmax --optim adam --categorical-dim 8 --latent-dim 4 -s ${SAMPLE_NUMBER} --seed $SEED
```
For example, to estimate the bias on one sample while setting seed to 0, run: 
```
python gumbel_softmax_vae.py --lr 5e-4 --temperature 1.3 --method reinmax --optim adam --categorical-dim 8 --latent-dim 4 -s 1 --seed 0
```