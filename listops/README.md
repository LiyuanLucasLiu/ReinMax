# ListOps
The implementation of MNIST VAE is based on:
- https://github.com/jihunchoi/unsupervised-treelstm
- https://github.com/chijames/GST/tree/master

We provided models trained with `reinmax` in the `reinmax_cps` folder, to reproduce the reported performance, run:
```
bash reinmax.sh
```

Also, we would like to note that, the implementation here is not very stable. To reproduce the training of these models, the easiest way is to:
1. use docker image `base/job/pytorch/1.11.0-cuda11.6:20220404T154700786` from `singularitybase.azurecr.io`
2. run the following command on a P40 GPU
```
python -m nlp.train --word-dim 300 --hidden-dim 300 --clf-hidden-dim 300 \
    --clf-num-layers 1 --batch-size 16 --max-epoch 20 --save-dir ${CPS} \
    --device cuda --pretrained glove.840B.300d --leaf-rnn --dropout 0.5 \
    --lower --task listops --lr 0.0007 --temperature 1.1 --method reinmax \
    --optimizer radam --seed ${SEED} | tee ${CPS}/reinmax_${SEED}.log
```