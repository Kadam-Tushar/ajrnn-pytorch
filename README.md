# AJ-RNN

Sample commands for training:

```bash
python3 ajrnn.py --batch_size 20 --epoch 100 --lamda_D 0.7 --G_epoch 5 --train_data_filename dataset/CBF/CBF_TRAIN --test_data_filename dataset/CBF/CBF_TEST --hidden_size 30
```

```bash
python3 ajrnn.py --batch_size 50 --epoch 150 --lamda_D 0.7 --G_epoch 5 --train_data_filename dataset/CBF/CBF_TEST --test_data_filename dataset/CBF/CBF_TEST --hidden_size 50 --layer_num 1 --missing_frac 0.5 --learning_rate 0.05
```
