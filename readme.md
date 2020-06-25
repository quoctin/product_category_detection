## Product detection
This code repo tackle the product detect problem. Given an image, the trained model needs to predict the category it belongs to.

## Hyperparameter adjustment
Most of hyperparameters are configured inside `main.py`, the main script to train/evaluate model.

- Number of epochs -> `FLAGS.epoch`
- Batch size -> `FLAGS.batch_size`
- Working dir -> `FLAGS.working_dir`
- Image input size -> `FLAGS.im_size`
- Training data path -> `FLAGS.training_data`
- Test data path -> `FLAGS.test_data`
- Learning rate -> `learning_rate`, default to 1e-4
- Starting learning decay iteration -> `lr_start_decay`, default to the 30-th epoch
- Learning rate decay frequency -> `lr_decay_every`, default to every 10 epochs
- Data augmentation mode -> `FLAGS.augmentation`, default to True
- TF records -> `FLAGS.use_tfrecord`


## Run training and monitoring

Execute:

```
python main.py
```

To monitor the training process:

```
tensorboard --logdir=model/summaries --port 5000
```

and nagigate to `localhost:5000` on the web browser.

