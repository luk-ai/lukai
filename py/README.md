# Luk.ai Python Management Library

This is a library for uploading machine learning models to
[Luk.ai](https://luk.ai).

## Upload Models

You'll need to [create an API token](https://luk.ai/dashboard) first.

```python
import lukai

# ... your model definition code

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Sets the Luk.ai API token.
lukai.set_api_token('<your token>')

# Uploads the model to Luk.ai and creates a training job.
lukai.upload(
    session=sess,
    domain='<your domain>',
    model_type='<your model type>',
    name=FLAGS.name,
    description=FLAGS.description,
    hyper_params=lukai.HyperParams(
        proportion_clients = 0.1,
        batch_size = 10,
        num_rounds = 100,
        learning_rate = learning_rate,
        num_local_rounds = 10,
    ),
    metrics={
      accuracy: lukai.REDUCE_MEAN,
    },
    event_targets={
      lukai.EVENT_TRAIN: (keep_prob.assign(0.5),),
      lukai.EVENT_INFER: (keep_prob.assign(1.0),),
      lukai.EVENT_EVAL: (keep_prob.assign(1.0),),
    },
)
```

See the [full mnist example](https://github.com/luk-ai/docs/tree/master/examples/mnist).

## Export Models

You can also directly output the `model.tar.gz` file if you'd like.

```python
from lukai import saver

# ... your model definition code

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print('Node names: x = {}, y_ = {}, train_step = {}, w = {}, b = {}, y = {}'.format(
  x.name, y_.name, train_step.name, w.name, b.name, y.name,
))

saver.save(sess)
```

See the [full leastsquares example](https://github.com/luk-ai/docs/blob/master/examples/leastsquares)
