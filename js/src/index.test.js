import {assert} from 'chai'
import * as lukai from './index.js'
import * as tf from '@tensorflow/tfjs'
import * as tmp from 'tmp'
import * as fse from 'fs-extra'

require('@tensorflow/tfjs-node') // Use '@tensorflow/tfjs-node-gpu' if running with GPU.
tf.setBackend('tensorflow')

describe('lukai', function () {
  const model = tf.sequential()
  model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}))
  model.add(tf.layers.dense({units: 1, activation: 'linear'}))

  const compileArgs = {optimizer: 'sgd', loss: 'categoricalCrossentropy'}
  model.compile(compileArgs)

  it('should should be able to serialize a model', function () {
    return lukai.serialize(model).then((model) => {
      assert.hasAllKeys(model, ['model.json', 'weights.bin'])
    })
  })

  it('should should be able to save and load a model', function () {
    const tmpobj = tmp.dirSync()
    const dir = tmpobj.name

    return lukai.serialize(model).then((model) => {
      return Promise.all(Object.keys(model).map((key) => {
        return fse.outputFile(dir + '/' + key, Buffer.from(model[key]))
      }))
    }).then(() => {
      return tf.loadModel('file://' + dir + '/model.json')
    })
  })
})
