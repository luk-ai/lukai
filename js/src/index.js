function stringify (obj) {
  return JSON.stringify(obj, (k, v) => {
    if (v instanceof tf.Tensor) {
      const data = v.dataSync()
      return {
        data,
        ...v
      }
    } else if (v instanceof Float32Array) {
      return Array.from(v)
    }
    return v
  })
}

export function serialize (model) {
  const {optimizer, loss, metrics, metricsNames} = model
  return model.save({
    save (data) {
      const {modelTopology, weightData, weightSpecs} = data

      const out = {
        modelTopology,
        weightsManifest: [{
          paths: ['weights.bin'],
          weights: weightSpecs
        }],
        optimizer,
        loss,
        metrics,
        metricsNames
      }
      return {
        'model.json': stringify(out),
        'weights.bin': weightData
      }
    }
  })
}
