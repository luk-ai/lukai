'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

exports.serialize = serialize;
function stringify(obj) {
  return JSON.stringify(obj, function (k, v) {
    if (v instanceof tf.Tensor) {
      var data = v.dataSync();
      return _extends({
        data: data
      }, v);
    } else if (v instanceof Float32Array) {
      return Array.from(v);
    }
    return v;
  });
}

function serialize(model) {
  var optimizer = model.optimizer,
      loss = model.loss,
      metrics = model.metrics,
      metricsNames = model.metricsNames;

  return model.save({
    save: function save(data) {
      var modelTopology = data.modelTopology,
          weightData = data.weightData,
          weightSpecs = data.weightSpecs;


      var out = {
        modelTopology: modelTopology,
        weightsManifest: [{
          paths: ['weights.bin'],
          weights: weightSpecs
        }],
        optimizer: optimizer,
        loss: loss,
        metrics: metrics,
        metricsNames: metricsNames
      };
      return {
        'model.json': stringify(out),
        'weights.bin': weightData
      };
    }
  });
}