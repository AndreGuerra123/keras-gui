<template>
  <div id="app">
    <kerasgui v-bind:json="json_object" ></kerasgui>
  </div>
</template>

<script>
  import kerasgui from './components/kerasgui.vue'

  export default {
    name: 'app',
    data: function () {
      return {
        json_object: JSON.parse('{"keras_version": "1.0.7", "class_name": "Sequential", "config": [{"class_name": "Convolution2D", "config": {"bias": true, "batch_input_shape": [null, 28, 28, 1], "W_regularizer": null, "activity_regularizer": null, "trainable": true, "nb_row": 3, "subsample": [1, 1], "dim_ordering": "tf", "W_constraint": null, "init": "glorot_uniform", "b_constraint": null, "input_dtype": "float32", "nb_filter": 32, "name": "convolution2d_1", "b_regularizer": null, "activation": "linear", "nb_col": 3, "border_mode": "valid"}}, {"class_name": "Activation", "config": {"trainable": true, "activation": "relu", "name": "activation_1"}}, {"class_name": "Convolution2D", "config": {"bias": true, "W_regularizer": null, "activity_regularizer": null, "trainable": true, "nb_row": 3, "subsample": [1, 1], "dim_ordering": "tf", "W_constraint": null, "init": "glorot_uniform", "b_constraint": null, "nb_filter": 32, "name": "convolution2d_2", "b_regularizer": null, "activation": "linear", "nb_col": 3, "border_mode": "valid"}}, {"class_name": "Activation", "config": {"trainable": true, "activation": "relu", "name": "activation_2"}}, {"class_name": "MaxPooling2D", "config": {"trainable": true, "strides": [2, 2], "name": "maxpooling2d_1", "dim_ordering": "tf", "pool_size": [2, 2], "border_mode": "valid"}}, {"class_name": "Dropout", "config": {"trainable": true, "name": "dropout_1", "p": 0.25}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true}}, {"class_name": "Dense", "config": {"bias": true, "W_regularizer": null, "activity_regularizer": null, "b_regularizer": null, "trainable": true, "b_constraint": null, "input_dim": null, "init": "glorot_uniform", "W_constraint": null, "output_dim": 128, "activation": "linear", "name": "dense_1"}}, {"class_name": "Activation", "config": {"trainable": true, "activation": "relu", "name": "activation_3"}}, {"class_name": "Dropout", "config": {"trainable": true, "name": "dropout_2", "p": 0.5}}, {"class_name": "Dense", "config": {"bias": true, "W_regularizer": null, "activity_regularizer": null, "b_regularizer": null, "trainable": true, "b_constraint": null, "input_dim": null, "init": "glorot_uniform", "W_constraint": null, "output_dim": 10, "activation": "linear", "name": "dense_2"}}, {"class_name": "Activation", "config": {"trainable": true, "activation": "softmax", "name": "activation_4"}}]}')
      }
    },
    components: {
      kerasgui
    }
  }
</script>

<style>
  #app {
    position: absolute;
    margin: auto;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
  }
</style>