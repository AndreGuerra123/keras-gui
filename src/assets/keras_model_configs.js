let l = require('lodash');
let KerasModelConf = {}

KerasModelConf.help = {
	optimizer: "str (name of optimizer) or optimizer object. See optimizers.",
	loss: "str (name of objective function) or objective function. See objectives.",
	metrics: "list of metrics to be evaluated by the model during training and testing. Typically you will use  metrics=['accuracy']. See metrics.",
	sample_weight_mode: "if you need to do timestep-wise sample weighting (2D weights), set this to 'temporal'. 'None' defaults to sample-wise weights (1D).",
	lr: "float >= 0. Learning rate.",
	beta_1: "float, 0 < beta < 1. Generally close to 1.",
	beta_2: "float, 0 < beta < 1. Generally close to 1.",
	epsilon: "float >= 0. Fuzz factor.",
	rho: "float >= 0.",
	decay: "float >= 0. Learning rate decay over each update.",
	momentum: "float >= 0. Parameter updates momentum.",
	nesterov: "boolean. Whether to apply Nesterov momentum.",
	input_length: "Length of input sequences, when it is constant. This argument is required if you are going to connect  Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).",
	padding: "int, or tuple of int (length 2), or dictionary.\n-If int: How many zeros to add at the beginning and end of the padding dimension (axis 1).\n-If tuple of int (length 2) How many zeros to add at the beginning and at the end of the padding dimension, in order '(left_pad, right_pad)'.\n-If dictionary: should contain the keys {'left_pad', 'right_pad'}. If any key is missing, default value of 0 will be used for the missing key.",
	length: "integer. Upsampling factor.",
	cropping: "tuple of int (length 2) How many units should be trimmed off at the beginning and end of the cropping dimension (axis 1).",
	kernel_dim1: "Length of the first dimension in the convolution kernel.",
	kernel_dim2: "Length of the second dimension in the convolution kernel.",
	kernel_dim3: "Length of the third dimension in the convolution kernel.",
	sigma: "float, standard deviation of the noise distribution",
	t_left_init: "initialization function for the left part intercept",
	a_left_init: "initialization function for the left part slope",
	t_right_init: "initialization function for the right part intercept",
	a_right_init: "initialization function for the right part slope",
	output_dim: "int > 0.",
	alpha: "float >= 0. Negative slope coefficient.",
	theta: "float >= 0. Threshold location of activation.",
	init: "name of initialization function for the weights of the layer (see initializations), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a weights argument.",
	activation: "name of activation function to use (see activations), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. 'linear' activation: a(x) = x).",
	weights: "list of Numpy arrays to set as initial weights. The list should have 2 elements, of shape (input_dim, output_dim) and (output_dim,) for weights and biases respectively.",
	W_regularizer: "instance of WeightRegularizer (eg. L1 or L2 regularization), applied to the main weights matrix.",
	b_regularizer: "instance of WeightRegularizer, applied to the bias.",
	activity_regularizer: "instance of ActivityRegularizer, applied to the network output.",
	W_constraint: "instance of the constraints module (eg. maxnorm, nonneg), applied to the main weights matrix.",
	b_constraint: "instance of the constraints module, applied to the bias.",
	bias: "whether to include a bias (i.e. make the layer affine rather than linear).",
	input_shape: "2D tensor with shape: ((nb_samples, input_dim)",
	input_dim: "dimensionality of the input (integer). This argument (or alternatively, the keyword argument input_shape) is required when using this layer as the first layer in a model.",
	p: "float between 0 and 1. Fraction of the input units to drop.",
	dim_ordering: "'th' or 'tf'. In 'th' mode, the channels dimension (the depth) is at index 1, in 'tf' mode is it at index 3. It defaults to the image_dim_ordering value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be 'tf'.",
	dims: "Tuple of integers. Permutation pattern, does not include the samples dimension. Indexing starts at 1. For instance,  (2, 1) permutes the first and second dimension of the input.",
	n: "integer, repetition factor.",
	layers: "can be a list of Keras tensors or a list of layer instances. Must be more than one layer/tensor.",
	mode: "string or lambda/function. If string, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'. If lambda/function, it should take as input a list of tensors and return a single tensor.",
	concat_axis: "integer, axis to use in mode concat.",
	dot_axes: "integer or tuple of integers, axes to use in mode dot or cos.",
	output_shape: "either a shape tuple (tuple of integers), or a lambda/function to compute output_shape (only if merge mode is a lambda/function). If the argument is a tuple, it should be expected output shape, not including the batch size (same convention as the input_shape argument in layers). If the argument is callable, it should take as input a list of shape tuples (1:1 mapping to input tensors) and return a single shape tuple, including the batch size (same convention as the  get_output_shape_for method of layers).",
	node_indices: "optional list of integers containing the output node index for each input layer (in case some input layers have multiple output nodes). will default to an array of 0s if not provided.",
	tensor_indices: "optional list of indices of output tensors to consider for merging (in case some input layer node returns multiple tensors).",
	output_mask: "mask or lambda/function to compute the output mask (only if merge mode is a lambda/function). If the latter case, it should take as input a list of masks and return a single mask.",
	"function": "The function to be evaluated. Takes input tensor as first argument.",
	arguments: "optional dictionary of keyword arguments to be passed to the function.",
	l1: "L1 regularization factor (positive float).",
	l2: "L2 regularization factor (positive float).",
	mask_value: "For each timestep in the input tensor (dimension #1 in the tensor), if all values in the input tensor at that timestep are equal to mask_value, then the timestep will masked (skipped) in all downstream layers (as long as they support masking).",
	nb_feature: "number of Dense layers to use internally.",
	nb_filter: "Number of convolution kernels to use (dimensionality of the output).",
	filter_length: "The extension (spatial or temporal) of each filter.",
	border_mode: "'valid' or 'same'.",
	subsample_length: "factor by which to subsample output.",
	layer: "Recurrent instance.",
	merge_mode: "Mode by which outputs of the forward and backward RNNs will be combined. One of {'sum', 'mul', 'concat', 'ave', None}. If None, the outputs will not be combined, they will be returned as a list.",
	mask_zero: "Whether or not the input value 0 is a special 'padding' value that should be masked out. This is useful for recurrent layers which may take variable length input. If this is True then all subsequent layers in the model need to support masking or an exception will be raised. If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary (input_dim should equal |vocabulary| + 2).",
	dropout: "float between 0 and 1. Fraction of the embeddings to drop.",
	pool_size: "tuple of D integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the image in each dimension.",
	strides: "tuple of D integers, or None. Strides values. If None, it will default to pool_size.",
	return_sequences: "Boolean. Whether to return the last output in the output sequence, or the full sequence.",
	go_backwards: "Boolean (default False). If True, process the input sequence backwards.",
	stateful: "Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.",
	unroll: "Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. When using TensorFlow, the network is always unrolled, so this argument does not do anything. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.",
	consume_less: "one of 'cpu', 'mem', or 'gpu' (LSTM/GRU only). If set to 'cpu', the RNN will use an implementation that uses fewer, larger matrix products, thus running faster on CPU but consuming more memory. If set to 'mem', the RNN will use more matrix products, but smaller ones, thus running slower (may actually be faster on GPU) while consuming less memory. If set to 'gpu' (LSTM/GRU only), the RNN will combine the input gate, the forget gate and the output gate into a single matrix, enabling more time-efficient parallelization on the GPU. Note: RNN dropout must be shared for all gates, resulting in a slightly reduced regularization.",
	inner_init: "initialization function of the inner cells.",
	U_regularizer: "instance of WeightRegularizer (eg. L1 or L2 regularization), applied to the recurrent weights matrices.",
	dropout_W: "float between 0 and 1. Fraction of the input units to drop for input gates.",
	dropout_U: "float between 0 and 1. Fraction of the input units to drop for recurrent connections.",
	beta_init: "name of initialization function for shift parameter (see initializations), or alternatively, Theano/TensorFlow function to use for weights initialization. This parameter is only relevant if you don't pass a weights argument.",
	gamma_init: "name of initialization function for scale parameter (see initializations), or alternatively, Theano/TensorFlow function to use for weights initialization. This parameter is only relevant if you don't pass a weights argument.",
	gamma_regularizer: "instance of WeightRegularizer (eg. L1 or L2 regularization), applied to the gamma vector.",
	beta_regularizer: "instance of WeightRegularizer, applied to the beta vector.",
	nb_row: "Number of rows in the convolution kernel.",
	nb_col: "Number of columns in the convolution kernel.",
	depth_multiplier: "how many output channel to use per input channel for the depthwise convolution step.",
	depthwise_regularizer: "instance of WeightRegularizer (eg. L1 or L2 regularization), applied to the depthwise weights matrix.",
	pointwise_regularizer: "instance of WeightRegularizer (eg. L1 or L2 regularization), applied to the pointwise weights matrix.",
	depthwise_constraint: "instance of the constraints module (eg. maxnorm, nonneg), applied to the depthwise weights matrix.",
	pointwise_constraint: "instance of the constraints module (eg. maxnorm, nonneg), applied to the pointwise weights matrix.",

}

KerasModelConf.enums = {
	Models: ['Model', "Sequential"],
	Layers: ["InputLayer",
		"Dense",
		"Activation",
		"Dropout",
		"SpatialDropout2D",
		"SpatialDropout3D",
		"Flatten",
		"Reshape",
		"Permute",
		"RepeatVector",
		"Merge",
		"Lambda",
		"ActivityRegularization",
		"Masking",
		"Highway",
		"MaxoutDense",
		"TimeDistributedDense",
		"Convolution1D",
		"AtrousConvolution1D",
		"Convolution2D",
		"AtrousConvolution2D",
		"SeparableConvolution2D",
		"Deconvolution2D",
		"Convolution3D",
		"Cropping1D",
		"Cropping2D",
		"Cropping3D",
		"UpSampling1D",
		"UpSampling2D",
		"UpSampling3D",
		"ZeroPadding1D",
		"ZeroPadding2D",
		"ZeroPadding3D",
		"MaxPooling1D",
		"MaxPooling2D",
		"MaxPooling3D",
		"AveragePooling1D",
		"AveragePooling2D",
		"AveragePooling3D",
		"GlobalMaxPooling1D",
		"GlobalAveragePooling1D",
		"GlobalMaxPooling2D",
		"GlobalAveragePooling2D",
		"GlobalMaxPooling3D",
		"GlobalAveragePooling3D",
		"LocallyConnected1D",
		"LocallyConnected2D",
		"LSTM",
		"GRU",
		"SimpleRNN",
		"Embedding",
		"LeakyReLU",
		"PReLU",
		"ELU",
		"ParametricSoftplus",
		"ThresholdedReLU",
		"SReLU",
		"BatchNormalization",
		"GaussianNoise",
		"GaussianDropout",
		"TimeDistributed",
		"Bidirectional"
	],

	Optimizers: [
		"SGD",
		"RMSprop",
		"Adagrad",
		"Adadelta",
		"Adam",
		"Adamax",
		"Nadam",
		"TFOptimizer"
	],

	loss: [
		"mean_squared_error",
		"mean_absolute_error",
		"mean_absolute_percentage_error",
		"mean_squared_logarithmic_error",
		"squared_hinge",
		"hinge",
		"binary_crossentropy",
		"categorical_crossentropy",
		"sparse_categorical_crossentropy",
		"kullback_leibler_divergence",
		"poisson",
		"cosine_proximity"
	],

	optimizer: [
		"sgd",
		"rmsprop",
		"adagrad",
		"adadelta",
		"adam",
		"adamax",
		"nadam",
		"custom"
	],

	input_dtype: [
		"float32",
		"float64"
	],

	activation: [
		"softmax",
		"softplus",
		"softsign",
		"relu",
		"tanh",
		"sigmoid",
		"hard_sigmoid",
		"linear"
	],

	inner_activation: [
		"softmax",
		"softplus",
		"softsign",
		"relu",
		"tanh",
		"sigmoid",
		"hard_sigmoid",
		"linear"
	],

	init: [
		"uniform",
		"lecun_uniform",
		"normal",
		"identity",
		"orthogonal",
		"zero",
		"glorot_normal",
		"glorot_uniform",
		"he_normal",
		"he_uniform"
	],

	inner_init: [
		"uniform",
		"lecun_uniform",
		"normal",
		"identity",
		"orthogonal",
		"zero",
		"glorot_normal",
		"glorot_uniform",
		"he_normal",
		"he_uniform"
	],

	forget_bias_init: [
		"uniform",
		"lecun_uniform",
		"normal",
		"identity",
		"orthogonal",
		"zero",
		"glorot_normal",
		"glorot_uniform",
		"he_normal",
		"he_uniform"
	],

	beta_init: [
		"uniform",
		"lecun_uniform",
		"normal",
		"identity",
		"orthogonal",
		"zero",
		"glorot_normal",
		"glorot_uniform",
		"he_normal",
		"he_uniform"
	],

	gamma_init: [
		"uniform",
		"lecun_uniform",
		"normal",
		"identity",
		"orthogonal",
		"zero",
		"glorot_normal",
		"glorot_uniform",
		"he_normal",
		"he_uniform"
	],

	t_left_init: [
		"uniform",
		"lecun_uniform",
		"normal",
		"identity",
		"orthogonal",
		"zero",
		"glorot_normal",
		"glorot_uniform",
		"he_normal",
		"he_uniform"
	],

	a_left_init: [
		"uniform",
		"lecun_uniform",
		"normal",
		"identity",
		"orthogonal",
		"zero",
		"glorot_normal",
		"glorot_uniform",
		"he_normal",
		"he_uniform"
	],

	t_right_init: [
		"uniform",
		"lecun_uniform",
		"normal",
		"identity",
		"orthogonal",
		"zero",
		"glorot_normal",
		"glorot_uniform",
		"he_normal",
		"he_uniform"
	],

	a_right_init: [
		"uniform",
		"lecun_uniform",
		"normal",
		"identity",
		"orthogonal",
		"zero",
		"glorot_normal",
		"glorot_uniform",
		"he_normal",
		"he_uniform"
	],

	border_mode: [
		"valid",
		"same"
	],

	dim_ordering: [
		"tf",
		"th",
		"default"
	],

	merge_mode: [
		'sum',
		'mul',
		'concat',
		'ave',
		"None"
	]
}

KerasModelConf.info = {
	InputLayer: {
		help: "A simple tensor input layer (batchsize, width, height, channels).",
		color: "#00ff00",
		args: ["batch_input_shape", "sparse", "input_dtype"]
	},
	Dense: {
		help: "Just your regular fully connected NN layer.",
		color: "#7f7f00",
		args: [
			"output_dim",
			"init",
			"activation",
			"weights",
			"W_regularizer",
			"b_regularizer",
			"activity_regularizer",
			"W_constraint",
			"b_constraint",
			"bias",
			"input_dim"
		]
	},

	Activation: {

		help: "Applies an activation function to an output.",
		color: "#ff5656",
		args: ["activation"]

	},

	Dropout: {
		help: "Applies Dropout to the input. Dropout consists in randomly setting a fraction p of input units to 0 at each update during training time, which helps prevent overfitting.",
		color: "#ff5656",
		args: ["p"]

	},

	SpatialDropout2D: {
		help: "This version performs the same function as Dropout, however it drops entire 2D feature maps instead of individual elements. If adjacent pixels within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout2D will help promote independence between feature maps and should be used instead.",
		color: "#ff5656",
		args: ["p", "dim_ordering"]
	},
	SpatialDropout3D: {
		help: "This version performs the same function as Dropout, however it drops entire 3D feature maps instead of individual elements. If adjacent pixels within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout#D will help promote independence between feature maps and should be used instead.",
		color: "#ff5656",
		args: ["p", "dim_ordering"]
	},
	Permute: {
		help: "Permutes the dimensions of the input according to a given pattern. Useful for e.g. connecting RNNs and convnets together.",
		color: "#000",
		args: ['dims']

	},

	Reshape: {
		help: "Reshapes an output to a certain shape.",
		color: "#000",
		args: []
	},

	Flatten: {
		help: "Flattens the input. Does not affect the batch size.",
		color: "#000",
		args: []
	},

	RepeatVector: {
		help: "Repeats the input n times.",
		color: "#000",
		args: ["n"]

	},

	Merge: {
		help: "A Merge layer can be used to merge a list of tensors into a single tensor, following some merge mode.",
		color: "#5fbf00",
		args: [
			"layers", "mode", "concat_axis", "output_shape", "node_indices", "tensor_indices", "output_mask"
		]
	},

	ActivityRegularization: {
		help: "Layer that passes through its input unchanged, but applies an update to the cost function based on the activity.",
		color: "#000",
		args: ["l1", "l2"]
	},
	Lambda: {
		help: "Used for evaluating an arbitrary Theano / TensorFlow expression on the output of the previous layer.",
		color: "#000",
		args: ['function', 'output_shape', 'arguments']

	},

	Masking: {
		help: "Masks an input sequence by using a mask value to identify timesteps to be skipped.",
		color: "#000",
		args: ["mask_value"]
	},

	Highway: {
		help: "Densely connected highway network, a natural extension of LSTMs to feedforward networks.",
		color: "#000",
		args: [
			"output_dim",
			"init",
			"activation",
			"weights",
			"W_regularizer",
			"b_regularizer",
			"activity_regularizer",
			"W_constraint",
			"b_constraint",
			"bias",
			"input_dim"
		]

	},
	MaxoutDense: {
		help: "A MaxoutDense layer takes the element-wise maximum of nb_feature Dense(input_dim, output_dim) linear layers. This allows the layer to learn a convex, piecewise linear activation function over the inputs.",
		color: "#ff0000",
		args: [
			"output_dim",
			"nb_features",
			"init",
			"activation",
			"weights",
			"W_regularizer",
			"b_regularizer",
			"activity_regularizer",
			"W_constraint",
			"b_constraint",
			"bias",
			"input_dim"
		]

	},
	TimeDistributedDense: {
		help: "Deprecated: use TimeDistributed",
		color: "#f00",
		args: ['deprecated']
	},

	Convolution1D: {
		help: "Convolution operator for filtering neighborhoods of one-dimensional inputs. When using this layer as the first layer in a model, either provide the keyword argument input_dim (int, e.g. 128 for sequences of 128-dimensional vectors), or input_shape (tuple of integers, e.g. (10, 128) for sequences of 10 vectors of 128-dimensional vectors).",
		color: "#005fbf",
		args: [
			'nb_filter',
			'filter_length',
			'border_mode',
			'subsample_length',
			'init',
			'activation',
			'weights',
			'W_regularizer',
			'b_regularizer',
			'activity_regularizer',
			'W_constraint',
			'b_constraint',
			'bias',
			'input_length'
		]

	},

	Convolution2D: {
		help: "Convolution operator for filtering windows of two-dimensional inputs. When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the sample axis), e.g.  input_shape=(3, 128, 128) for 128x128 RGB pictures.",
		color: "#005fbf",
		args: [
			'nb_filter',
			'nb_row',
			'nb_col',
			'filter_length',
			'border_mode',
			'subsample_length',
			'init',
			'activation',
			'weights',
			'W_regularizer',
			'b_regularizer',
			'activity_regularizer',
			'W_constraint',
			'b_constraint',
			'dim_ordering',
			'bias',
			'input_length'
		]

	},

	Deconvolution2D: {
		help: "Transposed convolution operator for filtering windows of two-dimensional inputs. The need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution. ",
		color: "#005fbf",
		args: [
			'nb_filter',
			'nb_row',
			'nb_col',
			'output_shape',
			'border_mode',
			'subsample',
			'init',
			'activation',
			'weights',
			'W_regularizer',
			'b_regularizer',
			'activity_regularizer',
			'W_constraint',
			'b_constraint',
			'dim_ordering',
			'bias'
		]
	},

	Convolution3D: {
		help: "Convolution operator for filtering windows of three-dimensional inputs. When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the sample axis), e.g.  input_shape=(3, 10, 128, 128) for 10 frames of 128x128 RGB pictures.",
		color: "#005fbf",
		args: [
			'nb_filter',
			'kernel_dim1',
			'kernel_dim2',
			'kernel_dim3',
			'border_mode',
			'subsample',
			'init',
			'activation',
			'weights',
			'W_regularizer',
			'b_regularizer',
			'activity_regularizer',
			'W_constraint',
			'b_constraint',
			'dim_ordering',
			'bias'
		]

	},

	Cropping1D: {
		help: "Cropping layer for 1D input (e.g. temporal sequence). It crops along the time dimension (axis 1).",
		color: "#ff5656",
		args: ['cropping']
	},

	Cropping2D: {
		help: "Cropping layer for 2D input (e.g. picture). It crops along spatial dimensions, i.e. width and height.",
		color: "#ff5656",
		args: ['cropping', "dim_ordering"]
	},

	Cropping3D: {
		help: "Cropping layer for 3D data (e.g. spatial or saptio-temporal).",
		color: "#ff5656",
		args: ['cropping', "dim_ordering"]
	},

	UpSampling1D: {
		help: "Repeat each temporal step length times along the time axis.",
		color: "#aaff56",
		args: ['length']
	},

	UpSampling2D: {
		help: "Repeat the rows and columns of the data by size[0] and size[1] respectively.",
		color: "#aaff56",
		args: ['size']
	},

	UpSampling3D: {
		help: "Repeat the first, second and third dimension of the data by size[0], size[1] and size[2] respectively.",
		color: "#aaff56",
		args: ['size', 'dim_ordering']
	},

	ZeroPadding1D: {
		help: "Zero-padding layer for 1D input (e.g. temporal sequence).",
		color: "#aaff56",
		args: ['padding']
	},

	ZeroPadding2D: {
		help: "Zero-padding layer for 1D input (e.g. temporal sequence).",
		color: "#aaff56",
		args: ['padding']
	},

	ZeroPadding3D: {
		help: "Zero-padding layer for 3D data (spatial or spatio-temporal).",
		color: "#aaff56",
		args: ['padding']
	},
	Bidirectional: {
		help: "Bidirectional wrapper for RNNs.",
		color: "#003f7f",
		args: ['layer', 'merge_mode']

	},
	Embedding: {
		help: "Turn positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]] \nThis layer can only be used as the first layer in a model.",
		color: "#003f7f",
		args: [
			'output_dim',
			'init',
			'activation',
			'weights',
			'W_regularizer',
			'b_regularizer',
			'activity_regularizer',
			'W_constraint',
			'b_constraint',
			'bias',
			'input_dim',
			'mask_zero',
			'dropout'
		]
	},
	MaxPooling1D: {
		help: "Max pooling operation for temporal data.",
		color: "#ff0000",
		args: ['pool_size', 'strides', 'border_mode']
	},
	MaxPooling2D: {
		help: "Max pooling operation for 2D data.",
		color: "#ff0000",
		args: ['pool_size', 'strides', 'border_mode', 'dim_ordering']

	},

	MaxPooling3D: {
		help: "Max pooling operation for 3D data (spatial or spatio-temporal).",
		color: "#ff0000",
		args: ['pool_size', 'strides', 'border_mode', 'dim_ordering']
	},


	AveragePooling1D: {
		help: "Average pooling operation for temporal data.",
		color: "#ff0000",
		args: ['pool_size', 'strides', 'border_mode']

	},


	AveragePooling2D: {
		help: "Average pooling operation for 2D data.",
		color: "#ff0000",
		args: ['pool_size', 'strides', 'border_mode', 'dim_ordering']

	},

	AveragePooling3D: {
		help: "Average pooling operation for 3D data (spatial or spatio-temporal).",
		color: "#ff0000",
		args: ['pool_size', 'strides', 'border_mode', 'dim_ordering']

	},

	GlobalMaxPooling1D: {
		help: "Global max pooling operation for temporal data.",
		color: "#ff0000",
		args: []
	},

	GlobalAveragePooling1D: {
		help: "Global average pooling operation for temporal data.",
		color: "#ff0000",
		args: []
	},

	GlobalMaxPooling2D: {
		help: "Global max pooling operation for spatial data.",
		color: "#ff0000",
		args: ['dim_ordering']
	},

	GlobalAveragePooling2D: {
		help: "Global average pooling operation for spatial data.",
		color: "#ff0000",
		args: ['dim_ordering']
	},

	LocallyConnected1D: {
		help: "The LocallyConnected1D layer works similarly to the Convolution1D layer, except that weights are unshared, that is, a different set of filters is applied at each different patch of the input. When using this layer as the first layer in a model, either provide the keyword argument input_dim (int, e.g. 128 for sequences of 128-dimensional vectors), or input_shape (tuple of integers, e.g. input_shape=(10, 128) for sequences of 10 vectors of 128-dimensional vectors). Also, note that this layer can only be used with a fully-specified input shape (None dimensions not allowed).",
		color: "#7f7f00",
		args: [
			'nb_filter',
			'filter_length',
			'init',
			'activation',
			'weights',
			'subsample_length',
			'W_regularizer',
			'b_regularizer',
			'activity_regularizer',
			'W_constraint',
			'b_constraint',
			'bias',
			'input_dim',
			'input_length'
		]
	},

	LocallyConnected2D: {
		help: "The LocallyConnected1D layer works similarly to the Convolution1D layer, except that weights are unshared, that is, a different set of filters is applied at each different patch of the input. When using this layer as the first layer in a model, either provide the keyword argument input_dim (int, e.g. 128 for sequences of 128-dimensional vectors), or input_shape (tuple of integers, e.g. input_shape=(10, 128) for sequences of 10 vectors of 128-dimensional vectors). Also, note that this layer can only be used with a fully-specified input shape (None dimensions not allowed).",
		color: "#7f7f00",
		args: [
			'nb_filter',
			'nb_row',
			'nb_col',
			'filter_length',
			'input_dim',
			'init',
			'activation',
			'weights',
			'subsample_length',
			'W_regularizer',
			'b_regularizer',
			'activity_regularizer',
			'W_constraint',
			'b_constraint',
			'bias',
			'dim_ordering'
		]
	},

	SimpleRNN: {
		help: "Fully-connected RNN where the output is to be fed back to input.",
		color: "#003f7f",
		args: [
			'output_dim',
			'init',
			'inner_init',
			'activation',
			'W_regularizer',
			'U_regularizer',
			'b_regularizer',
			'dropout_W',
			'dropout_U'
		]
	},
	GRU: {
		help: "Gated Recurrent Unit - Cho et al. 2014.",
		color: "#003f7f",
		args: [
			'output_dim',
			'init',
			'inner_init',
			'activation',
			'inner_activation',
			'W_regularizer',
			'U_regularizer',
			'b_regularizer',
			'dropout_W',
			'dropout_U'
		]

	},
	LSTM: {
		help: "Long-Short Term Memory unit - Hochreiter 1997.",
		color: "#003f7f",
		args: [
			'output_dim',
			'init',
			'inner_init',
			'forget_bias_init',
			'activation',
			'inner_activation',
			'W_regularizer',
			'U_regularizer',
			'b_regularizer',
			'dropout_W',
			'dropout_U'
		]

	},

	BatchNormalization: {
		help: "Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.",
		color: "#003f7f",
		args: [
			'epsilon',
			'mode',
			'axis',
			'momentum',
			'weights',
			'beta_init',
			'gamma_init',
			'gamma_regularizer',
			'beta_regularizer'
		]

	},

	SeparableConvolution2D: {
		help: "Separable convolution operator for 2D inputs. \n Separable convolutions consist in first performing a depthwise spatial convolution (which acts on each input channel separately) followed by a pointwise convolution which mixes together the resulting output channels. The  depth_multiplier argument controls how many output channels are generated per input channel in the depthwise step. \n Intuitively, separable convolutions can be understood as a way to factorize a convolution kernel into two smaller kernels, or as an extreme version of an Inception block.",
		color: "#007fff",
		args: [
			'nb_filter',
			'nb_row',
			'nb_col',
			'init',
			'activation',
			'weights',
			'border_mode',
			'subsample',
			'depth_multiplier',
			'depthwise_regularizer',
			'pointwise_regularizer',
			'b_regularizer',
			'activity_regularizer',
			'depthwise_constraint',
			'pointwise_constraint',
			'b_constraint',
			'dim_ordering',
			'bias'
		]

	},

	TimeDistributed: {
		help: "This wrapper allows to apply a layer to every temporal slice of an input. The input should be at least 3D, and the dimension of index one will be considered to be the temporal dimension. \nConsider a batch of 32 samples, where each sample is a sequence of 10 vectors of 16 dimensions. The batch input shape of the layer is then (32, 10, 16) (and the input_shape, not including the samples dimension, is  (10, 16)).",
		color: "#003f7f",
		args: ['layer']
	},

	LeakyReLU: {
		help: "Special version of a Rectified Linear Unit that allows a small gradient when the unit is not active: f(x) = alpha * x for x < 0, f(x) = x for x >= 0.",
		color: "#003f7f",
		args: ['alpha']
	},

	PReLU: {
		help: "Parametric Rectified Linear Unit: f(x) = alphas * x for x < 0, f(x) = x for x >= 0, where alphas is a learned array with the same shape as x.",
		color: "#003f7f",
		args: ['init', 'weights']
	},

	ELU: {
		help: "Exponential Linear Unit: f(x) =  alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0.",
		color: "#003f7f",
		args: ['alpha']
	},

	ParametricSoftplus: {
		help: "Parametric Softplus: alpha * log(1 + exp(beta * x))",
		color: "#003f7f",
		args: ['alpha_init', 'beta_init', 'weights']
	},

	ThresholdedReLU: {
		help: "Thresholded Rectified Linear Unit: f(x) = x for x > theta f(x) = 0 otherwise.",
		color: "#003f7f",
		args: ['theta']
	},

	SReLU: {
		help: "S-shaped Rectified Linear Unit.",
		color: "#003f7f",
		args: [
			't_left_init',
			'a_left_init',
			't_right_init',
			'a_right_init'
		]
	},

	GaussianNoise: {
		help: "Apply to the input an additive zero-centered Gaussian noise with standard deviation sigma. This is useful to mitigate overfitting (you could see it as a kind of random data augmentation). Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs. \nAs it is a regularization layer, it is only active at training time.",
		color: "#ffaaff",
		args: ['sigma']
	},

	GaussianDropout: {
		help: "Apply to the input an multiplicative one-centered Gaussian noise with standard deviation sqrt(p/(1-p)). \nAs it is a regularization layer, it is only active at training time.",
		color: "#ffaaff",
		args: ['p']
	},

	////////OPTIMIZERS////////
	SGD: {
		help: "Stochastic gradient descent, with support for momentum, learning rate decay, and Nesterov momentum.",
		color: "#000",
		args: [
			'lr',
			'momentum',
			'decay',
			'nesterov'
		]
	},
	RMSprop: {
		help: "It is recommended to leave the parameters of this optimizer at their default values (except the learning rate, which can be freely tuned). This optimizer is usually a good choice for recurrent neural networks.",
		color: "#000",
		args: [
			'lr',
			'rho',
			'decay',
			'epsilon'
		]
	},
	Adagrad: {
		help: "Adagrad optimizer. It is recommended to leave the parameters of this optimizer at their default values.",
		color: "#000",
		args: ['lr', 'epsilon', 'decay']
	},
	Adadelta: {
		help: "Adadelta optimizer. It is recommended to leave the parameters of this optimizer at their default values.",
		color: "#000",
		args: ['lr', 'epsilon', 'decay', 'rho']
	},
	Adam: {
		help: "Adam optimizer. Default parameters follow those provided in the original paper.",
		color: "#000",
		args: [
			'lr',
			'beta_1',
			'beta_2',
			'epsilon',
			'decay'
		]
	},
	Adamax: {
		help: "Adamax optimizer from Adam paper's Section 7. It is a variant of Adam based on the infinity norm. Default parameters follow those provided in the paper.",
		color: "#000",
		args: [
			'lr',
			'beta_1',
			'beta_2',
			'epsilon',
			'decay'
		]
	},
	Nadam: {
		help: "Nesterov Adam optimizer: Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum. Default parameters follow those provided in the paper. It is recommended to leave the parameters of this optimizer at their default values.",
		color: "#000",
		args: [
			'lr',
			'beta_1',
			'beta_2',
			'epsilon',
			'schedule_decay'
		]
	},
	TFOptimizer: {
		help: "TensorFlow Optimizer.",
		color: "#000",
		args: ['optimizer']
	},
	///////////MODELS/////////////////
	Model: {
		help: "Keras model.",
		color: "#00f",
		args: [
			'optimizer',
			'loss',
			'metrics',
			'sample_weight_mode'
		]
	},

	Sequential: {
		help: "Keras sequential model.",
		color: "#00f",
		args: [
			'optimizer',
			'loss',
			'metrics',
			'sample_weight_mode'
		]
	}
}

KerasModelConf.meta = {

	"InputLayer": "Core Layer",
	"Dense": "Core Layer",
	"Activation": "Core Layer",
	"Dropout": "Core Layer",
	"SpatialDropout2D": "Core Layer",
	"SpatialDropout3D": "Core Layer",
	"Flatten": "Core Layer",
	"Reshape": "Core Layer",
	"Permute": "Core Layer",
	"RepeatVector": "Core Layer",
	"Merge": "Core Layer",
	"Lambda": "Core Layer",
	"ActivityRegularization": "Core Layer",
	"Masking": "Core Layer",
	"Highway": "Core Layer",
	"MaxoutDense": "Core Layer",
	"TimeDistributedDense": "Core Layer",

	"Convolution1D": "Convolution Layer",
	"AtrousConvolution1D": "Convolution Layer",
	"Convolution2D": "Convolution Layer",
	"AtrousConvolution2D": "Convolution Layer",
	"SeparableConvolution2D": "Convolution Layer",
	"Deconvolution2D": "Convolution Layer",
	"Convolution3D": "Convolution Layer",
	"Cropping1D": "Convolution Layer",
	"Cropping2D": "Convolution Layer",
	"Cropping3D": "Convolution Layer",
	"UpSampling1D": "Convolution Layer",
	"UpSampling2D": "Convolution Layer",
	"UpSampling3D": "Convolution Layer",
	"ZeroPadding1D": "Convolution Layer",
	"ZeroPadding2D": "Convolution Layer",
	"ZeroPadding3D": "Convolution Layer",

	"MaxPooling1D": "Pooling Layer",
	"MaxPooling2D": "Pooling Layer",
	"MaxPooling3D": "Pooling Layer",
	"AveragePooling1D": "Pooling Layer",
	"AveragePooling2D": "Pooling Layer",
	"AveragePooling3D": "Pooling Layer",
	"GlobalMaxPooling1D": "Pooling Layer",
	"GlobalAveragePooling1D": "Pooling Layer",
	"GlobalMaxPooling2D": "Pooling Layer",
	"GlobalAveragePooling2D": "Pooling Layer",
	"GlobalMaxPooling3D": "Pooling Layer",
	"GlobalAveragePooling3D": "Pooling Layer",

	"LocallyConnected1D": "Locally-connected Layer",
	"LocallyConnected2D": "Locally-connected Layer",


	"LSTM": 'Recurrent Layer',
	"GRU": 'Recurrent Layer',
	"SimpleRNN": 'Recurrent Layer',

	"Embedding": 'Embedding Layer',

	"LeakyReLU": "Advanced Activation Layer",
	"PReLU": "Advanced Activation Layer",
	"ELU": "Advanced Activation Layer",
	"ParametricSoftplus": "Advanced Activation Layer",
	"ThresholdedReLU": "Advanced Activation Layer",
	"SReLU": "Advanced Activation Layer",

	"BatchNormalization": 'Normalization Layer',

	"GaussianNoise": "Noise Layer",
	"GaussianDropout": "Noise Layer",

	"TimeDistributed": "Layer wrapper",
	"Bidirectional": "Layer wrapper",

	"SGD": "Optimizer",
	"RMSprop": "Optimizer",
	"Adagrad": "Optimizer",
	"Adadelta": "Optimizer",
	"Adam": "Optimizer",
	"Adamax": "Optimizer",
	"Nadam": "Optimizer",
	"TFOptimizer": "Optimizer",

	"Model": "Model",
	"Sequential": "Model"
}

KerasModelConf.default_schemas = {
	a_left_init: {
		"default": "uniform",
		"type": "string"
	},
	a_right_init: {
		"default": "uniform",
		"type": "string"
	},
	activation: {
		"default": "tanh",
		"type": "string"
	},
	activity_regularizer: {
		"default": "activity_l2(0.01)",
		"type": "string"
	},
	alpha: {
		"default": 0.0,
		"type": "number"
	},
	alpha_init: {
		"default": 0.5,
		"type": "number"
	},
	arguments: {
		"default": "",
		"type": "string"
	},
	axis: {
		"default": 1,
		"type": "integer"
	},
	b_constraint: {
		"default": "maxnorm(m=2)",
		"type": "string"
	},
	b_regularizer: {
		"default": "l2(0.01)",
		"type": "string"
	},
	batch_input_shape: {
		"default": [null, 299, 299, 3],
		"type": "array"
	},
	beta: {
		"default": "l2(0.01)",
		"type": "string"
	},
	beta_1: {
		"default": 0.9,
		"type": "number"
	},
	beta_2: {
		"default": 0.999,
		"type": "number"
	},
	beta_init: {
		"default": "uniform",
		"type": "string"
	},
	bias: {
		"default": true,
		"type": "boolean"
	},
	border_mode: {
		"default": "valid",
		"type": "string"
	},
	concat_axis: {
		"default": 2,
		"type": "integer"
	},
	cropping: {
		"default": "(1, 1)",
		"type": "string"
	},
	decay: {
		"default": 0.9,
		"type": "number"
	},
	deprecated: {
		"default": "This class is deprecated.",
		"type": "string"
	},
	depth_multiplier: {
		"default": "",
		"type": "string"
	},
	depthwise_constraint: {
		"default": "",
		"type": "string"
	},
	depthwise_regularizer: {
		"default": "l2(0.01)",
		"type": "string"
	},
	dim_ordering: {
		"default": "tf",
		"type": "string"
	},
	dims: {
		"default": "(2, 1)",
		"type": "string"
	},
	dot_axes: {
		"default": "(tuple)",
		"type": "string"
	},
	dropout: {
		"default": 0.5,
		"type": "number"
	},
	dropout_U: {
		"default": 0.5,
		"type": "number"
	},
	dropout_W: {
		"default": 0.5,
		"type": "number"
	},
	epsilon: {
		"default": 0.01,
		"type": "number"
	},
	filter_length: {
		"default": "",
		"type": "string"
	},
	forget_bias_init: {
		"default": "uniform",
		"type": "string"
	},
	function: {
		"default": "",
		"type": "string"
	},
	gamma_init: {
		"default": "uniform",
		"type": "string"
	},
	gamma_regularizer: {
		"default": "l2(0.01)",
		"type": "string"
	},
	init: {
		"default": "uniform",
		"type": "string"
	},
	init_activation: {
		"default": "tanh",
		"type": "string"
	},
	inner_init: {
		"default": "uniform",
		"type": "string"
	},
	input_dim: {
		"default": 200,
		"type": "integer"
	},
	input_dtype: {
		"default": "float32",
		"type": "string"
	},
	input_length: {
		"default": 300,
		"type": "integer"
	},
	kernel_dim1: {
		"default": 200,
		"type": "integer"
	},
	kernel_dim2: {
		"default": 200,
		"type": "integer"
	},
	kernel_dim3: {
		"default": 200,
		"type": "integer"
	},
	l1: {
		"default": 0.0,
		"type": "number"
	},
	l2: {
		"default": 0.0,
		"type": "number"
	},
	layer: {
		"default": "",
		"type": "string"
	},
	layers: {
		"default": "[layer1, layer2]",
		"type": "string"
	},
	length: {
		"default": 2,
		"type": 'integer'
	},
	loss: {
		"default": "mean_squared_error",
		"type": "string"
	},
	lr: {
		"default": 0.1,
		"type": "number"
	},
	mask_value: {
		"default": 0.0,
		"type": "number"
	},
	mask_zero: {
		"default": false,
		"type": "boolean"
	},
	metrics: {
		"default": "",
		"type": "string"
	},
	mode: {
		"default": "sum",
		"type": "string"
	},
	momentum: {
		"default": 0.1,
		"type": "number"
	},
	n: {
		"default": 3,
		"type": "integer"
	},
	nb_col: {
		"default": 200,
		"type": "integer"
	},
	nb_features: {
		"default": 2,
		"type": "integer"
	},
	nb_filter: {
		"default": 1,
		"type": "integer"
	},
	nb_row: {
		"default": 200,
		"type": "integer"
	},
	nesterov: {
		"default": false,
		"type": "boolean"
	},
	node_indices: {
		"default": "",
		"type": "string"
	},
	optimizer: {
		"default": 'sgd',
		"type": "string"
	},
	output_dim: {
		"default": 1,
		"type": "integer"
	},
	output_mask: {
		"default": "",
		"type": "string"
	},
	output_shape: {
		"default": "(tuple integers)",
		"type": "string"
	},
	p: {
		"default": 0.5,
		"type": "number"
	},
	padding: {
		"default": 1,
		"type": "integer"
	},
	pointwise_constraint: {
		"default": "",
		"type": "string"
	},
	pointwise_regularizer: {
		"default": "l2(0.01)",
		"type": "string"
	},
	pool_size: {
		"default": "",
		"type": "string"
	},
	rho: {
		"default": 0.9,
		"type": "number"
	},
	sample_weight_mode: {
		"default": "None",
		"type": "string"
	},
	schedule_decay: {
		"default": 0.04,
		"type": "number"
	},
	sigma: {
		"default": 0.5,
		"type": "number"
	},
	size: {
		"default": '(2,2)',
		"type": "string"
	},
	sparse: {
		"default": false,
		"type": "boolean"
	},
	strides: {
		"default": "",
		"type": "string"
	},
	subsample: {
		"default": "",
		"type": "string"
	},
	subsample_length: {
		"default": 300,
		"type": "integer"
	},
	t_left_init: {
		"default": "uniform",
		"type": "string"
	},
	t_right_init: {
		"default": "uniform",
		"type": "string"
	},
	tensor_indices: {
		"default": "",
		"type": "string"
	},
	theta: {
		"default": 0.5,
		"type": "number"
	},
	U_regularizer: {
		"default": "l2(0.01)",
		"type": "string"
	},
	W_constraint: {
		"default": "maxnorm(m=2)",
		"type": "string"
	},
	W_regularizer: {
		"default": "l2(0.01)",
		"type": "string"
	},
	weights: {
		"default": "",
		"type": "string"
	}
}


KerasModelConf.override_schemas = {
	Lambda: {
		output_shape: {
			"type": "string",
			"default": "(input_shape[0], )"
		},
	},
	TimeDistributedDense: {

		deprecated: {
			"type": "string",
			"default": "use TimeDistributed"
		}

	},
	Deconvolution2D: {
		output_shape: {
			"type": "string",
			"default": ""
		},
	},
	Cropping2D: {
		cropping: {
			"type": "string",
			"default": "((0,0),(0,0))"
		}
	},
	Cropping3D: {
		cropping: {
			"type": "string",
			"default": "((1,1),(1,1),(1,1))"
		}
	},
	UpSampling3D: {
		size: {
			"type": "string",
			"default": "(2,2,2)"
		}
	},
	ZeroPadding2D: {
		padding: {
			"type": "string",
			"default": (1, 1)
		}
	},
	ZeroPadding3D: {
		padding: {
			"type": "string",
			"default": (1, 1, 1)
		}
	},
	LocallyConnected1D: {
		input_length: {
			"type": "string",
			"default": "(None)"
		}
	},
	SimpleRNN: {
		output_dim: {
			"type": "integer",
			"default": 200
		}
	},
	GRU: {
		output_dim: {
			"type": "integer",
			"default": 200
		}
	},
	LSTM: {
		output_dim: {
			"type": "integer",
			"default": 200
		}
	},
	BatchNormalization: {
		mode: {
			"type": "integer",
			"default": 0
		}
	},
	ELU: {
		alpha: {
			"type": "number",
			"default": 1
		}
	},
	SGD: {
		momentum: {
			"type": "number",
			"default": 0.9
		}
	},
	RMSprop: {
		decay: {
			"type": "number",
			"default": 0.005
		},
		epsilon: {
			"type": "number",
			"default": 0.0001
		}
	},
	Adagrad: {
		lr: {
			"type": "number",
			"default": 0.01
		},
		decay: {
			"type": "number",
			"default": 0.0
		},
		epsilon: {
			"type": "number",
			"default": 1e-08
		}
	},
	Adadelta: {
		lr: {
			"type": "number",
			"default": 0.01
		},
		decay: {
			"type": "number",
			"default": 0.0
		},
		epsilon: {
			"type": "number",
			"default": 1e-08
		},
		rho: {
			"type": "number",
			"default": 0.95
		}
	},
	Adam: {
		lr: {
			"type": "number",
			"default": 0.001
		},
		decay: {
			"type": "number",
			"default": 0.0
		},
		epsilon: {
			"type": "number",
			"default": 1e-08
		}
	},
	Adamax: {
		lr: {
			"type": "number",
			"default": 0.002
		},
		decay: {
			"type": "number",
			"default": 0.0
		},
		epsilon: {
			"type": "number",
			"default": 1e-08
		}
	},
	Nadam: {
		lr: {
			"type": "number",
			"default": 0.002
		},
		epsilon: {
			"type": "number",
			"default": 1e-08
		}
	}
}

KerasModelConf.layer_core_schema = {
	class_name: {
		"type": "string",
		"required": true,
		"default": "Dense",
		"enum": KerasModelConf.enums['Layers']
	},
	name: {
		"type": "string",
		"required": false,
		"default": "dense_"
	},
	inbound_nodes: {
		"type": "array",
		"required": true,
		"default": []
	},
	trainable: {
		"type": "boolean",
		"required": false,
		"default": true
	}

}

KerasModelConf.optimizer_core_schema = {
	class_name: {
		"type": "string",
		"required": true,
		"default": "SGD",
		"enum": KerasModelConf.enums['Optimizers']
	},
	name: {
		"type": "string",
		"required": false,
		"default": "sgd_"
	},
	trainable: {
		"type": "boolean",
		"required": false,
		"default": true
	}
}

KerasModelConf.model_core_schema = {
	class_name: {
		"type": "string",
		"required": true,
		"default": "Model",
		"enum": KerasModelConf.enums['Models']
	},
	name: {
		"type": "string",
		"required": false,
		"default": "model_"
	},
	inbound_nodes: {
		"type": "array",
		"required": false,
		"default": []
	},
	trainable: {
		"type": "boolean",
		"required": false,
		"default": true
	}
}
KerasModelConf.getInfo = function (obj_class) {

	let layerinfo = l.get(KerasModelConf.info, obj_class, null) || {
		help: "No description provided.",
		color: "#ffffff",
		args: []
	}
	layerinfo.meta = l.get(KerasModelConf.meta, obj_class, null) || "Unknown meta-class."
	return layerinfo;
}

KerasModelConf.isModel = function (obj_class) {
	return l.some(KerasModelConf.enums['Models'], obj_class);
}

KerasModelConf.isLayer = function (obj_class) {
	return l.some(KerasModelConf.enums['Layers'], obj_class);
}

KerasModelConf.isOptimizer = function (obj_class) {
	return l.some(KerasModelConf.enums['Optimizers'], obj_class);
}

KerasModelConf.getOverrideArguments = function (obj_class) {
	return l.get(KerasModelConf.override_schemas, obj_class, {})
}
KerasModelConf.getEnumeration = function (arg) {
	return l.get(KerasModelConf.enums, arg, null)
}
KerasModelConf.getDescription = function (arg) {
	return l.get(KerasModelConf.help, arg, null)
}
KerasModelConf.getCoreSchema = function (obj_class) {
	if (KerasModelConf.isModel(obj_class)) {
		return KerasModelConf.model_core_schema;

	} else if (KerasModelConf.isLayer(obj_class)) {
		return KerasModelConf.layer_core_schema;

	} else if (KerasModelConf.isOptimizer(obj_class)) {
		return KerasModelConf.optimizer_core_schema;

	} else {
		return {}
	}
}
KerasModelConf.getArguments = function (obj_class) {
	return l.get(KerasModelConf.getInfo(obj_class), "args", [])
}
KerasModelConf.getDefaultSchema = function (aa) {

	let toreturn = {};
	l.map(aa, function (a) {
		var tmp_arg_schema = l.get(KerasModelConf.default_schemas, a, {})
		if (KerasModelConf.getEnumeration(a)) {
			tmp_arg_schema.enum = KerasModelConf.getEnumeration(a)
		}
		if (KerasModelConf.getDescription(a)) {
			tmp_arg_schema.description = KerasModelConf.getDescription(a)
		}
		toreturn[a] = tmp_arg_schema
	})

	return toreturn;
}

KerasModelConf.getOverrideSchema = function (obj_class) {
	return l.get(KerasModelConf.override_schema, obj_class, {})
}

KerasModelConf.getProperties = function (obj_class) {

	let cs = KerasModelConf.getCoreSchema()

	let aa = KerasModelConf.getArguments(obj_class)

	let ds = KerasModelConf.getDefaultSchema(aa)

	let os = KerasModelConf.getOverrideSchema()

	return l.merge(cs, ds, os)

}

KerasModelConf.getClassDescription = function (obj_class) {

	},

	KerasModelConf.getClassProperties = function (obj_class) {

	},

	KerasModelConf.getSchema = function (class_name, obj_class) {

		return {
			"type": "object",
			"title": class_name + ' - ' + KerasModelConf.getClassMeta(obj_class),
			"description": KerasModelConf.getClassDescription(obj_class),
			"properties": KerasModelConf.getProperties(obj_class)
		}

	}


module.exports = KerasModelConf;