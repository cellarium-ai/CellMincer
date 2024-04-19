.. _reference:

Reference
#########

``preprocess``
--------------

Command line options
~~~~~~~~~~~~~~~~~~~~

.. argparse::
    :module: cellmincer.cli.base_cli
    :func: get_populated_argparser
    :prog: cellmincer
    :path: preprocess
    :nodefaultconst:

Configuration options
~~~~~~~~~~~~~~~~~~~~~

(please stay tuned for an upcoming documentation update)

**bfgs.history_size :** *int*

**bfgs.line_search_fn :** *string*

**bfgs.lr :** *float*

**bfgs.max_iter :** *int*

**bfgs.tolerance_change :** *float*

**bfgs.tolerance_grad :** *float*

**dejitter.detrending_method :** *string*

**dejitter.detrending_order :** *int*

**dejitter.enabled :** *bool*

**dejitter.show_diagnostic_plots :** *bool*

**dejitter.stft_lp_cutoff :** *float*

**dejitter.stft_lp_slope :** *float*

**dejitter.stft_noverlap :** *int*

**dejitter.stft_nperseg :** *int*

**detrend.smoothing :** *string or null*

**detrend.init_unc_decay_rate :** *float*

**detrend.max_iters_per_segment :** *int*

**detrend.plot_segments :** *bool*

**detrend.poly_order :** *int*

**detrend.trend_model :** *string*

**device :** *string*
    The device used by PyTorch for trend fitting.

**noise_estimation.n_bootstrap :** *int*

**noise_estimation.plot_example :** *bool*

**noise_estimation.plot_subsample :** *int*

**noise_estimation.stationarity_window :** *int*

**trim.n_frames_fit_left :** *int*
    The number of frames (after trimming) on the left end of each segment to fit the trend.

**trim.n_frames_fit_right :** *int*
    The number of frames (after trimming) on the right end of each segment to fit the trend.

**trim.trim_left :** *int*
    The number of frames to trim off the left end of each segment.

**trim.trim_right :** *int*
    The number of frames to trim off the right end of each segment.

``train``
---------

Command line options
~~~~~~~~~~~~~~~~~~~~

.. argparse::
    :module: cellmincer.cli.base_cli
    :func: get_populated_argparser
    :prog: cellmincer
    :path: train
    :nodefaultconst:

Configuration options
~~~~~~~~~~~~~~~~~~~~~

**model.occlude_padding :** *bool*
    Enables pixel masking on every frame padding pixel when reflection padding is used, preventing the model from "cheating" in its prediction task. Recommended to set True, particularly when training datasets have narrow frames and image crops often include the edge of the frame. (Note: padding is never occluded for data denoising.)

**model.padding_mode :** *string*
    The per-frame padding strategy for training and denoising, as one of the following string values.
    
    - 'reflect'
        Pads with the reflection of each frame along its edges.
    - 'constant'
        Pads with zeros.

**model.spatial_unet_activation :** *string*
    The conditional U-Net's activation function, as one of the following string values.

    .. list-table::
       :widths: 10 15
       :header-rows: 0

       * - 'relu'
         - ``torch.nn.ReLU()``
       * - 'elu'
         - ``torch.nn.ELU()``
       * - 'selu'
         - ``torch.nn.SELU()``
       * - 'sigmoid'
         - ``torch.nn.Sigmoid()``
       * - 'leaky_relu'
         - ``torch.nn.LeakyReLU()``
       * - 'softplus'
         - ``torch.nn.Softplus()``

**model.spatial_unet_attention :** *bool*
    Enables U-Net local attention.

**model.spatial_unet_batch_norm :** *bool*
    Enables U-Net batch normalization after each activation.

**model.spatial_unet_depth :** *int*
    Number of layers in the U-Net contraction and expansion path.

**model.spatial_unet_feature_mode :** *string*
    Configures the conditioning of the U-Net on global features, as one of the following string values.

    - 'repeat'
        At the beginning and before each subsequent step of the contracting path, concatenates an appropriately downsampled version of the global feature tensor to the partial embedding product.
    - 'once'
        Global features concatenated to input of U-Net.
    - 'none'
        No use of global features.

**model.spatial_unet_first_conv_channels :** *int*
    Number of output channels from the first convolution layer. After each contracting step, the channel size doubles.

**model.spatial_unet_kernel_size :** *int*
    U-Net convolution kernel size.

**model.spatial_unet_n_conv_layers :** *int*
    Number of convolution layers at each U-Net step.

**model.spatial_unet_padding :** *bool*
    Enables padding after each convolution layer. Set False when using whole-frame padding.

**model.spatial_unet_readout_kernel_size :** *int*
    Kernel size for processing readout from U-Net output. Not used in training.

**model.temporal_denoiser_activation :** *string*
    The temporal post-processor's activation function. See **model.spatial_unet_activation :** for permissible values.

**model.temporal_denoiser_conv_channels :** *int*
    The number of channels following the first temporal convolution (remains fixed for subsequent convolution layers).

**model.temporal_denoiser_hidden_dense_layer_dims :** *list[int]*
    The sequence of hidden layer dimensions in the temporal post-processor's channel contraction step.

**model.temporal_denoiser_kernel_size :** *int*
    Width of 1D convolutional kernel over the time dimension.

**model.temporal_denoiser_n_conv_layers :** *int*
    Number of time convolution layers.

**model.type :** *string*
    Name of model variation. As of CellMincer 0.1.0, the only available model variation is 'spatial-unet-2d-temporal-denoiser'.

.. note::
    The options 'model.temporal_denoiser_kernel_size' and 'model.temporal_denoiser_n_conv_layers' implicitly determine the model's effective context size through the following formula:
    
    .. math::
        \text{context_size}=1 + \text{n_conv_layers}\times(\text{kernel_size} - 1)

**train.importance :** *dict or null*
    If not null, the hyperparameters for biasing the training dataloader with importance sampling for high-intensity crops.
    
    **train.importance.n_samples :** *int*
        The number of crops sampled from each training dataset to estimate its intensity threshold.
    
    **train.importance.pivot :** *float*
        A value between 0 and 1 denoting the high-intensity proportion of crops to be resampled. For example, if pivot were set to 0.001, the most intensive 0.1\% of crops will be resampled to 50\% of each training minibatch. 

**train.lr_params :** *dict*
    The learning rate scheduler settings. Below are the options for **train.lr_params.type** and each type's associated hyperparameters.
    
    - 'constant': A fixed learning rate across training iterations.
        **train.lr_params.max_lr :** *float* -- the learning rate.
    - 'cosine-annealing-warmup': A cosine-annealing with linear warmup scheduler [implemented here](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/).
        **train.lr_params.max_lr :** *float* -- the maximum learning rate.
        **train.lr_params.min_lr :** *float* -- the minimum learning rate.
        **train.lr_params.warmup :** *float* -- the proportion of initial training allocated to linearly ramping from **min_lr** to **max_lr**.

**train.n_batch :** *int*
    The number of entries per device per minibatch.

**train.n_iters :** *int*
    The number of training iterations.

**train.norm_p :** *int*
    The parameterization of Lp loss.

**train.occlusion_prob :** *float*
    The Bernoulli parameter for masking pixels during training.

**train.occlusion_radius :** *int*
    The radius of additional occlusion centered on each masked pixel.

**train.optim_params :** *dict*
    The optimizer settings. Below are the options for **train.optim_params.type** and each type's associated hyperparameters.
    
    - 'adam': Adam optimizer.
        **train.optim_params.betas :** *list[float]* -- :math:`\beta_1` and :math:`\beta_2`.
        **train.optim_params.weight_decay :** *float* -- Weight decay parameter.
    - 'sgd': SGD optimizer.
        **train.lr_params.momentum :** *float* -- Momentum parameter.

**train.output_min_size_lims :** *list[int]*
    Lower and upper limits of training crop output size. At the start of training, the size maximizing the ratio of output size to (padded) receptive field is selected.

**train.t_tandem :** *int*
    Number of consecutive "middle" frames in which pixel masking is performed, in a context window.

``denoise``
-----------

Command line options
~~~~~~~~~~~~~~~~~~~~

.. argparse::
    :module: cellmincer.cli.base_cli
    :func: get_populated_argparser
    :prog: cellmincer
    :path: denoise
    :nodefaultconst: