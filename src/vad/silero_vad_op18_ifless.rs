// Generated from ONNX "../silero_vad_op18_ifless.onnx" by burn-onnx
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LinearLayout;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::Conv1d;
use burn::nn::conv::Conv1dConfig;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

use crate::custom_kernels::{CustomKernelsBackend, lstm_cell_fused};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    constant32: burn::module::Param<Tensor<B, 1, Int>>,
    constant41: burn::module::Param<Tensor<B, 1, Int>>,
    constant42: burn::module::Param<Tensor<B, 1>>,
    conv1d37: Conv1d<B>,
    conv1d38: Conv1d<B>,
    conv1d39: Conv1d<B>,
    conv1d40: Conv1d<B>,
    conv1d41: Conv1d<B>,
    linear13: Linear<B>,
    linear14: Linear<B>,
    conv1d42: Conv1d<B>,
    conv1d43: Conv1d<B>,
    conv1d44: Conv1d<B>,
    conv1d45: Conv1d<B>,
    conv1d46: Conv1d<B>,
    conv1d47: Conv1d<B>,
    linear15: Linear<B>,
    linear16: Linear<B>,
    conv1d48: Conv1d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}

impl<B: CustomKernelsBackend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack bytes");
        model
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    pub fn forward_sequence_16khz(
        &self,
        input: Tensor<B, 2>,
        state: Tensor<B, 3>,
        use_f16: bool,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        // Pad in f32 (Reflect mode doesn't support f16), then cast
        let pad7_out1 = input.pad(
            [(0usize, 0usize), (0usize, 64usize)],
            burn::tensor::ops::PadMode::Reflect,
        );
        let unsqueeze31_out1: Tensor<B, 3> = pad7_out1.unsqueeze_dims::<3>(&[1]);
        let unsqueeze31_out1 = if use_f16 {
            unsqueeze31_out1.cast(burn::tensor::FloatDType::F16)
        } else {
            unsqueeze31_out1
        };
        let conv1d37_out1 = self.conv1d37.forward(unsqueeze31_out1);
        let slice13_out1 = conv1d37_out1.clone().slice(s![.., 0..129, ..]);
        let slice14_out1 = conv1d37_out1.slice(s![.., 129.., ..]);
        let pow13_out1 = slice13_out1.clone() * slice13_out1;
        let pow14_out1 = slice14_out1.clone() * slice14_out1;
        let add19_out1 = pow13_out1.add(pow14_out1);
        let sqrt7_out1 = add19_out1.sqrt();
        let conv1d38_out1 = self.conv1d38.forward(sqrt7_out1);
        let relu31_out1 = burn::tensor::activation::relu(conv1d38_out1);
        let conv1d39_out1 = self.conv1d39.forward(relu31_out1);
        let relu32_out1 = burn::tensor::activation::relu(conv1d39_out1);
        let conv1d40_out1 = self.conv1d40.forward(relu32_out1);
        let relu33_out1 = burn::tensor::activation::relu(conv1d40_out1);
        let conv1d41_out1 = self.conv1d41.forward(relu33_out1);
        let relu34_out1 = burn::tensor::activation::relu(conv1d41_out1);
        let features = {
            let sliced = relu34_out1.slice(s![.., .., 0i64]);
            sliced.squeeze_dim::<2usize>(2)
        };

        let state = if use_f16 {
            state.cast(burn::tensor::FloatDType::F16)
        } else {
            state
        };
        let mut hidden = {
            let sliced = state.clone().slice(s![0i64, .., ..]);
            sliced.squeeze_dim::<2usize>(0)
        };
        let mut cell = {
            let sliced = state.slice(s![1i64, .., ..]);
            sliced.squeeze_dim::<2usize>(0)
        };

        let steps = features.dims()[0];

        // Pre-compute input gates for all steps at once (1 batched matmul
        // instead of N sequential ones).
        let input_gates_all = self.linear14.forward(features); // [steps, 512]

        // Extract weight and bias for the fused LSTM kernel
        let lstm_weight = self.linear13.weight.val();
        let lstm_bias = self
            .linear13
            .bias
            .as_ref()
            .expect("linear13 must have bias")
            .val();

        let mut hidden_states = Vec::with_capacity(steps);

        for step in 0..steps {
            let input_gates = input_gates_all
                .clone()
                .slice(s![step as i64..step as i64 + 1, ..]);

            let (new_hidden, new_cell) = lstm_cell_fused(
                hidden,
                cell,
                input_gates,
                lstm_weight.clone(),
                lstm_bias.clone(),
            );
            hidden = new_hidden;
            cell = new_cell;
            hidden_states.push(hidden.clone());
        }

        // Batch output head: process all hidden states at once instead of
        // per-step (eliminates N × (relu + conv1d + sigmoid) kernel launches).
        let all_hidden = burn::tensor::Tensor::cat(hidden_states, 0); // [steps, 128]
        let output = burn::tensor::activation::relu(all_hidden);
        let output: Tensor<B, 3> = output.unsqueeze_dims::<3>(&[-1]); // [steps, 128, 1]
        let output = self.conv1d42.forward(output); // [steps, 1, 1]
        let output = burn::tensor::activation::sigmoid(output);
        let output = output.reshape([steps, 1]); // [steps, 1]

        let hidden: Tensor<B, 3> = hidden.unsqueeze_dims::<3>(&[0]);
        let cell: Tensor<B, 3> = cell.unsqueeze_dims::<3>(&[0]);
        let state = burn::tensor::Tensor::cat([hidden, cell].into(), 0);

        (output, state)
    }
}

impl<B: CustomKernelsBackend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant32: burn::module::Param<Tensor<B, 1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::from_data([0i64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant41: burn::module::Param<Tensor<B, 1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::from_data([1i64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant42: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::from_data([2f64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let conv1d37 = Conv1dConfig::new(1, 258, 256)
            .with_stride(128)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d38 = Conv1dConfig::new(129, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d39 = Conv1dConfig::new(128, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d40 = Conv1dConfig::new(64, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d41 = Conv1dConfig::new(64, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear13 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear14 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d42 = Conv1dConfig::new(128, 1, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d43 = Conv1dConfig::new(1, 130, 128)
            .with_stride(64)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d44 = Conv1dConfig::new(65, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d45 = Conv1dConfig::new(128, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d46 = Conv1dConfig::new(64, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d47 = Conv1dConfig::new(64, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear15 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear16 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d48 = Conv1dConfig::new(128, 1, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            constant32,
            constant41,
            constant42,
            conv1d37,
            conv1d38,
            conv1d39,
            conv1d40,
            conv1d41,
            linear13,
            linear14,
            conv1d42,
            conv1d43,
            conv1d44,
            conv1d45,
            conv1d46,
            conv1d47,
            linear15,
            linear16,
            conv1d48,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        sr: i64,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let equal1_out1 = sr == 16000i64;
        let (if1_out1, if1_out2) = if equal1_out1 {
            let input = input.clone();
            let state = state.clone();
            let pad7_out1 = input.pad(
                [(0usize, 0usize), (0usize, 64usize)],
                burn::tensor::ops::PadMode::Reflect,
            );
            let unsqueeze31_out1: Tensor<B, 3> = pad7_out1.unsqueeze_dims::<3>(&[1]);
            let conv1d37_out1 = self.conv1d37.forward(unsqueeze31_out1);
            let slice13_out1 = conv1d37_out1.clone().slice(s![.., 0..129, ..]);
            let slice14_out1 = conv1d37_out1.slice(s![.., 129.., ..]);
            let pow13_out1 = slice13_out1.clone() * slice13_out1;
            let pow14_out1 = slice14_out1.clone() * slice14_out1;
            let add19_out1 = pow13_out1.add(pow14_out1);
            let sqrt7_out1 = add19_out1.sqrt();
            let conv1d38_out1 = self.conv1d38.forward(sqrt7_out1);
            let relu31_out1 = burn::tensor::activation::relu(conv1d38_out1);
            let conv1d39_out1 = self.conv1d39.forward(relu31_out1);
            let relu32_out1 = burn::tensor::activation::relu(conv1d39_out1);
            let conv1d40_out1 = self.conv1d40.forward(relu32_out1);
            let relu33_out1 = burn::tensor::activation::relu(conv1d40_out1);
            let conv1d41_out1 = self.conv1d41.forward(relu33_out1);
            let relu34_out1 = burn::tensor::activation::relu(conv1d41_out1);
            let gather21_out1 = {
                let sliced = relu34_out1.slice(s![.., .., 0i64]);
                sliced.squeeze_dim::<2usize>(2)
            };
            let gather22_out1 = {
                let sliced = state.clone().slice(s![0i64, .., ..]);
                sliced.squeeze_dim::<2usize>(0)
            };
            let gather23_out1 = {
                let sliced = state.slice(s![1i64, .., ..]);
                sliced.squeeze_dim::<2usize>(0)
            };
            let linear13_out1 = self.linear13.forward(gather22_out1);
            let linear14_out1 = self.linear14.forward(gather21_out1);
            let add20_out1 = linear13_out1.add(linear14_out1);
            let split_tensors = add20_out1.split_with_sizes([128, 128, 128, 128].into(), 1);
            let [split7_out1, split7_out2, split7_out3, split7_out4] =
                split_tensors.try_into().unwrap();
            let sigmoid25_out1 = burn::tensor::activation::sigmoid(split7_out1);
            let sigmoid26_out1 = burn::tensor::activation::sigmoid(split7_out2);
            let tanh13_out1 = split7_out3.tanh();
            let sigmoid27_out1 = burn::tensor::activation::sigmoid(split7_out4);
            let mul19_out1 = sigmoid26_out1.mul(gather23_out1);
            let mul20_out1 = sigmoid25_out1.mul(tanh13_out1);
            let add21_out1 = mul19_out1.add(mul20_out1);
            let tanh14_out1 = add21_out1.clone().tanh();
            let mul21_out1 = sigmoid27_out1.mul(tanh14_out1);
            let unsqueeze32_out1: Tensor<B, 3> = mul21_out1.clone().unsqueeze_dims::<3>(&[-1]);
            let unsqueeze33_out1: Tensor<B, 3> = mul21_out1.unsqueeze_dims::<3>(&[0]);
            let unsqueeze34_out1: Tensor<B, 3> = add21_out1.unsqueeze_dims::<3>(&[0]);
            let concat7_out1 =
                burn::tensor::Tensor::cat([unsqueeze33_out1, unsqueeze34_out1].into(), 0);
            let relu35_out1 = burn::tensor::activation::relu(unsqueeze32_out1);
            let conv1d42_out1 = self.conv1d42.forward(relu35_out1);
            let sigmoid28_out1 = burn::tensor::activation::sigmoid(conv1d42_out1);
            let squeeze7_out1 = sigmoid28_out1.squeeze_dims::<2>(&[1]);
            let reducemean7_out1 = { squeeze7_out1.mean_dim(1usize).squeeze_dims::<1usize>(&[1]) };
            let unsqueeze35_out1: Tensor<B, 2> = reducemean7_out1.unsqueeze_dims::<2>(&[1]);
            (unsqueeze35_out1, concat7_out1)
        } else {
            let input = input.clone();
            let state = state.clone();
            let pad8_out1 = input.pad(
                [(0usize, 0usize), (0usize, 32usize)],
                burn::tensor::ops::PadMode::Reflect,
            );
            let unsqueeze36_out1: Tensor<B, 3> = pad8_out1.unsqueeze_dims::<3>(&[1]);
            let conv1d43_out1 = self.conv1d43.forward(unsqueeze36_out1);
            let slice15_out1 = conv1d43_out1.clone().slice(s![.., 0..65, ..]);
            let slice16_out1 = conv1d43_out1.slice(s![.., 65.., ..]);
            let pow15_out1 = slice15_out1.clone() * slice15_out1;
            let pow16_out1 = slice16_out1.clone() * slice16_out1;
            let add22_out1 = pow15_out1.add(pow16_out1);
            let sqrt8_out1 = add22_out1.sqrt();
            let conv1d44_out1 = self.conv1d44.forward(sqrt8_out1);
            let relu36_out1 = burn::tensor::activation::relu(conv1d44_out1);
            let conv1d45_out1 = self.conv1d45.forward(relu36_out1);
            let relu37_out1 = burn::tensor::activation::relu(conv1d45_out1);
            let conv1d46_out1 = self.conv1d46.forward(relu37_out1);
            let relu38_out1 = burn::tensor::activation::relu(conv1d46_out1);
            let conv1d47_out1 = self.conv1d47.forward(relu38_out1);
            let relu39_out1 = burn::tensor::activation::relu(conv1d47_out1);
            let gather24_out1 = {
                let sliced = relu39_out1.slice(s![.., .., 0i64]);
                sliced.squeeze_dim::<2usize>(2)
            };
            let gather25_out1 = {
                let sliced = state.clone().slice(s![0i64, .., ..]);
                sliced.squeeze_dim::<2usize>(0)
            };
            let gather26_out1 = {
                let sliced = state.slice(s![1i64, .., ..]);
                sliced.squeeze_dim::<2usize>(0)
            };
            let linear15_out1 = self.linear15.forward(gather25_out1);
            let linear16_out1 = self.linear16.forward(gather24_out1);
            let add23_out1 = linear15_out1.add(linear16_out1);
            let split_tensors = add23_out1.split_with_sizes([128, 128, 128, 128].into(), 1);
            let [split8_out1, split8_out2, split8_out3, split8_out4] =
                split_tensors.try_into().unwrap();
            let sigmoid29_out1 = burn::tensor::activation::sigmoid(split8_out1);
            let sigmoid30_out1 = burn::tensor::activation::sigmoid(split8_out2);
            let tanh15_out1 = split8_out3.tanh();
            let sigmoid31_out1 = burn::tensor::activation::sigmoid(split8_out4);
            let mul22_out1 = sigmoid30_out1.mul(gather26_out1);
            let mul23_out1 = sigmoid29_out1.mul(tanh15_out1);
            let add24_out1 = mul22_out1.add(mul23_out1);
            let tanh16_out1 = add24_out1.clone().tanh();
            let mul24_out1 = sigmoid31_out1.mul(tanh16_out1);
            let unsqueeze37_out1: Tensor<B, 3> = mul24_out1.clone().unsqueeze_dims::<3>(&[-1]);
            let unsqueeze38_out1: Tensor<B, 3> = mul24_out1.unsqueeze_dims::<3>(&[0]);
            let unsqueeze39_out1: Tensor<B, 3> = add24_out1.unsqueeze_dims::<3>(&[0]);
            let concat8_out1 =
                burn::tensor::Tensor::cat([unsqueeze38_out1, unsqueeze39_out1].into(), 0);
            let relu40_out1 = burn::tensor::activation::relu(unsqueeze37_out1);
            let conv1d48_out1 = self.conv1d48.forward(relu40_out1);
            let sigmoid32_out1 = burn::tensor::activation::sigmoid(conv1d48_out1);
            let squeeze8_out1 = sigmoid32_out1.squeeze_dims::<2>(&[1]);
            let reducemean8_out1 = { squeeze8_out1.mean_dim(1usize).squeeze_dims::<1usize>(&[1]) };
            let unsqueeze40_out1: Tensor<B, 2> = reducemean8_out1.unsqueeze_dims::<2>(&[1]);
            (unsqueeze40_out1, concat8_out1)
        };
        (if1_out1, if1_out2)
    }
}
