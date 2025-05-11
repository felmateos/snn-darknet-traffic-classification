from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

@dataclass
class SchedulerParams:
    mode: str
    factor: float
    patience: int
    threshold: float
    threshold_mode: str
    cooldown: int
    min_lr: float
    eps: float
    T_max: int

@dataclass
class TrainParams:
    w: bool
    v: bool
    b: bool
    beta: bool
    surrogate_sigma: bool
    BN_scale: bool
    BN_offset: bool
    Readout_b_latency: bool
    Readout_latency_scale: bool
    BN_intermediate_mode: str
    surrogate_sigma_mode: str
    layer_beta_mode: str
    train_surrogate_mode: str
    test_surrogate_mode: str
    Readout_latency_output_mode: str
    Readout_max_latency_output_mode: str
    Readout_latency_scale_val: float
    Readout_steps_start: int
    Readout_steps_end: int
    beta_init_method: str
    beta_constant_val: float
    beta_normal_mean: float
    beta_normal_std: float
    beta_uniform_start: float
    beta_uniform_end: float
    b_init_method: str
    b_constant_val: float
    b_normal_mean: float
    b_normal_std: float
    b_uniform_start: float
    b_uniform_end: float

@dataclass
class DatasetParams:
    nb_bins_time: int
    nb_bins_size: int
    gaps: List[int]
    min_value: int
    plot_data_mode: str
    plot_data_type: str

@dataclass
class SpikeParams:
    nb_steps: int
    time_step: float
    spike_limit: bool
    spike_limit_mode: str
    Max_spikes_per_run: int
    Leaky_Neuron: bool

@dataclass
class FindParams:
    main_params: List[str]
    dont_care_params: List[str]
    main_mismatch_penalty: int
    non_main_mismatch_penalty: int
    main_no_exist_penalty: int
    non_main_no_exist_penalty: int
    Find_mode: str
    Find_print_details: bool
    Find_path: str

@dataclass
class UserParams:
    nb_epochs: int
    warmup_epochs: int
    snapshot_epochs: List[int]
    lateral_connections: bool
    # Leaky_Neuron: bool
    # spike_limit: bool
    # spike_limit_mode: str
    Regularization_Term: List[str]
    High_speed_mode: bool
    # Dropout_mode: str
    surrogate_backward_mode: str
    Readout_time_reduction: str
    warmup_Readout_mode: str
    optimizer: str
    scheduler: str
    Project: str
    Problem: str
    Environment: str
    Dataset: str
    Model_features: str
    User_comment: str
    Model_load: bool
    load_path: str
    Train_mode: bool
    Validation_mode: bool
    Test_mode: bool
    Results_save: bool
    Train_Profiler: bool
    Test_Profiler: bool
    Plot_graphs: bool
    plot_show: bool
    Find_model: bool
    reg_loss_coef: int
    spk_spread_loss_coef: int
    lr: float
    batch_size: int
    Batch_num_limit: int
    # nb_inputs: int
    image_W: int
    image_H: int
    Input_channels: int
    nb_dense_layer: int
    nb_outputs: int
    w_init_mean: float
    w_init_std: float
    surrogate_sigma_sigmoid: float
    surrogate_sigma_rec: float
    surrogate_a_fast_sigmoid_abs: float
    surrogate_a_fast_sigmoid_tanh: float
    surrogate_sigma_scale: float
    nb_steps: int
    weight_decay: float
    betas: List[float]
    time_step: float
    gamma: float
    eps: float
    Max_spikes_per_run: int
    Scheduler_params: SchedulerParams
    Train_params: TrainParams
    Dataset_params: DatasetParams
    spike_params: SpikeParams
    device: Dict
    find_params: FindParams
    Results_path: str
