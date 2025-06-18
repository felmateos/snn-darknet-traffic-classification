from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

@dataclass
class SchedulerParams:
    mode: List[str]
    factor: List[float]
    patience: List[int]
    threshold: List[float]
    threshold_mode: List[str]
    cooldown: List[int]
    min_lr: List[float]
    eps: float
    T_max: List[int]

@dataclass
class TrainParams:
    w: List[bool]
    v: List[bool]
    b: List[bool]
    beta: List[bool]
    surrogate_sigma: List[bool]
    BN_scale: List[bool]
    BN_offset: List[bool]
    Readout_b_latency: List[bool]
    Readout_latency_scale: List[bool]
    BN_intermediate_mode: List[str]
    surrogate_sigma_mode: List[str]
    layer_beta_mode: List[str]
    train_surrogate_mode: List[str]
    test_surrogate_mode: List[str]
    Readout_latency_output_mode: List[str]
    Readout_max_latency_output_mode: List[str]
    Readout_latency_scale_val: List[float]
    Readout_steps_start: List[int]
    Readout_steps_end: List[int]
    beta_init_method: List[str]
    beta_constant_val: List[float]
    beta_normal_mean: List[float]
    beta_normal_std: List[float]
    beta_uniform_start: List[float]
    beta_uniform_end: List[float]
    b_init_method: List[str]
    b_constant_val: List[float]
    b_normal_mean: List[float]
    b_normal_std: List[float]
    b_uniform_start: List[float]
    b_uniform_end: List[float]

@dataclass
class DatasetParams:
    nb_bins_time: int
    nb_bins_size: int
    gaps: List[List]
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
class UserParamsTrain:
    nb_epochs: List[int]
    warmup_epochs: List[int]
    snapshot_epochs: List[List]
    lateral_connections: List[bool]
    # Leaky_Neuron: bool
    # spike_limit: bool
    # spike_limit_mode: str
    Regularization_Term: List[List]
    High_speed_mode: List[bool]
    # Dropout_mode: str
    surrogate_backward_mode: List[str]
    Readout_time_reduction: List[str]
    warmup_Readout_mode: List[str]
    optimizer: List[str]
    scheduler: List[str]
    Project: str
    Problem: str
    Environment: str
    Dataset: List[str]
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
    lr: List[float]
    batch_size: List[int]
    Batch_num_limit: int
    # nb_inputs: int
    image_W: int
    image_H: int
    Input_channels: int
    nb_dense_layer: List[int]
    nb_outputs: List[int]
    w_init_mean: List[float]
    w_init_std: List[float]
    surrogate_sigma_sigmoid: float
    surrogate_sigma_rec: float
    surrogate_a_fast_sigmoid_abs: float
    surrogate_a_fast_sigmoid_tanh: float
    surrogate_sigma_scale: List[float]
    nb_steps: List[int]
    weight_decay: List[float]
    betas: List[List]
    time_step: List[float]
    gamma: List[float]
    eps: float
    Max_spikes_per_run: List[int]
    Scheduler_params: SchedulerParams
    Train_params: TrainParams
    Dataset_params: DatasetParams
    spike_params: SpikeParams
    device: Dict
    find_params: FindParams
    Results_path: str
