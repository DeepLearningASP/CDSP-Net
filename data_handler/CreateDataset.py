"""
Synthetic array-data and correlation-matrix generation utilities.
"""
import math
from typing import Sequence, Union
import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset, random_split
from tqdm import tqdm


Number = Union[int, float]
ScalarOrSequence = Union[Number, Sequence[Number], torch.Tensor]


def _first_value(value: ScalarOrSequence) -> Number:
    """Return the first value from a scalar or sequence-like input."""
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _expand_to_batch(
    value: ScalarOrSequence,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand a scalar parameter to a batch vector."""
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        return tensor.expand(batch_size)
    if tensor.numel() == batch_size:
        return tensor
    raise ValueError(f"Expected a scalar or {batch_size} values, but received {tensor.numel()}.")


def _build_multihot_labels(
    doa_batch_rad: torch.Tensor,
    deg_l: ScalarOrSequence,
    deg_p: ScalarOrSequence,
    n_grids: int,
) -> torch.Tensor:
    """Map DOAs in radians to multi-hot grid labels."""
    device = doa_batch_rad.device
    deg_l_tensor = torch.as_tensor(deg_l, device=device, dtype=torch.float32)
    deg_p_tensor = torch.as_tensor(deg_p, device=device, dtype=torch.float32)

    doa_batch_deg = torch.rad2deg(doa_batch_rad)
    indices = torch.round((doa_batch_deg - deg_l_tensor) / deg_p_tensor).long()
    indices = torch.clamp(indices, 0, n_grids - 1)

    batch_size = doa_batch_rad.shape[0]
    labels = torch.zeros(
        batch_size,
        n_grids,
        device=device,
        dtype=torch.float32,
    )
    sample_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    labels[sample_indices, indices] = 1.0
    return labels


class CorrMtrGenerator:
    """Generate received array signals and lagged correlation matrices."""

    LIGHT_SPEED = 3e8
    NO_COUPLING_SENTINEL = -999

    def __init__(self, N, M, tau, array_pos, d_init, lamda, eta,
                 logger, coh_type, band_type, device):
        self.N = N
        self.M = M
        self.tau = tau
        self.array_pos = array_pos
        self.d_init = d_init
        self.coh_type = coh_type
        self.lamda = lamda
        self.fc = self.LIGHT_SPEED / lamda
        self.device = torch.device(device)
        self.eta = eta
        self.logger = logger
        self.band_type = band_type
        self.coup_noise_std = 0.05

    def generate_array_pos(
        self,
        batch_size,
        eta_batch,
    ) -> torch.Tensor:
        """Generate perturbed sensor positions for each sample."""
        eta_batch = eta_batch.reshape(-1, 1)
        pos_errors = (
            (
                2
                * torch.rand(
                    batch_size,
                    self.N,
                    device=self.device,
                )
                - 1
            )
            * eta_batch
        )
        return self.array_pos.unsqueeze(0) + pos_errors

    def generate_coup_matrix(
        self,
        d_ary,
        coup_batch,
    ) -> torch.Tensor:
        """Generate distance-dependent mutual-coupling matrices."""
        d_ary_expanded = d_ary.unsqueeze(-1)
        delta_d = torch.abs(d_ary_expanded - d_ary_expanded.transpose(-1, -2))

        coup_batch_1d = torch.atleast_1d(coup_batch).to(
            device=d_ary.device,
            dtype=torch.float32,
        )

        no_coupling = coup_batch_1d == self.NO_COUPLING_SENTINEL
        if no_coupling.all().item():
            batch_size, num_sensors = d_ary.shape
            identity = torch.eye(
                num_sensors,
                device=d_ary.device,
                dtype=torch.complex64,
            )
            return identity.unsqueeze(0).expand(batch_size, -1, -1)

        coup_batch_expanded = coup_batch_1d.masked_fill(no_coupling, 0.0)[:, None, None]
        log_10 = torch.log(
            torch.tensor(
                10.0,
                device=d_ary.device,
                dtype=torch.float32,
            )
        )
        coup_alpha = -(coup_batch_expanded / 20.0) * log_10

        amplitude = torch.exp(-2 * coup_alpha * delta_d / self.lamda)
        phase = torch.exp(-1j * (2 * torch.pi / self.lamda) * delta_d)
        deterministic_coupling = amplitude * phase

        noise_shape = (
            d_ary.shape[0],
            d_ary.shape[1],
            d_ary.shape[1],
        )
        real_noise = torch.randn(
            noise_shape,
            device=d_ary.device,
            dtype=torch.float32,
        )
        imag_noise = torch.randn(
            noise_shape,
            device=d_ary.device,
            dtype=torch.float32,
        )
        complex_noise = (real_noise + 1j * imag_noise) * (self.coup_noise_std / math.sqrt(2))

        coupling = deterministic_coupling + complex_noise
        if no_coupling.any().item():
            identity = torch.eye(
                d_ary.shape[1],
                device=d_ary.device,
                dtype=coupling.dtype,
            )
            coupling = torch.where(no_coupling[:, None, None], identity, coupling)
        return coupling

    def generate_gaussian_sig_batch(
        self,
        batch_size,
        T_batch,
    ) -> torch.Tensor:
        """Generate narrowband Gaussian source signals."""
        T_max = int(T_batch.max().item())
        signal_shape = (batch_size, self.M, T_max)

        signal_real = torch.randn(
            signal_shape,
            device=self.device,
        ) / math.sqrt(2)
        signal_imag = torch.randn(
            signal_shape,
            device=self.device,
        ) / math.sqrt(2)

        if self.coh_type == "NonCoherent":
            signal_complex = torch.complex(
                signal_real,
                signal_imag,
            )
        elif self.coh_type in {"Coherent", "Mixed"}:
            base_signal = torch.complex(
                signal_real[:, :1, :],
                signal_imag[:, :1, :],
            )
            gamma = 0.5 + 0.5 * torch.rand(
                batch_size,
                self.M,
                1,
                device=self.device,
            )
            phi = (
                2
                * torch.pi
                * torch.rand(
                    batch_size,
                    self.M,
                    1,
                    device=self.device,
                )
            )
            gamma[:, 0, :] = 1.0
            phi[:, 0, :] = 0.0
            coefficients = torch.polar(gamma, phi)
            signal_complex = base_signal * coefficients
            if self.coh_type == "Mixed":
                # Equal-probability mixture of the original source models.
                # Each realization is either noncoherent or fully coherent.
                coherent_mask = torch.rand(batch_size, 1, 1, device=self.device) < 0.5
                independent_signal = torch.complex(signal_real, signal_imag)
                signal_complex = torch.where(coherent_mask, signal_complex, independent_signal)
        else:
            raise ValueError(f"Unsupported coherence type: {self.coh_type}")

        valid_mask = torch.arange(T_max, device=self.device)[None, None, :] < T_batch[:, None, None]
        signal_complex = signal_complex * valid_mask

        # Keep the original real/imaginary concatenated representation.
        return torch.cat(
            [signal_complex.real, signal_complex.imag],
            dim=1,
        )

    def generate_array_res_batch(
        self,
        sig_batch,
        doa_batch,
        snr_batch,
        coup_batch,
        T_batch,
        eta_batch,
    ):
        """Generate noisy array observations."""
        batch_size = sig_batch.shape[0]
        T_max = sig_batch.shape[-1]

        d_ary = self.generate_array_pos(
            batch_size,
            eta_batch,
        )
        coup_matrix = self.generate_coup_matrix(
            d_ary,
            coup_batch,
        )

        antenna_pos = d_ary.unsqueeze(-1)
        sin_theta = torch.sin(doa_batch).unsqueeze(1)

        if self.band_type == "Narrow":
            phase = -2 * torch.pi * torch.matmul(antenna_pos, sin_theta) / self.lamda
            steering_matrix = torch.polar(
                torch.ones_like(phase),
                phase,
            )
            array_response = torch.matmul(
                steering_matrix,
                sig_batch,
            )

        elif self.band_type == "Broad":
            delay_matrix = (antenna_pos * sin_theta) / self.LIGHT_SPEED
            signal_freq = torch.fft.fft(
                sig_batch,
                dim=-1,
            )
            frequencies = torch.fft.fftfreq(
                T_max,
                1 / self.fs,
                device=self.device,
            )
            phase = (
                -1j
                * 2
                * torch.pi
                * (self.fc + frequencies.view(1, 1, 1, -1))
                * delay_matrix.unsqueeze(-1)
            )
            received_freq = torch.sum(
                signal_freq.unsqueeze(1) * torch.exp(phase),
                dim=2,
            )
            array_response = torch.fft.ifft(
                received_freq,
                dim=-1,
            )
        else:
            raise ValueError(f"Unsupported band type: {self.band_type}")

        coupled_response = torch.matmul(
            coup_matrix,
            array_response,
        )

        valid_mask = torch.arange(T_max, device=self.device)[None, None, :] < T_batch[:, None, None]
        response_power = (torch.abs(coupled_response).square() * valid_mask).sum(dim=(1, 2))
        response_power = response_power / (self.N * T_batch.to(torch.float32))
        response_power = response_power.view(-1, 1, 1)

        snr_linear = torch.pow(
            10.0,
            snr_batch.view(-1, 1, 1) / 10.0,
        )
        noise_power = response_power / snr_linear
        noise_std = torch.sqrt(noise_power / 2.0)
        noise = torch.complex(
            noise_std * torch.randn_like(coupled_response.real),
            noise_std * torch.randn_like(coupled_response.real),
        )
        noise = noise * valid_mask

        return coupled_response + noise

    def generate_corr_matrix_batch(
        self,
        sig_n_batch,
        T_batch,
    ) -> torch.Tensor:
        """Compute lagged sample correlation matrices."""
        batch_size, _, max_T = sig_n_batch.shape

        time_indices = torch.arange(
            max_T,
            device=self.device,
        ).view(1, 1, max_T)
        valid_mask = time_indices < T_batch.view(-1, 1, 1)
        centered_signal = sig_n_batch * valid_mask

        corr_matrices = torch.zeros(
            (
                batch_size,
                self.tau,
                2 * self.N,
                self.N,
            ),
            device=self.device,
            dtype=torch.float32,
        )

        for lag in range(self.tau):
            if max_T - lag <= 0:
                break

            effective_length = torch.clamp(
                (T_batch - lag).to(torch.float32),
                min=1e-8,
            )
            signal_left = centered_signal[:, :, : max_T - lag]
            signal_right = centered_signal[:, :, lag:]

            corr_sum = torch.matmul(
                signal_left,
                signal_right.transpose(1, 2).conj(),
            )
            corr_current = corr_sum / effective_length.view(-1, 1, 1)

            corr_matrices[:, lag, : self.N, :] = corr_current.real
            corr_matrices[:, lag, self.N :, :] = corr_current.imag

        return corr_matrices

    def creat_corr_mtr_batch(
        self,
        batch_size,
        doa_batch,
        snr_batch,
        coup_batch,
        T_batch,
        eta_batch,
    ):
        """Generate correlation matrices and received signals.

        The original method name is retained for backward compatibility.
        """
        if self.band_type == "Narrow":
            signal_batch = self.generate_gaussian_sig_batch(
                batch_size,
                T_batch,
            )
            signal_complex = torch.complex(
                signal_batch[:, : self.M, :],
                signal_batch[:, self.M:, :],
            )
        else:
            signal_complex = self.generate_ofdm_sig_batch(
                batch_size,
                T_batch,
            )

        received_signal = self.generate_array_res_batch(
            signal_complex,
            doa_batch,
            snr_batch,
            coup_batch,
            T_batch,
            eta_batch,
        )
        corr_matrix = self.generate_corr_matrix_batch(
            received_signal,
            T_batch,
        )
        return corr_matrix, received_signal


class DOAGenerator:
    """Generate random, fixed, or fixed-separation DOA samples."""

    def __init__(self, M, Gap, deta, fix, angle_type, device):
        self.M = M
        self.angle_type = angle_type
        self.device = torch.device(device)
        self.gap = torch.tensor(
            Gap,
            dtype=torch.float32,
            device=self.device,
        )
        self.max_diff = torch.tensor(
            120 - Gap,
            dtype=torch.float32,
            device=self.device,
        )
        self.D2R = torch.tensor(
            np.pi / 180.0,
            dtype=torch.float32,
            device=self.device,
        )

        self.deta = (
            deta.clone()
            .detach()
            .to(
                dtype=torch.float32,
                device=self.device,
            )
            if self.angle_type == "deta_angle"
            else None
        )
        self.fix = (
            torch.tensor(
                fix,
                dtype=torch.float32,
                device=self.device,
            )
            if self.angle_type == "fix_angle"
            else None
        )

    def generate_doa_batch(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """Generate a batch of DOAs in radians."""
        if self.angle_type == "rand_angle":
            if self.M == 1:
                doa_deg = (
                    torch.rand(
                        batch_size,
                        self.M,
                        device=self.device,
                    )
                    * 120
                    - 60
                )
                return doa_deg * self.D2R

            doa_final = []
            current_count = 0
            while current_count < batch_size:
                doa_candidate = (
                    torch.rand(
                        batch_size,
                        self.M,
                        device=self.device,
                    )
                    * 120
                    - 60
                )
                doa_sorted, _ = torch.sort(
                    doa_candidate,
                    dim=1,
                )
                angle_differences = torch.diff(
                    doa_sorted,
                    dim=1,
                )
                valid_mask = torch.all(
                    (angle_differences > self.gap) & (angle_differences < self.max_diff),
                    dim=1,
                )
                valid_samples = doa_sorted[valid_mask]
                if valid_samples.shape[0] > 0:
                    doa_final.append(valid_samples)
                    current_count += valid_samples.shape[0]

            return torch.cat(doa_final, dim=0)[:batch_size] * self.D2R

        if self.angle_type == "deta_angle":
            doa_first = (
                torch.rand(
                    batch_size,
                    1,
                    device=self.device,
                )
                * (120 - self.deta)
                - 60
            )
            doa_second = doa_first + self.deta
            doa_batch = torch.cat(
                [doa_first, doa_second],
                dim=1,
            )
            doa_batch, _ = torch.sort(
                doa_batch,
                dim=1,
            )
            return doa_batch * self.D2R

        if self.angle_type == "fix_angle":
            doa_batch = self.fix.unsqueeze(0).expand(
                batch_size,
                -1,
            )
            return doa_batch * self.D2R

        raise ValueError(f"Unsupported angle type: {self.angle_type}")


def create_train_dataset(num_sample, N, M, T_list, snr, tau, lamda, coup_level, eta, logger, Gap, array_pos,
                         d_init, coh_type, band_type, datasets_ratio, device, deg_l, deg_u, deg_p):
    """Create training and validation TensorDataset subsets."""
    device = torch.device(device)

    snr_values = torch.tensor(
        snr,
        device=device,
        dtype=torch.float32,
    )
    coup_values = torch.tensor(
        coup_level,
        device=device,
        dtype=torch.float32,
    )
    snapshot_values = torch.tensor(
        T_list,
        device=device,
        dtype=torch.long,
    )
    eta_values = torch.tensor(
        eta,
        device=device,
        dtype=torch.float32,
    )

    global_T_max = int(snapshot_values.max().item())
    split_ratio = np.asarray(
        datasets_ratio,
        dtype=np.float32,
    )

    doa_generator = DOAGenerator(M, Gap, None, None, "rand_angle", device)
    corr_generator = CorrMtrGenerator(N, M, tau, array_pos, d_init, lamda,
                                      eta, logger, coh_type, band_type, device)

    n_grids = int(round((deg_u - deg_l) / deg_p)) + 1
    batch_size = 500
    num_batches = math.ceil(num_sample / batch_size)

    corr_all = torch.empty(
        num_sample,
        tau,
        2 * N,
        N,
        device=device,
        dtype=torch.float32,
    )
    sig_n_all = torch.zeros(
        num_sample,
        N,
        global_T_max,
        device=device,
        dtype=torch.complex64,
    )
    doa_all = torch.empty(
        num_sample,
        M,
        device=device,
        dtype=torch.float32,
    )
    label_all = torch.empty(
        num_sample,
        n_grids,
        device=device,
        dtype=torch.float32,
    )

    start_index = 0
    for _ in tqdm(
        range(num_batches),
        desc="Create train datasets",
    ):
        current_batch_size = min(
            batch_size,
            num_sample - start_index,
        )

        doa_batch = doa_generator.generate_doa_batch(current_batch_size)
        snapshot_batch = snapshot_values[
            torch.randint(
                0,
                len(snapshot_values),
                (current_batch_size,),
                device=device,
            )
        ]
        snr_batch = snr_values[
            torch.randint(
                0,
                len(snr_values),
                (current_batch_size,),
                device=device,
            )
        ]
        coup_batch = coup_values[
            torch.randint(
                0,
                len(coup_values),
                (current_batch_size,),
                device=device,
            )
        ]
        eta_batch = eta_values[
            torch.randint(
                0,
                len(eta_values),
                (current_batch_size,),
                device=device,
            )
        ]

        corr_batch, sig_n_batch = corr_generator.creat_corr_mtr_batch(
            current_batch_size,
            doa_batch,
            snr_batch,
            coup_batch,
            snapshot_batch,
            eta_batch,
        )
        label_batch = _build_multihot_labels(
            doa_batch,
            deg_l,
            deg_p,
            n_grids,
        )

        end_index = start_index + current_batch_size
        batch_T_max = sig_n_batch.shape[-1]
        corr_all[start_index:end_index] = corr_batch
        sig_n_all[
            start_index:end_index,
            :,
            :batch_T_max,
        ] = sig_n_batch
        doa_all[start_index:end_index] = doa_batch
        label_all[start_index:end_index] = label_batch

        start_index = end_index

    full_dataset = TensorDataset(
        corr_all,
        sig_n_all,
        doa_all,
        label_all,
    )

    train_size = int(num_sample * split_ratio[0])
    val_size = num_sample - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
    )
    logger.info(
        "Train dataset size: %d, validation dataset size: %d",
        train_size,
        val_size,
    )
    return train_dataset, val_dataset


def create_test_dataset(num_sample, N, M, Ts, snrs, params, tau, lamda, coup_levels, etas, logger, Gap,
                        deta, fix, array_pos, d_init, coh_type, band_type, test_type, angle_type, device, 
                        deg_l, deg_u, deg_p):
    """Create test data for the configured evaluation protocol."""
    device = torch.device(device)
    batch_size = 1000
    num_batches = math.ceil(num_sample / batch_size)
    num_sources = M[0] if isinstance(M, (list, tuple)) else M

    if isinstance(deg_p, (list, tuple)):
        if test_type != "Grid":
            raise ValueError(
                "A sequence of grid intervals is only supported " "when test_type='Grid'."
            )
        n_grids = None
    else:
        n_grids = int(round((deg_u - deg_l) / deg_p)) + 1

    coup_default = torch.as_tensor(
        coup_levels,
        device=device,
        dtype=torch.float32,
    )
    eta_default = torch.as_tensor(
        etas,
        device=device,
        dtype=torch.float32,
    )

    def make_corr_generator(
        current_M,
        current_eta,
    ):
        return CorrMtrGenerator(N, current_M, tau, array_pos, d_init, lamda,
                                current_eta, logger, coh_type, band_type, device)

    def fill_common_batch(
        corr_storage,
        signal_storage,
        doa_storage,
        label_storage,
        storage_index,
        start_index,
        corr_batch,
        signal_batch,
        doa_batch,
        label_batch,
    ):
        end_index = start_index + doa_batch.shape[0]
        corr_storage[storage_index, start_index:end_index] = corr_batch
        signal_storage[storage_index, start_index:end_index] = signal_batch
        doa_storage[storage_index, start_index:end_index] = doa_batch
        label_storage[storage_index, start_index:end_index] = label_batch
        return end_index

    # 1. Number of sources
    if test_type == "Src_Num":
        source_counts = M if isinstance(M, (list, tuple)) else [M]
        snapshot_value = int(_first_value(Ts))
        snr_value = float(_first_value(snrs))

        corr_all = []
        sig_n_all = []
        doa_all = []
        label_all = []

        for current_M in tqdm(
            source_counts,
            desc=f"{angle_type}_{test_type}",
        ):
            doa_generator = DOAGenerator(
                current_M,
                Gap,
                deta,
                fix,
                angle_type,
                device,
            )
            corr_generator = make_corr_generator(
                current_M,
                etas,
            )

            corr_current = torch.empty(
                num_sample,
                tau,
                2 * N,
                N,
                device=device,
                dtype=torch.float32,
            )
            signal_current = torch.empty(
                num_sample,
                N,
                snapshot_value,
                device=device,
                dtype=torch.complex64,
            )
            doa_current = torch.empty(
                num_sample,
                current_M,
                device=device,
                dtype=torch.float32,
            )
            label_current = torch.empty(
                num_sample,
                n_grids,
                device=device,
                dtype=torch.float32,
            )

            start_index = 0
            for _ in range(num_batches):
                current_batch_size = min(
                    batch_size,
                    num_sample - start_index,
                )
                snapshot_batch = torch.full(
                    (current_batch_size,),
                    snapshot_value,
                    device=device,
                    dtype=torch.long,
                )
                snr_batch = torch.full(
                    (current_batch_size,),
                    snr_value,
                    device=device,
                    dtype=torch.float32,
                )
                eta_batch = _expand_to_batch(
                    etas,
                    current_batch_size,
                    device,
                    torch.float32,
                )

                doa_batch = doa_generator.generate_doa_batch(current_batch_size)
                corr_batch, signal_batch = corr_generator.creat_corr_mtr_batch(
                    current_batch_size,
                    doa_batch,
                    snr_batch,
                    coup_default,
                    snapshot_batch,
                    eta_batch,
                )
                label_batch = _build_multihot_labels(
                    doa_batch,
                    deg_l,
                    deg_p,
                    n_grids,
                )

                end_index = start_index + current_batch_size
                corr_current[start_index:end_index] = corr_batch
                signal_current[start_index:end_index] = signal_batch
                doa_current[start_index:end_index] = doa_batch
                label_current[start_index:end_index] = label_batch
                start_index = end_index

            corr_all.append(corr_current)
            sig_n_all.append(signal_current)
            doa_all.append(doa_current)
            label_all.append(label_current)

        return corr_all, sig_n_all, doa_all, label_all

    # 2. SNR sweep
    if test_type in (
        "SNR",
        "Ablation_SNR",
        "Array_type_SNR",
    ):
        doa_generator = DOAGenerator(
            num_sources,
            Gap,
            deta,
            fix,
            angle_type,
            device,
        )
        snr_values = torch.as_tensor(
            snrs,
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
        snapshot_value = int(_first_value(Ts))
        num_snr = len(snr_values)

        corr_all = torch.empty(
            num_snr,
            num_sample,
            tau,
            2 * N,
            N,
            device=device,
            dtype=torch.float32,
        )
        sig_n_all = torch.empty(
            num_snr,
            num_sample,
            N,
            snapshot_value,
            device=device,
            dtype=torch.complex64,
        )
        doa_all = torch.empty(
            num_snr,
            num_sample,
            num_sources,
            device=device,
            dtype=torch.float32,
        )
        label_all = torch.empty(
            num_snr,
            num_sample,
            n_grids,
            device=device,
            dtype=torch.float32,
        )

        for snr_index, snr_value in enumerate(
            tqdm(
                snr_values,
                desc=f"{angle_type}_{test_type}",
            )
        ):
            corr_generator = make_corr_generator(
                num_sources,
                etas,
            )
            start_index = 0

            for _ in range(num_batches):
                current_batch_size = min(
                    batch_size,
                    num_sample - start_index,
                )
                snr_batch = torch.full(
                    (current_batch_size,),
                    snr_value.item(),
                    device=device,
                    dtype=torch.float32,
                )
                snapshot_batch = torch.full(
                    (current_batch_size,),
                    snapshot_value,
                    device=device,
                    dtype=torch.long,
                )
                eta_batch = _expand_to_batch(
                    etas,
                    current_batch_size,
                    device,
                    torch.float32,
                )

                doa_batch = doa_generator.generate_doa_batch(current_batch_size)
                corr_batch, signal_batch = corr_generator.creat_corr_mtr_batch(
                    current_batch_size,
                    doa_batch,
                    snr_batch,
                    coup_default,
                    snapshot_batch,
                    eta_batch,
                )
                label_batch = _build_multihot_labels(
                    doa_batch,
                    deg_l,
                    deg_p,
                    n_grids,
                )

                start_index = fill_common_batch(
                    corr_all,
                    sig_n_all,
                    doa_all,
                    label_all,
                    snr_index,
                    start_index,
                    corr_batch,
                    signal_batch,
                    doa_batch,
                    label_batch,
                )

        return corr_all, sig_n_all, doa_all, label_all

    # 3. Snapshot sweep
    if test_type in (
        "Snapshot",
        "Ablation_T",
        "Array_type_T",
    ):
        doa_generator = DOAGenerator(
            num_sources,
            Gap,
            deta,
            fix,
            angle_type,
            device,
        )
        snapshot_values = list(Ts)
        num_snapshots = len(snapshot_values)

        corr_all = torch.empty(
            num_snapshots,
            num_sample,
            tau,
            2 * N,
            N,
            device=device,
            dtype=torch.float32,
        )
        sig_n_all = [
            torch.empty(
                num_sample,
                N,
                current_T,
                device=device,
                dtype=torch.complex64,
            )
            for current_T in snapshot_values
        ]
        doa_all = torch.empty(
            num_snapshots,
            num_sample,
            num_sources,
            device=device,
            dtype=torch.float32,
        )
        label_all = torch.empty(
            num_snapshots,
            num_sample,
            n_grids,
            device=device,
            dtype=torch.float32,
        )

        for snapshot_index, snapshot_value in enumerate(
            tqdm(
                snapshot_values,
                desc=f"{angle_type}_{test_type}",
            )
        ):
            corr_generator = make_corr_generator(
                num_sources,
                etas,
            )
            start_index = 0

            for _ in range(num_batches):
                current_batch_size = min(
                    batch_size,
                    num_sample - start_index,
                )
                doa_batch = doa_generator.generate_doa_batch(current_batch_size)
                snapshot_batch = torch.full(
                    (current_batch_size,),
                    snapshot_value,
                    device=device,
                    dtype=torch.long,
                )
                snr_batch = _expand_to_batch(
                    snrs,
                    current_batch_size,
                    device,
                    torch.float32,
                )
                eta_batch = _expand_to_batch(
                    etas,
                    current_batch_size,
                    device,
                    torch.float32,
                )

                corr_batch, signal_batch = corr_generator.creat_corr_mtr_batch(
                    current_batch_size,
                    doa_batch,
                    snr_batch,
                    coup_default,
                    snapshot_batch,
                    eta_batch,
                )
                label_batch = _build_multihot_labels(
                    doa_batch,
                    deg_l,
                    deg_p,
                    n_grids,
                )

                end_index = start_index + current_batch_size
                corr_all[
                    snapshot_index,
                    start_index:end_index,
                ] = corr_batch
                sig_n_all[snapshot_index][start_index:end_index] = signal_batch
                doa_all[
                    snapshot_index,
                    start_index:end_index,
                ] = doa_batch
                label_all[
                    snapshot_index,
                    start_index:end_index,
                ] = label_batch
                start_index = end_index

        return corr_all, sig_n_all, doa_all, label_all

    # 4. Other evaluation protocols
    supported_other_types = {
        "Srcs_Dist",
        "Temp",
        "Grid",
        "Coup_Level",
        "Pos_Err",
        "CDF",
        "Complexity",
        "Weight",
    }
    if test_type not in supported_other_types:
        raise ValueError(f"Unsupported test type: {test_type}")

    snapshot_value = int(_first_value(Ts))
    if test_type == "Srcs_Dist":
        loop_values = torch.as_tensor(
            deta,
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
    elif test_type == "Coup_Level":
        loop_values = torch.as_tensor(
            coup_levels,
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
    elif test_type == "Pos_Err":
        loop_values = torch.as_tensor(
            etas,
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
    elif test_type in {"Temp", "Grid"}:
        loop_values = torch.as_tensor(
            params,
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
    else:
        loop_values = torch.zeros(
            1,
            device=device,
            dtype=torch.float32,
        )

    num_loop = len(loop_values)
    corr_all = torch.empty(
        num_loop,
        num_sample,
        tau,
        2 * N,
        N,
        device=device,
        dtype=torch.float32,
    )
    sig_n_all = torch.empty(
        num_loop,
        num_sample,
        N,
        snapshot_value,
        device=device,
        dtype=torch.complex64,
    )
    doa_all = torch.empty(
        num_loop,
        num_sample,
        num_sources,
        device=device,
        dtype=torch.float32,
    )

    if test_type == "Grid":
        label_all = [
            torch.empty(
                num_sample,
                int(round((deg_u - deg_l) / interval.item())) + 1,
                device=device,
                dtype=torch.float32,
            )
            for interval in loop_values
        ]
    else:
        label_all = torch.empty(
            num_loop,
            num_sample,
            n_grids,
            device=device,
            dtype=torch.float32,
        )

    for loop_index, loop_value in enumerate(tqdm(loop_values, desc=test_type)):
        current_eta = loop_value if test_type == "Pos_Err" else eta_default
        current_coup = loop_value if test_type == "Coup_Level" else coup_default

        if test_type == "Grid":
            current_deg_p = loop_value
            current_n_grids = label_all[loop_index].shape[1]
        else:
            current_deg_p = deg_p
            current_n_grids = n_grids

        current_deta = loop_value if test_type == "Srcs_Dist" else deta
        doa_generator = DOAGenerator(
            num_sources,
            Gap,
            current_deta,
            fix,
            angle_type,
            device,
        )
        corr_generator = make_corr_generator(
            num_sources,
            current_eta,
        )

        start_index = 0
        for _ in range(num_batches):
            current_batch_size = min(
                batch_size,
                num_sample - start_index,
            )
            snapshot_batch = torch.full(
                (current_batch_size,),
                snapshot_value,
                device=device,
                dtype=torch.long,
            )
            snr_batch = _expand_to_batch(
                snrs,
                current_batch_size,
                device,
                torch.float32,
            )
            eta_batch = _expand_to_batch(
                current_eta,
                current_batch_size,
                device,
                torch.float32,
            )

            doa_batch = doa_generator.generate_doa_batch(current_batch_size)
            corr_batch, signal_batch = corr_generator.creat_corr_mtr_batch(
                current_batch_size,
                doa_batch,
                snr_batch,
                current_coup,
                snapshot_batch,
                eta_batch,
            )
            label_batch = _build_multihot_labels(
                doa_batch,
                deg_l,
                current_deg_p,
                current_n_grids,
            )

            end_index = start_index + current_batch_size
            corr_all[
                loop_index,
                start_index:end_index,
            ] = corr_batch
            sig_n_all[
                loop_index,
                start_index:end_index,
            ] = signal_batch
            doa_all[
                loop_index,
                start_index:end_index,
            ] = doa_batch

            if test_type == "Grid":
                label_all[loop_index][start_index:end_index] = label_batch
            else:
                label_all[
                    loop_index,
                    start_index:end_index,
                ] = label_batch

            start_index = end_index

    if test_type in {"CDF", "Complexity"}:
        return (
            corr_all[0],
            sig_n_all[0],
            doa_all[0],
            label_all[0],
        )
    return corr_all, sig_n_all, doa_all, label_all


def move_tensordataset_to_device(
    dataset,
    device,
) -> TensorDataset:
    """Move all tensors in a TensorDataset or Subset to a device."""
    if isinstance(dataset, Subset):
        source_tensors = dataset.dataset.tensors
        indices = dataset.indices
        tensors = tuple(tensor[indices].to(device) for tensor in source_tensors)
    else:
        tensors = tuple(tensor.to(device) for tensor in dataset.tensors)

    return TensorDataset(*tensors)
