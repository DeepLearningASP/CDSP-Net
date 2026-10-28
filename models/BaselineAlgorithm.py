"""Baseline estimation methods."""

import torch
import numpy as np
import scipy.signal


class TraditionalMethods:

    def __init__(self, N, M, T, fc, grid_step, spectrum_step, array_pos, lamda, device,
                 *, real_dtype=torch.float32):

        self.N = N
        self.M = M
        self.T = T
        self.fc = fc
        self.grid_step = grid_step
        self.spectrum_step = spectrum_step
        if real_dtype not in (torch.float32, torch.float64):
            raise ValueError("real_dtype must be torch.float32 or torch.float64.")
        self.real_dtype = real_dtype
        input_positions = torch.as_tensor(array_pos, device=device)
        self._array_pos_input_dtype = input_positions.dtype
        self.array_pos = input_positions.to(dtype=real_dtype)
        self.lamda = lamda
        self.device = device
        self.dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        self.triu_idx = torch.triu_indices(N, N, device=device)
        self.dict_dim = N * (N + 1) // 2
        self.D2R = torch.tensor(np.pi / 180.0, dtype=real_dtype, device=device)
        self.R2D = torch.tensor(180.0 / np.pi, dtype=real_dtype, device=device)
        self.grid_num = int(120 / self.grid_step + 1)
        self.theta_grid = torch.linspace(-60, 60, self.grid_num, dtype=real_dtype).to(
            self.device
        )
        self.spectrum_num = int(120 / self.spectrum_step + 1)
        self.theta_spectrum = torch.linspace(-60, 60, self.spectrum_num, dtype=real_dtype).to(
            self.device
        )

        self.epsilon = 1e-8

        self.Rs = 1
        self.Rc = 3
        self.gamma = 4

        self.oc_dict = self._precompute_dict()

        self.standard_oc_dict = self._precompute_standard_dict()

        self._precompute_nomp_grid()

        self._precompute_music_steering_vectors()

    def _precompute_dict(self):

        array_pos = self.array_pos.unsqueeze(0).unsqueeze(-1)
        theta_rad = self.theta_grid * torch.pi / 180
        sin_theta = torch.sin(theta_rad).unsqueeze(0).unsqueeze(0)

        phase = -1j * 2 * torch.pi * array_pos * sin_theta / self.lamda
        a = torch.exp(phase).squeeze(0).to(self.dtype)

        a_outer = torch.einsum("mg,ng->mng", a, a.conj())

        oc_dict = a_outer[self.triu_idx[0], self.triu_idx[1], :]

        oc_dict = oc_dict / torch.norm(oc_dict, p=2, dim=0, keepdim=True)

        return oc_dict

    def smv_omp(self, current_corr):
        batch_size = current_corr.shape[0]

        Rx_real = current_corr[:, 0, : self.N, :]
        Rx_imag = current_corr[:, 0, self.N :, :]
        Rx_complex = torch.complex(Rx_real, Rx_imag)

        y = Rx_complex[:, self.triu_idx[0], self.triu_idx[1]]

        residual = y.clone()
        lambda_set_batch = torch.full(
            (batch_size, self.M), -1, dtype=torch.long, device=self.device
        )
        unselected_mask_batch = torch.ones(
            (batch_size, self.grid_num), dtype=torch.bool, device=self.device
        )

        oc_dict_batch = self.oc_dict.unsqueeze(0).repeat(batch_size, 1, 1)
        converged = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for iter_idx in range(self.M):
            active_mask = ~converged
            if not torch.any(active_mask):
                break

            inner_prod_full = torch.abs(torch.einsum("bng,bn->bg", oc_dict_batch.conj(), residual))

            inner_prod_masked = inner_prod_full.masked_fill(~unselected_mask_batch, -torch.inf)
            idx_batch = torch.argmax(inner_prod_masked, dim=1)

            lambda_set_batch[active_mask, iter_idx] = idx_batch[active_mask]
            unselected_mask_batch.scatter_(1, idx_batch.unsqueeze(1), False)

            idx_gather = lambda_set_batch[:, : iter_idx + 1].unsqueeze(1)
            Phi_selected_batch = torch.gather(
                oc_dict_batch, dim=2, index=idx_gather.expand(-1, self.dict_dim, -1)
            )

            A_batch = torch.matmul(Phi_selected_batch.conj().transpose(1, 2), Phi_selected_batch)
            b_batch = torch.einsum("bnk,bn->bk", Phi_selected_batch.conj(), y)

            A_real = torch.real(A_batch)[active_mask]
            b_real = torch.real(b_batch)[active_mask]

            x_nn = torch.zeros_like(b_real)

            if x_nn.shape[0] > 0:

                L = torch.linalg.matrix_norm(A_real, ord=2)

                step_size = (1.0 / (L + 1e-12)).unsqueeze(1)

                num_pgd_steps = 20
                for _ in range(num_pgd_steps):

                    grad = torch.einsum("bkj,bj->bk", A_real, x_nn) - b_real

                    x_nn = torch.relu(x_nn - step_size * grad)

            x_selected_iter = torch.zeros(
                (batch_size, iter_idx + 1), dtype=self.dtype, device=self.device
            )
            x_selected_iter[active_mask] = x_nn.to(self.dtype)

            residual_new = y - torch.einsum("bnk,bk->bn", Phi_selected_batch, x_selected_iter)
            residual[active_mask] = residual_new[active_mask]

            residual_norm = torch.norm(residual, dim=1)
            converged[residual_norm < self.epsilon] = True

        doa_est_rad = self.theta_grid[lambda_set_batch] * (torch.pi / 180.0)
        return doa_est_rad

    def _precompute_standard_dict(self):

        array_pos = self.array_pos.unsqueeze(-1)
        theta_rad = self.theta_grid * (torch.pi / 180.0)
        sin_theta = torch.sin(theta_rad).unsqueeze(0)

        phase = -1j * 2 * torch.pi * array_pos * sin_theta / self.lamda
        A = torch.exp(phase).to(self.dtype)

        A = A / torch.norm(A, p=2, dim=0, keepdim=True)

        return A

    def mmv_omp(self, current_sig_n):

        batch_size = current_sig_n.shape[0]
        Y = current_sig_n

        if Y.dim() == 2:
            Y = Y.unsqueeze(-1)
        batch_size, N, T = Y.shape

        residual = Y.clone()

        lambda_set_batch = torch.full(
            (batch_size, self.M), -1, dtype=torch.long, device=self.device
        )

        unselected_mask_batch = torch.ones(
            (batch_size, self.grid_num), dtype=torch.bool, device=self.device
        )

        oc_dict_batch = self.standard_oc_dict.unsqueeze(0).repeat(batch_size, 1, 1)
        converged = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for iter_idx in range(self.M):
            active_mask = ~converged
            if not torch.any(active_mask):
                break

            corr = torch.einsum("bng,bnt->bgt", oc_dict_batch.conj(), residual)
            inner_prod_full = torch.norm(corr, p=2, dim=2)

            inner_prod_masked = inner_prod_full.masked_fill(~unselected_mask_batch, -torch.inf)
            inner_prod_masked[~active_mask, :] = -torch.inf

            idx_batch = torch.argmax(inner_prod_masked, dim=1)

            lambda_set_batch[active_mask, iter_idx] = idx_batch[active_mask]
            active_indices = torch.where(active_mask)[0]
            unselected_mask_batch[active_indices, idx_batch[active_indices]] = False

            idx_selected = lambda_set_batch[:, : iter_idx + 1].clamp(min=0)
            idx_gather = idx_selected.unsqueeze(1)
            Phi_selected_batch = torch.gather(
                oc_dict_batch, dim=2, index=idx_gather.expand(-1, self.N, -1)
            )

            x_selected_iter = torch.zeros(
                (batch_size, iter_idx + 1, T), dtype=self.dtype, device=self.device
            )

            if torch.any(active_mask):

                lstsq_result = torch.linalg.lstsq(
                    Phi_selected_batch[active_mask], Y[active_mask], driver="gels"
                )
                x_selected_iter[active_mask] = lstsq_result.solution

            residual_new = Y - torch.matmul(Phi_selected_batch, x_selected_iter)
            residual[active_mask] = residual_new[active_mask]

            residual_norm = torch.norm(residual.reshape(batch_size, -1), p=2, dim=1)
            newly_converged = residual_norm < self.epsilon
            converged = converged | newly_converged

        lambda_set_safe = lambda_set_batch.clamp(min=0)
        doa_est = self.theta_grid[lambda_set_safe]
        doa_est_rad = doa_est * self.D2R

        return doa_est_rad

    def _precompute_nomp_grid(self):

        array_pos = torch.as_tensor(self.array_pos, dtype=self.real_dtype, device=self.device)

        self.grid_num_os = int(round(self.grid_num * self.gamma))
        self.theta_grid_os = torch.linspace(
            -60, 60, self.grid_num_os, dtype=self.real_dtype, device=self.device
        )

        theta_rad_os = self.theta_grid_os * self.D2R
        sin_theta_os = torch.sin(theta_rad_os).unsqueeze(0)
        array_pos_unsq = array_pos.unsqueeze(1)
        phase_os = -1j * 2 * torch.pi * array_pos_unsq * sin_theta_os / self.lamda
        self.a_os = torch.exp(phase_os).to(self.dtype)

        self.a_os = self.a_os / torch.norm(self.a_os, p=2, dim=0, keepdim=True)

        self._nomp_atom_norm = torch.sqrt(
            torch.tensor(float(self.N), device=self.device, dtype=self.real_dtype)
        )

    def _compute_steering_vector(self, theta_deg_batch):

        if theta_deg_batch.dim() == 0:
            theta_deg_batch = theta_deg_batch.unsqueeze(0)

        theta_deg_batch = theta_deg_batch.to(self.device).reshape(-1)
        array_pos = torch.as_tensor(
            self.array_pos, dtype=self.real_dtype, device=self.device
        ).unsqueeze(0)

        theta_rad_batch = theta_deg_batch * self.D2R
        sin_theta_batch = torch.sin(theta_rad_batch).unsqueeze(1)
        phase_batch = -1j * 2 * torch.pi * array_pos * sin_theta_batch / self.lamda
        a = torch.exp(phase_batch).to(self.dtype)

        a = a / self._nomp_atom_norm
        return a

    def _compute_steering_matrix_batch(self, theta_deg_batch):

        theta_deg_batch = theta_deg_batch.to(self.device)
        if theta_deg_batch.dim() == 1:
            theta_deg_batch = theta_deg_batch.unsqueeze(1)

        array_pos = (
            torch.as_tensor(self.array_pos, dtype=self.real_dtype, device=self.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        theta_rad = theta_deg_batch * self.D2R
        sin_theta = torch.sin(theta_rad).unsqueeze(1)
        phase = -1j * 2 * torch.pi * array_pos * sin_theta / self.lamda
        A = torch.exp(phase).to(self.dtype)

        A = A / self._nomp_atom_norm
        return A

    def _newton_refinement_batch(self, theta_init_deg, r_batch, Rs):

        theta_opt = theta_init_deg.clone().to(self.device)
        step_max = 0.5

        array_pos = torch.as_tensor(
            self.array_pos, dtype=self.real_dtype, device=self.device
        ).unsqueeze(0)
        c = (2 * torch.pi / self.lamda) * array_pos

        for _ in range(Rs):
            theta_rad = theta_opt * self.D2R
            sin_theta = torch.sin(theta_rad).unsqueeze(1)
            cos_theta = torch.cos(theta_rad).unsqueeze(1)

            phase = -1j * c * sin_theta
            a_raw = torch.exp(phase).to(self.dtype)

            a = a_raw / self._nomp_atom_norm
            da_dtheta = (-1j * c * cos_theta * a_raw) / self._nomp_atom_norm

            d2a_dtheta2 = (
                ((1j * c * sin_theta) - (c * cos_theta) ** 2) * a_raw / self._nomp_atom_norm
            )

            q = torch.einsum("bn,bnt->bt", a.conj(), r_batch)
            q1 = torch.einsum("bn,bnt->bt", da_dtheta.conj(), r_batch)
            q2 = torch.einsum("bn,bnt->bt", d2a_dtheta2.conj(), r_batch)

            G = torch.sum(torch.abs(q) ** 2, dim=1)

            G_dot = 2 * torch.real(torch.sum(torch.conj(q) * q1, dim=1))
            G_ddot = 2 * (
                torch.sum(torch.abs(q1) ** 2, dim=1)
                + torch.real(torch.sum(torch.conj(q) * q2, dim=1))
            )

            mask = G_ddot < -1e-12
            if torch.any(mask):
                delta_theta_rad = -G_dot[mask] / G_ddot[mask]
                delta_theta_deg = delta_theta_rad * self.R2D
                delta_theta_deg = torch.clamp(delta_theta_deg, -step_max, step_max)
                theta_opt[mask] = theta_opt[mask] + delta_theta_deg

            theta_opt = torch.clamp(theta_opt, -60.0, 60.0)

        a_opt = self._compute_steering_vector(theta_opt)
        q_opt = torch.einsum("bn,bnt->bt", a_opt.conj(), r_batch)
        g_opt_mat = q_opt
        G_opt = torch.sum(torch.abs(q_opt) ** 2, dim=1)

        return theta_opt, g_opt_mat, G_opt

    def nomp(self, current_sig_n):

        if current_sig_n.dim() == 2:
            current_sig_n = current_sig_n.unsqueeze(-1)

        batch_size, N, T = current_sig_n.shape
        K = self.M

        sig_n_orig = current_sig_n.clone()

        theta_est = torch.full((batch_size, K), -999.0, dtype=self.real_dtype, device=self.device)
        g_mat = torch.zeros((batch_size, K, T), dtype=self.dtype, device=self.device)
        r = current_sig_n.clone()

        for iter_idx in range(K):
            num_sel = iter_idx + 1

            inner_prod = torch.einsum("ng,bnt->bgt", self.a_os.conj(), r)
            G = torch.sum(torch.abs(inner_prod) ** 2, dim=2)
            G_max, max_idx = torch.max(G, dim=1)

            theta_coarse = self.theta_grid_os[max_idx]
            a_coarse = self._compute_steering_vector(theta_coarse)
            g_coarse = torch.einsum("bn,bnt->bt", a_coarse.conj(), r)

            theta_refined, g_refined, G_refined = self._newton_refinement_batch(
                theta_coarse, r, self.Rs
            )

            rac_mask = G_refined > G_max
            theta_current = torch.where(rac_mask, theta_refined, theta_coarse)
            g_current = torch.where(rac_mask.unsqueeze(1), g_refined, g_coarse)

            theta_est[:, iter_idx] = theta_current
            g_mat[:, iter_idx, :] = g_current

            for _ in range(self.Rc):
                for k_idx in range(num_sel):
                    theta_selected = theta_est[:, :num_sel].clone()
                    g_selected = g_mat[:, :num_sel].clone()

                    theta_old = theta_selected[:, k_idx].clone()

                    if num_sel == 1:
                        r_excl = sig_n_orig
                    else:
                        temp_idx = torch.cat(
                            [
                                torch.arange(k_idx, device=self.device),
                                torch.arange(k_idx + 1, num_sel, device=self.device),
                            ]
                        )
                        theta_excl = theta_selected[:, temp_idx]
                        g_excl = g_selected[:, temp_idx, :]
                        A_excl = self._compute_steering_matrix_batch(theta_excl)
                        r_excl = sig_n_orig - torch.einsum("bnm,bmt->bnt", A_excl, g_excl)

                    theta_new, g_new, G_new = self._newton_refinement_batch(
                        theta_old, r_excl, self.Rs
                    )

                    a_old = self._compute_steering_vector(theta_old)
                    q_old = torch.einsum("bn,bnt->bt", a_old.conj(), r_excl)
                    G_old = torch.sum(torch.abs(q_old) ** 2, dim=1)

                    rac_cyc_mask = G_new > G_old

                    theta_selected[:, k_idx] = torch.where(rac_cyc_mask, theta_new, theta_old)
                    g_selected[:, k_idx, :] = torch.where(
                        rac_cyc_mask.unsqueeze(1), g_new, g_selected[:, k_idx, :]
                    )

                    theta_est[:, :num_sel] = theta_selected
                    g_mat[:, :num_sel] = g_selected

            A_selected = self._compute_steering_matrix_batch(theta_est[:, :num_sel])
            A_H = A_selected.conj().transpose(1, 2)

            normal_mat = torch.matmul(A_H, A_selected)
            eye_batch = (
                torch.eye(num_sel, dtype=self.dtype, device=self.device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
            normal_mat_reg = normal_mat + (1e-8 + 0j) * eye_batch

            rhs = torch.matmul(A_H, sig_n_orig)
            g_updated = torch.linalg.solve(normal_mat_reg, rhs)

            r = sig_n_orig - torch.matmul(A_selected, g_updated)

            g_mat[:, :num_sel] = g_updated

        g_power = torch.mean(torch.abs(g_mat) ** 2, dim=2)
        _, sort_idx = torch.sort(g_power, dim=1, descending=True)

        theta_est_sorted = torch.gather(theta_est, 1, sort_idx)
        theta_est_sorted = torch.clamp(theta_est_sorted, -60.0, 60.0)

        doa_est_rad = theta_est_sorted * self.D2R
        return doa_est_rad

    def _precompute_music_steering_vectors(self):

        theta_rad = self.theta_spectrum * self.D2R
        sin_theta = torch.sin(theta_rad).unsqueeze(0)
        array_pos_unsq = self.array_pos.unsqueeze(1)

        phase = -1j * 2 * torch.pi * array_pos_unsq * sin_theta / self.lamda
        self.a_music = torch.exp(phase).to(self.dtype)

    def _select_music_peaks(self, spec, min_dist_idx, max_dist_idx):
        # Padding includes one-sided boundary maxima in both MUSIC variants.
        padded_spec = np.pad(spec, (1, 1), mode="constant", constant_values=-np.inf)
        peaks, _ = scipy.signal.find_peaks(padded_spec, distance=min_dist_idx)
        peaks = sorted(peaks - 1, key=lambda index: spec[index], reverse=True)
        selected = []
        for candidates in (peaks, np.argsort(spec)[::-1]):
            for index in candidates:
                if index in selected:
                    continue
                if selected and (
                    any(abs(index - other) < min_dist_idx for other in selected)
                    or max(max(selected), index) - min(min(selected), index) > max_dist_idx
                ):
                    continue
                selected.append(int(index))
                if len(selected) == self.M:
                    return sorted(selected)
        raise ValueError("Unable to select M MUSIC peaks satisfying the requested angular gaps.")

    def music(self, current_corr, min_degree_gap=0, max_degree_gap=120):
        if not 0 < self.M < self.N:
            raise ValueError("MUSIC requires 0 < M < N.")
        if (
            not np.isfinite(min_degree_gap)
            or not np.isfinite(max_degree_gap)
            or not 0 <= min_degree_gap <= max_degree_gap
        ):
            raise ValueError("MUSIC requires finite angular gaps with 0 <= min_gap <= max_gap.")

        Rx_real = current_corr[:, 0, : self.N, :]
        Rx_imag = current_corr[:, 0, self.N :, :]
        Rx_complex = torch.complex(Rx_real, Rx_imag)

        return self._music_core(
            Rx_complex, self.a_music, min_degree_gap, max_degree_gap
        )

    def _select_music_peaks_batch(self, spectrum, min_dist_idx, max_dist_idx):
        """Device-resident peak selection, parallel across scenes and grid.

        Include boundary maxima and plateau midpoints, as the SciPy reference
        does. Equal-height ties choose the smaller grid index deterministically.
        The source-selection loop is shared across every scene in the batch.
        """
        batch, grid_size = spectrum.shape
        indices = torch.arange(grid_size, device=spectrum.device)[None].expand(batch, -1)
        left_change = torch.ones_like(spectrum, dtype=torch.bool)
        right_change = torch.ones_like(left_change)
        left_change[:, 1:] = spectrum[:, 1:] != spectrum[:, :-1]
        right_change[:, :-1] = spectrum[:, :-1] != spectrum[:, 1:]
        starts = torch.where(left_change, indices, 0).cummax(-1).values
        ends = torch.where(right_change, indices, grid_size - 1).flip(-1).cummin(-1).values.flip(-1)
        left = spectrum.gather(1, (starts - 1).clamp(min=0)).masked_fill(starts == 0, -torch.inf)
        right = spectrum.gather(1, (ends + 1).clamp(max=grid_size - 1)).masked_fill(ends == grid_size - 1, -torch.inf)
        peaks = ((indices == (starts + ends) // 2) & (spectrum > left) & (spectrum > right))

        # For a restricted maximum span, reproduce SciPy's initial distance
        # pruning BEFORE applying the maximum-span constraint. With the
        # usual full-FOV maximum span, only the requested M peaks are needed.
        if min_dist_idx > 1 and max_dist_idx < grid_size - 1:
            remaining = peaks.clone()
            peaks = torch.zeros_like(peaks)
            while bool(remaining.any()):
                values, chosen = spectrum.masked_fill(~remaining, -torch.inf).max(-1)
                active = torch.isfinite(values)
                peaks.scatter_(1, chosen[:, None], peaks.gather(1, chosen[:, None]) | active[:, None])
                remaining &= ~(active[:, None] & ((indices - chosen[:, None]).abs() < min_dist_idx))

        selected = torch.full((batch, self.M), -1, dtype=torch.long, device=spectrum.device)
        counts = torch.zeros(batch, dtype=torch.long, device=spectrum.device)
        banned = torch.zeros_like(peaks)
        low = torch.full_like(counts, grid_size)
        high = torch.full_like(counts, -1)
        for candidates in (peaks, torch.ones_like(peaks)):
            for _ in range(self.M):
                too_wide = ((torch.maximum(high[:, None], indices)
                             - torch.minimum(low[:, None], indices)) > max_dist_idx)
                allowed = candidates & ~banned & ~((counts > 0)[:, None] & too_wide)
                values, chosen = spectrum.masked_fill(~allowed, -torch.inf).max(-1)
                active = (counts < self.M) & torch.isfinite(values)
                slot = counts.clamp(max=self.M - 1)[:, None]
                selected.scatter_(1, slot, torch.where(active[:, None], chosen[:, None], selected.gather(1, slot)))
                counts += active.long()
                low = torch.where(active, torch.minimum(low, chosen), low)
                high = torch.where(active, torch.maximum(high, chosen), high)
                banned |= active[:, None] & ((indices - chosen[:, None]).abs() < min_dist_idx)
        if bool(torch.any(counts != self.M)):
            raise ValueError("Unable to select M MUSIC peaks satisfying the requested angular gaps.")
        return selected.sort(-1).values

    def _music_core(self, Rx_complex, steering_vectors, min_degree_gap=0, max_degree_gap=120, normalize=False):

        _, eigenvectors = torch.linalg.eigh(Rx_complex)

        noise_dim = Rx_complex.shape[-1] - self.M
        Un = eigenvectors[:, :, :noise_dim]

        Un_H = Un.conj().transpose(-2, -1)
        projection = torch.matmul(Un_H, steering_vectors)
        denom = torch.sum(torch.abs(projection) ** 2, dim=1)
        spectrum = 1.0 / (denom + 1e-12)

        if normalize:
            spectrum = spectrum / (spectrum.max(dim=1, keepdim=True)[0] + 1e-12)

        grid_resolution = 120.0 / (self.spectrum_num - 1)

        min_dist_idx = max(int(np.ceil(min_degree_gap / grid_resolution)), 1)

        max_dist_idx = int(np.floor(max_degree_gap / grid_resolution))

        top_m_peaks = self._select_music_peaks_batch(spectrum, min_dist_idx, max_dist_idx)
        doa_est = self.theta_spectrum[top_m_peaks]
        doa_est_rad = doa_est * self.D2R

        return doa_est_rad, spectrum

    def g_music(self, current_corr, min_degree_gap=0, max_degree_gap=120):

        Rx_real = current_corr[:, 0, : self.N, :]
        Rx_imag = current_corr[:, 0, self.N :, :]
        Rx_complex = torch.complex(Rx_real, Rx_imag)

        return self._g_music_core(
            Rx_complex, self.a_music, min_degree_gap, max_degree_gap
        )

    def _g_music_core(self, Rx_complex, steering_vectors, min_degree_gap=0, max_degree_gap=120):
        """Mestre--Lagunas (2008), Eqs. (26), (27), (29), (30)."""
        output_dtype = Rx_complex.real.dtype
        N = Rx_complex.shape[-1]
        Rx_complex = Rx_complex.to(torch.complex128)
        Rx_complex = (Rx_complex + Rx_complex.mH) / 2.0
        eigenvalues, eigenvectors = torch.linalg.eigh(Rx_complex)

        # Common scaling cancels in Eq. (27); remove only numerical PSD zeros.
        scale = eigenvalues.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
        eigenvalues = eigenvalues / scale
        tol = 8 * N * torch.finfo(torch.float64).eps
        eigenvalues = torch.where(eigenvalues.abs() < tol, 0.0, eigenvalues).clamp_min(0.0)
        root = eigenvalues.sqrt()
        # Eq. (29): mu are eigenvalues of diag(lambda)-sqrt(lambda)sqrt(lambda)^T/T.
        mu = torch.linalg.eigvalsh(
            torch.diag_embed(eigenvalues) - root[:, :, None] * root[:, None, :] / self.T
        )
        mu = torch.where(mu.abs() < tol, 0.0, mu)
        noise_dim = N - self.M

        def ratio(numerator, denominator):
            # Footnote 1: terms of the form 0/0 are zero.
            return torch.where(numerator == 0.0, 0.0, numerator / denominator)

        noise_weights = 1.0 + (
            ratio(eigenvalues[:, None, noise_dim:], eigenvalues[:, :noise_dim, None] - eigenvalues[:, None, noise_dim:])
            - ratio(mu[:, None, noise_dim:], eigenvalues[:, :noise_dim, None] - mu[:, None, noise_dim:])
        ).sum(-1)
        signal_weights = -(
            ratio(eigenvalues[:, None, :noise_dim], eigenvalues[:, noise_dim:, None] - eigenvalues[:, None, :noise_dim])
            - ratio(mu[:, None, :noise_dim], eigenvalues[:, noise_dim:, None] - mu[:, None, :noise_dim])
        ).sum(-1)
        weights = torch.cat((noise_weights, signal_weights), dim=-1)
        projection = eigenvectors.mH @ steering_vectors.to(Rx_complex.dtype)
        cost = (weights[:, :, None] * projection.abs().square()).sum(1) / N
        # Eq. (30) selects minima of a possibly negative cost; do not invert it.
        spectrum = -cost

        grid_resolution = 120.0 / (self.spectrum_num - 1)
        min_dist_idx = max(int(np.ceil(min_degree_gap / grid_resolution)), 1)
        max_dist_idx = int(np.floor(max_degree_gap / grid_resolution))
        top_m_peaks = self._select_music_peaks_batch(spectrum, min_dist_idx, max_dist_idx)
        doa_est_rad = self.theta_spectrum[top_m_peaks] * self.D2R

        return doa_est_rad.to(output_dtype), spectrum.to(output_dtype)

    def esprit(self, current_corr):

        Rx_real = current_corr[:, 0, : self.N, :]
        Rx_imag = current_corr[:, 0, self.N :, :]
        Rx_complex = torch.complex(Rx_real, Rx_imag)

        d = (self.array_pos[1] - self.array_pos[0]).to(Rx_real.dtype)

        return self._esprit_core(Rx_complex, d)

    def _esprit_core(self, Rx_complex, d):

        _, eigenvectors = torch.linalg.eigh(Rx_complex)

        Us = eigenvectors[:, :, -self.M :]

        Us_upper = Us[:, :-1, :]
        Us_lower = Us[:, 1:, :]

        phi = torch.linalg.pinv(Us_upper) @ Us_lower

        phi_eigenvalues, _ = torch.linalg.eig(phi)

        eigenvalues_angles = torch.angle(phi_eigenvalues).to(d.dtype)

        sin_est = -self.lamda * eigenvalues_angles / (2 * torch.pi * d)

        sin_est = torch.clamp(sin_est, min=-1.0, max=1.0)

        doa_est_rad = torch.arcsin(sin_est)
        doa_est_rad = torch.clamp(doa_est_rad, min=-torch.pi / 3, max=torch.pi / 3)

        doa_est_rad, _ = torch.sort(doa_est_rad, dim=1)

        return doa_est_rad

    def unitary_esprit(self, current_corr):

        Rx_real = current_corr[:, 0, : self.N, :]
        Rx_imag = current_corr[:, 0, self.N :, :]
        Rx_complex = torch.complex(Rx_real, Rx_imag)

        d = self.array_pos[1] - self.array_pos[0]

        return self._unitary_esprit_core(Rx_complex, d)

    def _unitary_esprit_core(self, Rx_complex, d):

        N = Rx_complex.shape[-1]
        Rx_fb = (Rx_complex + Rx_complex.conj().flip(dims=[1, 2])) / 2.0

        Q = self._unitary_matrix(N, Rx_complex.dtype, Rx_complex.device)
        Q_sub = self._unitary_matrix(N - 1, Rx_complex.dtype, Rx_complex.device)

        Rx_unitary = (Q.conj().transpose(0, 1) @ Rx_fb @ Q).real
        Rx_unitary = (Rx_unitary + Rx_unitary.transpose(1, 2)) / 2.0

        _, eigenvectors = torch.linalg.eigh(Rx_unitary)

        Us = eigenvectors[:, :, -self.M :]

        # Real selection matrices for the overlapping ULA subarrays.
        selection = Q_sub.conj().transpose(0, 1) @ Q[1:, :]
        K1 = 2.0 * selection.real
        K2 = 2.0 * selection.imag

        phi = torch.linalg.pinv(K1 @ Us) @ (K2 @ Us)

        phi_eigenvalues = torch.linalg.eigvals(phi)
        eigenvalues_angles = 2.0 * torch.atan(phi_eigenvalues.real)

        sin_est = -self.lamda * eigenvalues_angles / (2 * torch.pi * d)
        sin_est = torch.clamp(sin_est, min=-1.0, max=1.0)

        doa_est_rad = torch.arcsin(sin_est)
        doa_est_rad = torch.clamp(doa_est_rad, min=-torch.pi / 3, max=torch.pi / 3)

        doa_est_rad, _ = torch.sort(doa_est_rad, dim=1)

        return doa_est_rad

    def _unitary_matrix(self, N, dtype, device):

        Q = torch.zeros((N, N), dtype=dtype, device=device)
        half = N // 2
        idx = torch.arange(half, device=device)
        right = idx + half + N % 2

        Q[idx, idx] = 1.0 / np.sqrt(2.0)
        Q[N - 1 - idx, idx] = 1.0 / np.sqrt(2.0)
        Q[idx, right] = 1j / np.sqrt(2.0)
        Q[N - 1 - idx, right] = -1j / np.sqrt(2.0)

        if N % 2:
            Q[half, half] = 1.0

        return Q

    def _check_ula_spacing(self, atol=1e-8):
        original_pos = torch.as_tensor(self.array_pos, device=self.device)
        pos = original_pos.to(torch.float64)
        if pos.ndim != 1 or pos.numel() != self.N or pos.numel() < 2:
            raise ValueError("array_pos must contain N sensor positions, with N >= 2.")
        if not bool(torch.isfinite(pos).all()):
            raise ValueError("array_pos must contain finite sensor positions.")
        diffs = pos[1:] - pos[:-1]
        d_mean = diffs.mean()
        # Account for rounding in the original position dtype before conversion.
        rounding_atol = 0.0
        original_dtype = getattr(self, "_array_pos_input_dtype", original_pos.dtype)
        if original_dtype.is_floating_point:
            rounding_atol = 4 * torch.finfo(original_dtype).eps * float(pos.abs().max())
        if d_mean <= 0 or torch.max(torch.abs(diffs - d_mean)) > max(atol, rounding_atol):
            return False, pos, None
        return True, pos, d_mean

    def _build_virtual_coarray(self, Rx_complex):

        batch_size, N, _ = Rx_complex.shape
        device = Rx_complex.device
        dtype = Rx_complex.dtype
        d0 = self.lamda / 2.0

        pos = torch.as_tensor(self.array_pos, dtype=torch.float64, device=device)
        pos0 = pos.min()
        idx_float = (pos - pos0) / d0
        idx = torch.round(idx_float).long()

        if torch.max(torch.abs(idx_float - idx.to(idx_float.dtype))) > 1e-6:
            raise ValueError("SLA Coarray requires array_pos on λ/2 grid.")

        lag_mat = idx[:, None] - idx[None, :]
        lag_set = set(lag_mat.flatten().detach().cpu().tolist())

        Lmax = 0
        while (Lmax + 1) in lag_set and -(Lmax + 1) in lag_set:
            Lmax += 1

        r = torch.zeros((batch_size, 2 * Lmax + 1), dtype=dtype, device=device)
        for lag in range(-Lmax, Lmax + 1):
            mask = lag_mat == lag
            vals = Rx_complex[:, mask]
            r[:, lag + Lmax] = vals.mean(dim=1)

        Vv = Lmax + 1
        if Vv <= self.M:
            raise ValueError(f"Virtual DOF Vv={Vv} must be > M={self.M}.")

        R_virtual = torch.zeros((batch_size, Vv, Vv), dtype=dtype, device=device)
        for i in range(Vv):
            for j in range(Vv):
                R_virtual[:, i, j] = r[:, i - j + Lmax]

        R_virtual = (R_virtual + R_virtual.conj().transpose(-1, -2)) / 2.0
        virtual_pos = torch.arange(Vv, device=device, dtype=torch.float64) * d0

        return R_virtual, virtual_pos

    def _spatial_smoothing(self, Rx_complex, forward_backward=True):
        if not 0 < self.M < self.N:
            raise ValueError("Spatial smoothing requires 0 < M < N.")

        is_ula, pos, d_mean = self._check_ula_spacing()
        if not is_ula:
            raise ValueError("Spatial smoothing requires uniformly spaced, increasing sensor positions.")

        if is_ula:

            batch_size, N, _ = Rx_complex.shape

            if not forward_backward:

                L_min = self.M + 1
                L_max = N - self.M + 1

                L_optimal = (N + 2) // 2
            else:

                L_min = self.M + 1
                L_max = N - (self.M + 1) // 2 + 1

                L_optimal = (N + self.M + 1) // 2

            if L_min > L_max:
                raise ValueError(
                    f"Insufficient array degrees of freedom for N={N}, M={self.M}; {'FBSS' if forward_backward else 'FSS'} cannot recover full rank."
                )

            L = min(max(L_optimal, L_min), L_max)
            K = N - L + 1

            Rx_fwd = torch.zeros(
                (batch_size, L, L), dtype=Rx_complex.dtype, device=Rx_complex.device
            )
            for k in range(K):
                Rx_fwd += Rx_complex[:, k : k + L, k : k + L]
            Rx_fwd /= K

            if not forward_backward:
                sub_array_pos = pos[:L] - pos[0]
                return (Rx_fwd + Rx_fwd.conj().transpose(-1, -2)) / 2.0, sub_array_pos

            J = torch.flip(torch.eye(L, dtype=Rx_complex.dtype, device=Rx_complex.device), dims=[1])
            Rx_bwd = J @ Rx_fwd.conj() @ J
            Rx_smooth = (Rx_fwd + Rx_bwd) / 2.0
            return (Rx_smooth + Rx_smooth.conj().transpose(-1, -2)) / 2.0, pos[:L] - pos[0]

    def _compute_subarray_steering_vector(self, sub_array_pos):
        device = sub_array_pos.device
        theta_rad = self.theta_spectrum.to(device=device, dtype=torch.float64) * self.D2R
        sin_theta = torch.sin(theta_rad)[None, :]
        x = sub_array_pos[:, None].to(dtype=torch.float64)
        k = 2.0 * torch.pi / self.lamda
        phase = -1j * k * x * sin_theta
        return torch.exp(phase).to(self.dtype)

    def sps_music(self, current_corr, forward_backward=True):

        Rx_real = current_corr[:, 0, : self.N, :]
        Rx_imag = current_corr[:, 0, self.N :, :]
        Rx_complex = torch.complex(Rx_real, Rx_imag)

        Rx_smooth, sub_array_pos = self._spatial_smoothing(Rx_complex, forward_backward)
        a_sub = self._compute_subarray_steering_vector(sub_array_pos)

        return self._music_core(Rx_smooth, a_sub, normalize=True)

    def sps_g_music(self, current_corr, forward_backward=True):

        Rx_real = current_corr[:, 0, : self.N, :]
        Rx_imag = current_corr[:, 0, self.N :, :]
        Rx_complex = torch.complex(Rx_real, Rx_imag).to(torch.complex128)

        Rx_smooth, sub_array_pos = self._spatial_smoothing(Rx_complex, forward_backward)
        a_sub = self._compute_subarray_steering_vector(sub_array_pos)
        doa_est_rad, spectrum = self._g_music_core(Rx_smooth, a_sub)

        return doa_est_rad.to(Rx_real.dtype), spectrum.to(Rx_real.dtype)

    def sps_esprit(self, current_corr, forward_backward=True):

        Rx_real = current_corr[:, 0, : self.N, :]
        Rx_imag = current_corr[:, 0, self.N :, :]
        Rx_complex = torch.complex(Rx_real, Rx_imag)

        Rx_smooth, sub_array_pos = self._spatial_smoothing(Rx_complex, forward_backward)
        d_eff = torch.mean(sub_array_pos[1:] - sub_array_pos[:-1]).to(
            dtype=torch.float64, device=sub_array_pos.device
        )

        return self._esprit_core(Rx_smooth, d_eff)

    def sps_unitary_esprit(self, current_corr, forward_backward=True):

        Rx_real = current_corr[:, 0, : self.N, :]
        Rx_imag = current_corr[:, 0, self.N :, :]
        Rx_complex = torch.complex(Rx_real, Rx_imag)

        Rx_smooth, sub_array_pos = self._spatial_smoothing(Rx_complex, forward_backward)
        d_eff = torch.mean(sub_array_pos[1:] - sub_array_pos[:-1]).to(
            dtype=torch.float64, device=sub_array_pos.device
        )

        return self._unitary_esprit_core(Rx_smooth, d_eff)
