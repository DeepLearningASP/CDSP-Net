"""Ablation-model definitions."""

import numpy as np
from models.SubNet.CADDLN import *
from models.SubNet.ADMapNet import *


class CDSP_Net_StaticDict(nn.Module):
    def __init__(self, sources_num, array_num, lamda, grid_step, temp):
        super(CDSP_Net_StaticDict, self).__init__()
        self.D2R = torch.tensor(np.pi / 180.0, dtype=torch.float32)
        self.sources_num = sources_num
        self.array_num = array_num
        self.lamda = lamda
        self.temp = temp
        self.grid_step = grid_step
        self.grid_num = int(120 / self.grid_step + 1)

        self.ADMapNet = ADMapNet(self.sources_num)
        self.triu_idx = torch.triu(torch.ones((self.array_num, self.array_num), dtype=torch.bool))
        self.dict_dim = torch.sum(self.triu_idx).item()
        self.theta_grid = torch.linspace(-60, 60, self.grid_num, dtype=torch.float32)

    def forward(self, Rx: torch.Tensor, array_pos):
        batch_size = Rx.shape[0]
        device = Rx.device
        dtype = torch.complex64
        D2R = self.D2R.to(device)
        theta_grid = self.theta_grid.to(device)

        R_tau_zero = Rx[:, 0, :, :]
        N = self.array_num
        front_part = R_tau_zero[:, :N, :]
        back_part = R_tau_zero[:, N:, :]
        R_est = torch.complex(front_part, back_part)
        y_vec_triu = R_est[:, self.triu_idx]

        array_pos = (
            torch.tensor(array_pos, dtype=torch.float32, device=device)
            if isinstance(array_pos, np.ndarray)
            else array_pos
        )
        array_pos = array_pos.unsqueeze(0).unsqueeze(-1)
        theta_rad = theta_grid * torch.pi / 180
        sin_theta = torch.sin(theta_rad).unsqueeze(0).unsqueeze(0)
        phase = -1j * 2 * torch.pi * array_pos * sin_theta / self.lamda
        a = torch.exp(phase).squeeze(0).to(dtype)

        a_outer = torch.einsum("mg,ng->mng", a, a.conj())

        oc_dict = a_outer[self.triu_idx, :].reshape(self.dict_dim, self.grid_num).to(dtype)
        oc_dict = oc_dict / torch.norm(oc_dict, p=2, dim=0, keepdim=True)

        oc_dict_expand = oc_dict.unsqueeze(0).repeat(batch_size, 1, 1)
        oc_dict_est = oc_dict_expand

        r = y_vec_triu.clone()
        soft_weights_list = []

        for iter_idx in range(self.sources_num):
            k = iter_idx + 1
            inner_prod = torch.abs(torch.einsum("bdg,bd->bg", oc_dict_est.conj(), r))
            P = softsort(inner_prod, self.temp)
            soft_weights = P[:, 0, :]
            soft_weights_list.append(soft_weights)
            W_soft_real = torch.stack(soft_weights_list, dim=1)
            W_soft = W_soft_real.to(oc_dict_est.dtype)
            Phi_soft = torch.einsum("bdg,bkg->bdk", oc_dict_est, W_soft)
            A_batch = torch.matmul(Phi_soft.conj().transpose(1, 2), Phi_soft)
            eye_batch = (
                torch.eye(k, dtype=oc_dict_est.dtype, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
            A_reg_batch = A_batch + (1e-8 + 0j) * eye_batch
            b_batch = torch.einsum("bdk,bd->bk", Phi_soft.conj(), y_vec_triu)
            x_selected_iter = torch.linalg.solve(A_reg_batch, b_batch)
            r = y_vec_triu - torch.einsum("bdk,bk->bd", Phi_soft, x_selected_iter)

        doa_est = self.ADMapNet(Phi_soft)
        doa_est_rad = doa_est * D2R

        return doa_est_rad


def softsort(v, tau):

    sorted_v, _ = torch.sort(v, dim=1, descending=True)
    diff = torch.abs(sorted_v.unsqueeze(2) - v.unsqueeze(1))
    logits = -diff / tau
    P = torch.softmax(logits, dim=-1)
    return P


class CDSP_Net_HardSort(nn.Module):
    def __init__(self, sources_num, array_num, lamda, grid_step):
        super(CDSP_Net_HardSort, self).__init__()
        self.D2R = torch.tensor(np.pi / 180.0, dtype=torch.float32)
        self.sources_num = sources_num
        self.array_num = array_num
        self.lamda = lamda
        self.grid_step = grid_step
        self.grid_num = int(120 / self.grid_step + 1)

        self.ADMapNet = ADMapNet(self.sources_num)
        self.triu_idx = torch.triu(torch.ones((self.array_num, self.array_num), dtype=torch.bool))
        self.dict_dim = torch.sum(self.triu_idx).item()

        self.CADDLN = CADDLN(
            dict_dim=self.dict_dim, grid_num=self.grid_num, d_model=64, nhead=2, num_layers=2
        )
        self.theta_grid = torch.linspace(-60, 60, self.grid_num, dtype=torch.float32)

    def forward(self, Rx: torch.Tensor, array_pos):
        batch_size = Rx.shape[0]
        device = Rx.device
        dtype = torch.complex64
        D2R = self.D2R.to(device)
        theta_grid = self.theta_grid.to(device)

        mean_signal = Rx[:, 0, :, :]
        N = self.array_num
        front_part = mean_signal[:, :N, :]
        back_part = mean_signal[:, N:, :]
        R_est = torch.complex(front_part, back_part)
        y_vec_triu = R_est[:, self.triu_idx]

        array_pos = (
            torch.tensor(array_pos, dtype=torch.float32, device=device)
            if isinstance(array_pos, np.ndarray)
            else array_pos
        )
        array_pos = array_pos.unsqueeze(0).unsqueeze(-1)
        theta_rad = theta_grid * torch.pi / 180
        sin_theta = torch.sin(theta_rad).unsqueeze(0).unsqueeze(0)
        phase = -1j * 2 * torch.pi * array_pos * sin_theta / self.lamda
        a = torch.exp(phase).squeeze(0).to(dtype)

        a_outer = torch.einsum("mg,ng->mng", a, a.conj())

        oc_dict = a_outer[self.triu_idx, :].reshape(self.dict_dim, self.grid_num).to(dtype)
        oc_dict = oc_dict / torch.norm(oc_dict, p=2, dim=0, keepdim=True)

        oc_dict_expand = oc_dict.unsqueeze(0).repeat(batch_size, 1, 1)
        oc_dict_est = self.CADDLN(y_vec_triu, oc_dict_expand)
        oc_dict_est = oc_dict_est.to(dtype)

        r = y_vec_triu.clone()
        lambda_set_batch = torch.full(
            (batch_size, self.sources_num), -1, dtype=torch.long, device=device
        )

        for iter_idx in range(self.sources_num):
            k = iter_idx + 1

            inner_prod = torch.abs(torch.einsum("bdg,bd->bg", oc_dict_est.conj(), r))
            idx_batch = torch.argmax(inner_prod, dim=1)
            lambda_set_batch[:, iter_idx] = idx_batch

            idx_gather = lambda_set_batch[:, :k].unsqueeze(1)

            Phi_selected_batch = torch.gather(
                oc_dict_est, dim=2, index=idx_gather.expand(-1, self.dict_dim, -1)
            )

            A_batch = torch.matmul(Phi_selected_batch.conj().transpose(1, 2), Phi_selected_batch)
            eye_batch = (
                torch.eye(k, dtype=dtype, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            )
            A_reg_batch = A_batch + (1e-8 + 0j) * eye_batch
            b_batch = torch.einsum("bdk,bd->bk", Phi_selected_batch.conj(), y_vec_triu)
            x_selected_iter = torch.linalg.solve(A_reg_batch, b_batch)
            r = y_vec_triu - torch.einsum("bdk,bk->bd", Phi_selected_batch, x_selected_iter)

        idx_gather_final = lambda_set_batch.unsqueeze(1)
        Phi_selected_final = torch.gather(
            oc_dict_est, dim=2, index=idx_gather_final.expand(-1, self.dict_dim, -1)
        )
        doa_est = self.ADMapNet(Phi_selected_final)
        doa_est_rad = doa_est * D2R

        return doa_est_rad


class CDSP_Net_DiffInput(nn.Module):
    def __init__(self, sources_num, array_num, lamda, grid_step, temp, in_type):
        super(CDSP_Net_DiffInput, self).__init__()
        self.D2R = torch.tensor(np.pi / 180.0, dtype=torch.float32)
        self.sources_num = sources_num
        self.array_num = array_num
        self.lamda = lamda
        self.temp = temp
        self.in_type = in_type
        self.grid_step = grid_step
        self.grid_num = int(120 / self.grid_step + 1)

        self.ADMapNet = ADMapNet(self.sources_num)
        if self.in_type == "VSCM":
            self.dict_dim = array_num * array_num
        elif self.in_type == "MCV":

            self.dict_dim = array_num

        self.CADDLN = CADDLN(
            dict_dim=self.dict_dim, grid_num=self.grid_num, d_model=64, nhead=2, num_layers=2
        )
        self.theta_grid = torch.linspace(-60, 60, self.grid_num, dtype=torch.float32)

    def forward(self, Rx, signal, array_pos):
        batch_size = Rx.shape[0]
        device = Rx.device
        dtype = torch.complex64
        D2R = self.D2R.to(device)
        theta_grid = self.theta_grid.to(device)

        if self.in_type == "VSCM":
            R_tau_zero = Rx[:, 0, :, :]
            front_part = R_tau_zero[:, : self.array_num, :]
            back_part = R_tau_zero[:, self.array_num :, :]
            R_est = torch.complex(front_part, back_part)

            y_vec_triu = R_est.reshape(batch_size, -1)
        elif self.in_type == "MCV":

            R_tau_zero = Rx[:, 0, :, :]
            front_part = R_tau_zero[:, : self.array_num, :]
            back_part = R_tau_zero[:, self.array_num :, :]
            R_est = torch.complex(front_part, back_part)

            y_vec_triu = torch.zeros((batch_size, self.array_num), dtype=dtype, device=device)

            for i, lag in enumerate(range(0, self.array_num)):
                diag_elements = torch.diagonal(R_est, offset=lag, dim1=1, dim2=2)
                y_vec_triu[:, i] = torch.mean(diag_elements, dim=-1)

        array_pos = (
            torch.tensor(array_pos, dtype=torch.float32, device=device)
            if isinstance(array_pos, np.ndarray)
            else array_pos
        )
        array_pos = array_pos.unsqueeze(0).unsqueeze(-1)
        theta_rad = theta_grid * torch.pi / 180
        sin_theta = torch.sin(theta_rad).unsqueeze(0).unsqueeze(0)

        phase = -1j * 2 * torch.pi * array_pos * sin_theta / self.lamda
        a = torch.exp(phase).squeeze(0).to(dtype)

        if self.in_type == "VSCM":

            a_outer = torch.einsum("mg,ng->mng", a, a.conj())

            oc_dict = a_outer.reshape(-1, self.grid_num)
        elif self.in_type == "MCV":

            d = array_pos[0, 1, 0] - array_pos[0, 0, 0]

            lags = torch.arange(0, self.array_num, device=device).float()
            v_array_pos = (lags * d).view(-1, 1)

            v_phase = -1j * 2 * torch.pi * v_array_pos * sin_theta / self.lamda
            oc_dict = torch.exp(v_phase).squeeze(0)

        oc_dict = oc_dict / torch.norm(oc_dict, p=2, dim=0, keepdim=True)

        oc_dict_expand = oc_dict.unsqueeze(0).repeat(batch_size, 1, 1)

        oc_dict_est = self.CADDLN(y_vec_triu, oc_dict_expand)
        oc_dict_est = oc_dict_est.to(dtype)

        r = y_vec_triu.clone()
        soft_weights_list = []

        for iter_idx in range(self.sources_num):
            k = iter_idx + 1

            inner_prod = torch.abs(torch.einsum("bdg,bd->bg", oc_dict_est.conj(), r))

            P = softsort(inner_prod, self.temp)
            soft_weights = P[:, 0, :]
            soft_weights_list.append(soft_weights)

            W_soft_real = torch.stack(soft_weights_list, dim=1)

            W_soft = W_soft_real.to(oc_dict_est.dtype)

            Phi_soft = torch.einsum("bdg,bkg->bdk", oc_dict_est, W_soft)

            A_batch = torch.matmul(Phi_soft.conj().transpose(1, 2), Phi_soft)
            eye_batch = (
                torch.eye(k, dtype=oc_dict_est.dtype, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
            A_reg_batch = A_batch + (1e-8 + 0j) * eye_batch
            b_batch = torch.einsum("bdk,bd->bk", Phi_soft.conj(), y_vec_triu)
            x_selected_iter = torch.linalg.solve(A_reg_batch, b_batch)
            r = y_vec_triu - torch.einsum("bdk,bk->bd", Phi_soft, x_selected_iter)

        doa_est = self.ADMapNet(Phi_soft)
        doa_est_rad = doa_est * D2R

        return doa_est_rad


class CDSP_Net_TopM(nn.Module):
    def __init__(self, sources_num, array_num, lamda, grid_step, temp):
        super().__init__()
        self.D2R = torch.tensor(np.pi / 180.0, dtype=torch.float32)
        self.sources_num = sources_num
        self.array_num = array_num
        self.lamda = lamda
        self.temp = temp
        self.grid_step = grid_step
        self.grid_num = int(120 / self.grid_step + 1)
        self.ADMapNet = ADMapNet(self.sources_num)
        self.triu_idx = torch.triu(torch.ones((self.array_num, self.array_num), dtype=torch.bool))
        self.dict_dim = torch.sum(self.triu_idx).item()
        self.CADDLN = CADDLN(
            dict_dim=self.dict_dim, grid_num=self.grid_num, d_model=64, nhead=2, num_layers=2
        )
        self.theta_grid = torch.linspace(-60, 60, self.grid_num, dtype=torch.float32)

    def softsort(self, v, temp):
        sorted_v, _ = torch.sort(v, dim=1, descending=True)
        diff = torch.abs(sorted_v.unsqueeze(2) - v.unsqueeze(1))
        return torch.softmax(-diff / temp, dim=-1)

    def forward(self, Rx, array_pos):
        batch_size = Rx.shape[0]
        device = Rx.device
        dtype = torch.complex64
        D2R = self.D2R.to(device)
        theta_grid = self.theta_grid.to(device)

        R_tau_zero = Rx[:, 0, :, :]
        N = self.array_num
        R_est = torch.complex(R_tau_zero[:, :N, :], R_tau_zero[:, N:, :])
        y_vec_triu = R_est[:, self.triu_idx]

        array_pos = (
            torch.tensor(array_pos, dtype=torch.float32, device=device)
            if isinstance(array_pos, np.ndarray)
            else array_pos
        )
        array_pos = array_pos.unsqueeze(0).unsqueeze(-1)
        theta_rad = theta_grid * torch.pi / 180
        sin_theta = torch.sin(theta_rad).unsqueeze(0).unsqueeze(0)
        phase = -1j * 2 * torch.pi * array_pos * sin_theta / self.lamda
        a = torch.exp(phase).squeeze(0).to(dtype)
        a_outer = torch.einsum("mg,ng->mng", a, a.conj())
        oc_dict = a_outer[self.triu_idx, :].reshape(self.dict_dim, self.grid_num).to(dtype)
        oc_dict = oc_dict / torch.norm(oc_dict, p=2, dim=0, keepdim=True)

        oc_dict_expand = oc_dict.unsqueeze(0).repeat(batch_size, 1, 1)
        oc_dict_est = self.CADDLN(y_vec_triu, oc_dict_expand).to(dtype)

        Phi_soft = oc_dict_est[:, :, : self.sources_num]

        doa_est = self.ADMapNet(Phi_soft)
        return doa_est * D2R
