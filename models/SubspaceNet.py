"""SubspaceNet model definition."""

import torch
import torch.nn as nn


def root_music(N, array_pos, freq, M, Rz, batch_size):

    device = Rz.device
    B = batch_size
    output_real_dtype = Rz.real.dtype
    output_complex_dtype = Rz.dtype
    # Nearly reciprocal roots can cross the unit circle in single precision.
    # Use double precision for the subspace and polynomial, retaining gradients.
    Rz = Rz.to(torch.complex128)
    dtype = Rz.dtype
    c_light = 3e8
    lambda_ = c_light / freq
    lambda_half = lambda_ / 2
    phi_scale = torch.pi
    positions = array_pos.to(device=device, dtype=torch.float64)
    if positions.ndim != 1 or positions.numel() != N or not bool(torch.isfinite(positions).all()):
        raise ValueError("Root-MUSIC requires N finite sensor positions.")
    array_pos_norm = (positions - positions.min()) / lambda_half
    lattice_positions = array_pos_norm.round()
    if not torch.allclose(array_pos_norm, lattice_positions, atol=1e-5, rtol=0):
        raise ValueError("Root-MUSIC requires sensor positions on a half-wavelength lattice.")
    array_pos_norm = lattice_positions
    pos_max = torch.max(array_pos_norm).item()
    poly_order = int(2 * pos_max)
    if not 0 < M < N or poly_order < 2 * M:
        raise ValueError("Root-MUSIC requires 0 < M < N and sufficient array aperture.")

    eps_eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0) * 1e-9
    U, S, Vh = torch.linalg.svd(Rz + eps_eye)
    Us = U[:, :, :M]

    I = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
    F = I - torch.matmul(Us, Us.conj().transpose(1, 2))
    F = (F + F.conj().transpose(-2, -1)) / 2

    i_idx, j_idx = torch.meshgrid(array_pos_norm, array_pos_norm, indexing="ij")
    power_diff = j_idx - i_idx
    coeff_idx = (poly_order - (power_diff + pos_max)).round().long()
    P_coeff = torch.zeros(B, poly_order + 1, dtype=dtype, device=device)
    P_coeff.scatter_add_(1, coeff_idx.view(-1).unsqueeze(0).expand(B, -1), F.reshape(B, -1))

    lead_coeffs = P_coeff[:, 0:1] + 1e-12
    coeffs_norm = P_coeff / lead_coeffs
    n_poly = coeffs_norm.shape[1] - 1

    comp = torch.zeros((B, n_poly, n_poly), dtype=dtype, device=device)
    if n_poly > 1:
        comp[:, 1:, :-1] = torch.eye(n_poly - 1, device=device, dtype=dtype).unsqueeze(0)
    comp[:, 0, :] = -coeffs_norm[:, 1:]

    roots_all = torch.linalg.eigvals(comp)
    roots_abs = torch.abs(roots_all)
    dist_to_unit = torch.abs(roots_abs - 1.0)
    dist_to_unit = torch.where(roots_abs > 1.0, torch.tensor(10.0, device=device), dist_to_unit)
    _, topk_idx = torch.topk(dist_to_unit, k=M, dim=-1, largest=False)
    roots_final = torch.gather(roots_all, dim=-1, index=topk_idx)

    doa_prediction = torch.arcsin(torch.clamp(-torch.angle(roots_final) / phi_scale, -0.999, 0.999))
    doa_all_prediction = torch.arcsin(
        torch.clamp(-torch.angle(roots_all) / phi_scale, -0.999, 0.999)
    )

    return (
        doa_prediction.to(output_real_dtype),
        doa_all_prediction.to(output_real_dtype),
        roots_all.to(output_complex_dtype),
    )


class SubspaceNet(nn.Module):
    def __init__(self, tau, M, fc, array_pos, array_type="ULA"):
        super(SubspaceNet, self).__init__()
        self.M = M
        self.fc = fc
        self.array_pos = (
            array_pos if isinstance(array_pos, torch.Tensor) else torch.tensor(array_pos).float()
        )
        self.array_type = array_type
        self.tau = tau

        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()

        self.diff_method = root_music

    def gram_diagonal_overload(self, Kx: torch.Tensor, eps: float) -> torch.Tensor:
        B, N, _ = Kx.shape
        Kx_H = torch.conj(Kx).transpose(1, 2)
        Kx_gram = torch.matmul(Kx_H, Kx)
        eps_addition = eps * torch.eye(N, device=Kx.device).unsqueeze(0)
        return Kx_gram + eps_addition

    def anti_rectifier(self, X):
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def forward(self, Rx_tau: torch.Tensor):

        N_physical = Rx_tau.shape[-1]
        B = Rx_tau.shape[0]

        x = self.anti_rectifier(self.conv1(Rx_tau))
        x = self.anti_rectifier(self.conv2(x))
        x = self.anti_rectifier(self.conv3(x))
        x = self.anti_rectifier(self.deconv2(x))
        x = self.anti_rectifier(self.deconv3(x))
        Rx = self.deconv4(self.DropOut(x))

        Rx_View = Rx.view(B, 2 * N_physical, N_physical)
        Kx_tag = torch.complex(Rx_View[:, :N_physical, :], Rx_View[:, N_physical:, :])

        Rz_final = self.gram_diagonal_overload(Kx=Kx_tag, eps=0.1)

        if self.array_type.upper() == "ULA":

            N_eval = N_physical
            pos_eval = self.array_pos.to(Rx_tau.device)
            Rz_eval = Rz_final

        method_output = self.diff_method(
            N=N_eval, array_pos=pos_eval, freq=self.fc, M=self.M, Rz=Rz_eval, batch_size=B
        )
        doa_prediction, doa_all_predictions, roots = method_output

        return doa_prediction, doa_all_predictions, roots, Rz_eval
