"""TriFlow-MoE with a conservative snow, soil, and three-reservoir backbone."""

import math
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeteoFeatureLayout:
    """Consistent feature indexing for the five core meteorological inputs."""

    CORE_SIZE = 5

    SUPPORTED = {
        "P_SW_TMAX_TMIN_VP": {"p": 0, "sw": 1, "tmax": 2, "tmin": 3, "vp": 4},
        "P_TMAX_TMIN_VP_SW": {"p": 0, "tmax": 1, "tmin": 2, "vp": 3, "sw": 4},
    }

    def __init__(self, order: str = "P_SW_TMAX_TMIN_VP") -> None:
        if order not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported feature_order='{order}'. Supported orders: {list(self.SUPPORTED.keys())}"
            )
        self.order = order
        self.idx = self.SUPPORTED[order]

    def split(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        p = x[..., self.idx["p"] : self.idx["p"] + 1]
        sw = x[..., self.idx["sw"] : self.idx["sw"] + 1]
        tmax = x[..., self.idx["tmax"] : self.idx["tmax"] + 1]
        tmin = x[..., self.idx["tmin"] : self.idx["tmin"] + 1]
        vp = x[..., self.idx["vp"] : self.idx["vp"] + 1]
        t_avg = 0.5 * (tmax + tmin)
        return p, sw, tmax, tmin, vp, t_avg

    @staticmethod
    def build_other(
        sw: torch.Tensor, tmax: torch.Tensor, tmin: torch.Tensor, vp: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([sw, tmax, tmin, vp], dim=-1)


_METEO_ALIASES = {
    "P": ("prcp", "prec", "precip", "precipitation", "pr", "rain", "rainfall"),
    "SW": ("srad", "sw", "swdown", "shortwave", "solar", "rad"),
    "TMAX": ("tmax", "tmaxc", "maxtemp"),
    "TMIN": ("tmin", "tminc", "mintemp"),
    "VP": ("vp", "vapor", "vapour", "vpress"),
}


def _canonical_meteo(name: str) -> Optional[str]:
    token = re.split(r"[^a-z0-9]+", name.strip().lower())[0]
    for canon, aliases in _METEO_ALIASES.items():
        if token in aliases:
            return canon
    return None


def infer_feature_order(dynamic_input: Sequence[str]) -> str:
    """Map the forcing column names to the model’s positional layout."""
    core = list(dynamic_input)[: MeteoFeatureLayout.CORE_SIZE]
    if len(core) < MeteoFeatureLayout.CORE_SIZE:
        raise ValueError(
            f"TriFlowMoE needs at least {MeteoFeatureLayout.CORE_SIZE} meteorological "
            f"inputs; dynamic_input has {len(dynamic_input)}: {list(dynamic_input)}"
        )

    tokens: List[str] = []
    for name in core:
        canon = _canonical_meteo(name)
        if canon is None:
            raise ValueError(
                f"Cannot classify dynamic input '{name}' as one of "
                f"{sorted(_METEO_ALIASES)}. Set 'feature_order' explicitly, or extend "
                "_METEO_ALIASES in this module."
            )
        tokens.append(canon)

    if len(set(tokens)) != len(tokens):
        raise ValueError(
            f"Duplicate meteorological roles inferred from {core}: {tokens}"
        )

    order = "_".join(tokens)
    if order not in MeteoFeatureLayout.SUPPORTED:
        raise ValueError(
            f"dynamic_input implies feature_order='{order}', which TriFlowMoE does not "
            f"support. Reorder dynamic_input to match one of "
            f"{list(MeteoFeatureLayout.SUPPORTED)}."
        )
    return order


class VariableEmbedding(nn.Module):
    """Embed liquid water input separately from the other meteorological drivers."""

    def __init__(self, d_model: int = 64, aux_input_size: int = 0) -> None:
        super().__init__()
        self.d_p = max(1, d_model // 2)
        self.d_other = d_model - self.d_p
        self.aux_input_size = aux_input_size

        self.proj_p = nn.Linear(1, self.d_p)
        self.proj_other = nn.Linear(4, self.d_other)
        self.proj_fuse = nn.Linear(d_model, d_model)
        self.proj_aux = (
            nn.Linear(aux_input_size, d_model) if aux_input_size > 0 else None
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        p: torch.Tensor,
        other: torch.Tensor,
        x_aux: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_p = self.proj_p(p)
        h_other = self.proj_other(other)
        fused = torch.cat([h_p, h_other], dim=-1)
        e = self.proj_fuse(fused)

        if self.proj_aux is not None and x_aux is not None and x_aux.shape[-1] > 0:
            e = e + self.proj_aux(x_aux)

        e = self.norm(e)
        return e, h_p, h_other


class StaticFeatureEncoder(nn.Module):
    """Encode static basin attributes into a compact catchment representation."""

    def __init__(
        self,
        input_size: int,
        context_size: int,
        hidden_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, context_size),
            nn.ReLU(),
        )

    def forward(self, x_static: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_static is None:
            return None

        if x_static.dim() == 3:
            x_static = x_static[:, 0, :]
        elif x_static.dim() == 1:
            x_static = x_static.unsqueeze(0)

        return self.net(x_static)


class StaticConditioner(nn.Module):
    """Produce basin-conditioned feature modulation, slow-attention state, and gate biases."""

    def __init__(
        self,
        context_size: int,
        d_p: int,
        d_other: int,
        d_model: int,
        n_experts: int = 3,
    ) -> None:
        super().__init__()

        self.to_gamma_p = nn.Linear(context_size, d_p)
        self.to_beta_p = nn.Linear(context_size, d_p)
        self.to_gamma_other = nn.Linear(context_size, d_other)
        self.to_beta_other = nn.Linear(context_size, d_other)
        self.to_gamma_e = nn.Linear(context_size, d_model)
        self.to_beta_e = nn.Linear(context_size, d_model)

        self.to_baseflow_init = nn.Linear(context_size, 1)

        self.to_router_bias = nn.Linear(context_size, n_experts)

    @staticmethod
    def _bounded_scale(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def _bounded_shift(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.tanh(x)

    def forward(
        self, static_context: Optional[torch.Tensor]
    ) -> Dict[str, Optional[torch.Tensor]]:
        if static_context is None:
            return {
                "gamma_p": None,
                "beta_p": None,
                "gamma_other": None,
                "beta_other": None,
                "gamma_e": None,
                "beta_e": None,
                "baseflow_init": None,
                "router_bias": None,
            }

        return {
            "gamma_p": self._bounded_scale(self.to_gamma_p(static_context)),
            "beta_p": self._bounded_shift(self.to_beta_p(static_context)),
            "gamma_other": self._bounded_scale(self.to_gamma_other(static_context)),
            "beta_other": self._bounded_shift(self.to_beta_other(static_context)),
            "gamma_e": self._bounded_scale(self.to_gamma_e(static_context)),
            "beta_e": self._bounded_shift(self.to_beta_e(static_context)),
            "baseflow_init": F.softplus(self.to_baseflow_init(static_context)),
            "router_bias": self.to_router_bias(static_context),
        }


class FiLM(nn.Module):
    """Feature-wise linear modulation."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        gamma: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if gamma is None or beta is None:
            return x
        return (1.0 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)


def _causal_fir(p: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a causal finite impulse response filter with left zero padding."""
    k_len = kernel.shape[0]
    x = F.pad(p.unsqueeze(1), (k_len - 1, 0))
    w = kernel.flip(0).to(dtype=p.dtype).reshape(1, 1, k_len)
    return F.conv1d(x, w).squeeze(1)


def antecedent_index(
    p: torch.Tensor,
    rho_raw: torch.Tensor,
    window: int = 30,
    include_current: bool = True,
) -> torch.Tensor:
    """Compute the exponentially weighted antecedent precipitation index."""
    if p.dim() != 2:
        raise ValueError(f"p must have shape [B, T]; got {tuple(p.shape)}")
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")

    seq_len = p.shape[1]
    k_len = min(int(window), seq_len)
    rho = torch.sigmoid(rho_raw)

    exps = torch.arange(k_len, device=p.device, dtype=p.dtype)
    kernel = rho.to(p.dtype) ** exps
    if not include_current:
        # Exclude current-day precipitation without an in-place write.
        kernel = kernel * (exps > 0).to(p.dtype)

    return _causal_fir(p, kernel)


class QuickFlowExpert(nn.Module):
    """Fast causal attention with a temperature- and radiation-conditioned query."""

    def __init__(
        self,
        d_model: int = 64,
        d_k: int = 32,
        d_p: int = 32,
        window: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("QuickFlowExpert.window must be >= 1")
        self.d_k = d_k
        self.window = int(window)
        self.W_QP = nn.Linear(d_p, d_k, bias=False)
        self.gamma_T = nn.Parameter(torch.full((), -3.0))
        self.W_QT = nn.Linear(1, d_k, bias=False)
        self.gamma_SW = nn.Parameter(torch.full((), -3.0))
        self.W_QSW = nn.Linear(1, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V_free = nn.Linear(d_model, d_model, bias=False)

        self.W_V_P_INIT_BIAS = -4.0
        self.W_V_P_raw = nn.Parameter(
            torch.randn(d_model, 1) * 0.02 + self.W_V_P_INIT_BIAS
        )
        self.lambda1_raw = nn.Parameter(torch.tensor(0.5))
        self.drop = nn.Dropout(dropout)

        self.rate_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.rate_head.weight)
        nn.init.zeros_(self.rate_head.bias)

    def _v_monotone(self, e: torch.Tensor, p_std: torch.Tensor) -> torch.Tensor:

        w_p_nn = F.softplus(self.W_V_P_raw)
        return self.W_V_free(e) + (p_std @ w_p_nn.T)

    def forward(
        self,
        e: torch.Tensor,
        h_p: torch.Tensor,
        p_std: torch.Tensor,
        t_avg: torch.Tensor,
        sw: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the meteorological baseline logit, before state modulation."""
        _, seq_len, _ = e.shape
        device = e.device
        lambda1 = F.softplus(self.lambda1_raw)

        q = (
            self.W_QP(h_p)
            + torch.sigmoid(self.gamma_T) * self.W_QT(t_avg)
            + torch.sigmoid(self.gamma_SW) * self.W_QSW(sw)
        )
        k = self.W_K(e)
        v = self._v_monotone(e, p_std)

        idx = torch.arange(seq_len, device=device)
        delta = idx.unsqueeze(1) - idx.unsqueeze(0)
        causal_ok = (delta >= 0) & (delta <= (self.window - 1))
        hard_mask = torch.where(
            causal_ok,
            torch.zeros(seq_len, seq_len, device=device),
            torch.full((seq_len, seq_len), float("-inf"), device=device),
        )
        decay_bias = torch.where(
            causal_ok,
            -lambda1 * delta.float(),
            torch.zeros(seq_len, seq_len, device=device),
        )

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        scores = scores + hard_mask.unsqueeze(0) + decay_bias.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        ctx = self.drop(torch.bmm(attn, v))

        return self.rate_head(ctx).squeeze(-1)


class InterflowExpert(nn.Module):
    """Intermediate response specialist with a configurable delayed-peak prior."""

    def __init__(
        self,
        d_model: int = 64,
        d_k: int = 32,
        d_other: int = 32,
        window: int = 30,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("InterflowExpert.window must be >= 1")
        self.d_k = d_k
        self.window = int(window)
        self.alpha_raw = nn.Parameter(torch.tensor(1.5))
        self.beta_raw = nn.Parameter(torch.tensor(0.3))
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_V_other = nn.Linear(d_other, d_model, bias=False)
        self.phi_raw = nn.Parameter(torch.tensor([3.0, 2.0, 1.0]))
        self.drop = nn.Dropout(dropout)

        self.rate_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.rate_head.weight)
        nn.init.zeros_(self.rate_head.bias)

    def forward(
        self,
        e: torch.Tensor,
        h_other: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the intermediate-store outflow-rate logit, shape [B, T]."""
        _, seq_len, _ = e.shape
        device = e.device
        alpha = F.softplus(self.alpha_raw) + 1.0
        beta = F.softplus(self.beta_raw)

        q = self.W_Q(e)
        k = self.W_K(e)
        v = self.W_V(e) + self.W_V_other(h_other)

        idx = torch.arange(seq_len, device=device)
        delta = idx.unsqueeze(1) - idx.unsqueeze(0)
        valid = (delta >= 0) & (delta <= (self.window - 1))
        hard_mask = torch.where(
            valid,
            torch.zeros(seq_len, seq_len, device=device),
            torch.full((seq_len, seq_len), float("-inf"), device=device),
        )
        d_float = delta.float().clamp(min=0)
        gamma_bias = torch.where(
            valid,
            (alpha - 1) * torch.log(d_float + 1) - beta * d_float,
            torch.zeros(seq_len, seq_len, device=device),
        )

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        scores = scores + hard_mask.unsqueeze(0) + gamma_bias.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, v)

        phi = torch.softmax(self.phi_raw, dim=0)
        ctx0 = phi[0] * ctx
        ctx1 = (
            phi[1] * F.pad(ctx, (0, 0, 1, 0))[:, :seq_len]
            if seq_len > 1
            else torch.zeros_like(ctx)
        )
        ctx2 = (
            phi[2] * F.pad(ctx, (0, 0, 2, 0))[:, :seq_len]
            if seq_len > 2
            else torch.zeros_like(ctx)
        )
        ctx_smooth = self.drop(ctx0 + ctx1 + ctx2)

        return self.rate_head(ctx_smooth).squeeze(-1)


def _baseflow_f3_scan_impl(
    base: torch.Tensor,
    s_r: torch.Tensor,
    v_bar: torch.Tensor,
    k_bf: torch.Tensor,
    f3_init: torch.Tensor,
) -> torch.Tensor:
    """Resolve the scalar slow-expert attention state sequentially."""
    seq_len = base.size(1)
    f3 = f3_init
    lagged: List[torch.Tensor] = []
    for t in range(seq_len):
        lagged.append(f3)
        row = base[:, t, : t + 1] + f3.unsqueeze(1) * s_r[:, : t + 1]
        attn = torch.softmax(row, dim=1)
        drive = (attn * v_bar[:, : t + 1]).sum(dim=1)
        f3 = k_bf * f3 + (1.0 - k_bf) * drive
    return torch.stack(lagged, dim=1)


try:
    _baseflow_f3_scan = torch.jit.script(_baseflow_f3_scan_impl)
except Exception:
    _baseflow_f3_scan = _baseflow_f3_scan_impl


def _ltv_scan(
    x: torch.Tensor, c: torch.Tensor, y0: torch.Tensor, chunk: int = 64
) -> torch.Tensor:
    """Evaluate y[t] = c[t] * y[t-1] + x[t] by chunked cumulative products."""
    seq_len = x.shape[1]
    state = y0
    parts: List[torch.Tensor] = []
    for c0 in range(0, seq_len, chunk):
        xc = x[:, c0 : c0 + chunk]
        cc = c[:, c0 : c0 + chunk]
        length = xc.shape[1]
        lower = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        strict = lower.tril(-1)

        z = torch.where(
            strict, cc.unsqueeze(2), torch.ones((), dtype=x.dtype, device=x.device)
        )
        transfer = torch.cumprod(z, dim=1) * lower
        yc = torch.bmm(transfer, xc.unsqueeze(-1)).squeeze(-1) + torch.cumprod(
            cc, dim=1
        ) * state.unsqueeze(1)
        parts.append(yc)
        state = yc[:, -1]
    return torch.cat(parts, dim=1)


def _reverse_ltv_scan(g: torch.Tensor, jac: torch.Tensor) -> torch.Tensor:
    """Adjoint recursion a_t = g_t + jac_t * a_{t+1}, a_T = 0 (time-reversed _ltv_scan)."""
    return _ltv_scan(g.flip(1), jac.flip(1), g.new_zeros(g.shape[0])).flip(1)


class _F3ScanFunction(torch.autograd.Function):
    """Differentiate the slow-expert scalar recursion with a parallel adjoint."""

    @staticmethod
    def forward(ctx, base, s_r, v_bar, k_bf, f3_init):
        lagged = _baseflow_f3_scan(base, s_r, v_bar, k_bf, f3_init)
        ctx.save_for_backward(base, s_r, v_bar, k_bf, lagged)
        return lagged

    @staticmethod
    def backward(ctx, grad_lagged):
        base, s_r, v_bar, k_bf, f = ctx.saved_tensors
        seq_len = f.shape[1]
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=f.device).tril()
        scores = (base + f.unsqueeze(2) * s_r.unsqueeze(1)).masked_fill(
            ~causal, float("-inf")
        )
        attn = torch.softmax(scores, dim=2)
        del scores
        drive = torch.bmm(attn, v_bar.unsqueeze(2)).squeeze(2)
        mean_sr = torch.bmm(attn, s_r.unsqueeze(2)).squeeze(2)
        mean_vsr = torch.bmm(attn, (v_bar * s_r).unsqueeze(2)).squeeze(2)
        jac = k_bf + (1.0 - k_bf) * (mean_vsr - drive * mean_sr)
        lam = _reverse_ltv_scan(grad_lagged.double(), jac.double()).to(f.dtype)
        lam_next = torch.cat([lam[:, 1:], lam.new_zeros(lam.shape[0], 1)], dim=1)
        mu = (1.0 - k_bf) * lam_next
        grad_v = torch.bmm(mu.unsqueeze(1), attn).squeeze(1)

        d_scores = attn.mul_(mu.unsqueeze(2)).mul_(
            v_bar.unsqueeze(1) - drive.unsqueeze(2)
        )
        grad_sr = torch.bmm(f.unsqueeze(1), d_scores).squeeze(1)
        grad_k = (lam_next * (f - drive)).sum().reshape(k_bf.shape)
        return d_scores, grad_sr, grad_v, grad_k, lam[:, 0]


_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


def _baseflow_reservoir_scan(
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_r: torch.Tensor,
    decay_bias: torch.Tensor,
    inv_sqrt_dk: float,
    k_bf: torch.Tensor,
    f3_init: torch.Tensor,
    fast_backward: bool = True,
) -> torch.Tensor:
    """Resolve the lagged scalar state, then evaluate full-history causal attention."""
    seq_len = v.size(1)
    device = v.device

    base = torch.bmm(a, k.transpose(1, 2)) * inv_sqrt_dk + decay_bias.unsqueeze(0)
    s_r = torch.matmul(k, w_r) * inv_sqrt_dk
    v_bar = v.mean(dim=-1)

    # The first pass only reads keys on or before the current day.

    if fast_backward:
        f3_lagged = _F3ScanFunction.apply(base, s_r, v_bar, k_bf, f3_init)
    else:
        f3_lagged = _baseflow_f3_scan(base, s_r, v_bar, k_bf, f3_init)

    idx = torch.arange(seq_len, device=device)
    causal = (idx.unsqueeze(1) - idx.unsqueeze(0)) >= 0
    attn_bias = decay_bias.masked_fill(~causal, float("-inf"))

    # Insert the lagged scalar state into the query for batched attention.
    q = a + f3_lagged.unsqueeze(-1) * w_r

    if _HAS_SDPA:
        return F.scaled_dot_product_attention(
            q.unsqueeze(1),
            k.unsqueeze(1),
            v.unsqueeze(1),
            attn_mask=attn_bias,
            scale=inv_sqrt_dk,
        ).squeeze(1)

    scores = torch.bmm(q, k.transpose(1, 2)) * inv_sqrt_dk + attn_bias.unsqueeze(0)
    return torch.bmm(torch.softmax(scores, dim=-1), v)


class BaseflowExpert(nn.Module):
    """Attend to all available causal history with a lag penalty and a recurrent query state."""

    def __init__(
        self,
        d_model: int = 64,
        d_k: int = 32,
        dropout: float = 0.0,
        fast_scans: bool = True,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.fast_scans = bool(fast_scans)
        self.W_Q_e = nn.Linear(d_model, d_k, bias=False)
        self.W_Q_r = nn.Linear(1, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.lambda3_raw = nn.Parameter(torch.tensor(-2.0))
        self.w_AR = nn.Parameter(torch.tensor(2.0))

        self.drop = nn.Dropout(dropout)

        self.rate_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.rate_head.weight)
        nn.init.zeros_(self.rate_head.bias)

    def forward(
        self,
        e: torch.Tensor,
        init_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns the slow-store outflow-rate logit, shape [B, T]."""
        bsz, seq_len, _ = e.shape
        device = e.device
        lambda3 = F.softplus(self.lambda3_raw)
        k_bf = torch.sigmoid(self.w_AR)

        a = self.W_Q_e(e)
        k = self.W_K(e)
        v = self.W_V(e)
        # The lagged scalar state enters the query through a linear projection.
        w_r = self.W_Q_r.weight.squeeze(-1)

        idx = torch.arange(seq_len, device=device)
        delta = idx.unsqueeze(1) - idx.unsqueeze(0)
        decay_bias = torch.where(
            delta >= 0,
            -lambda3 * delta.float(),
            torch.zeros(seq_len, seq_len, device=device),
        )

        # Only the scalar attention state is recurrent.
        if init_state is None:
            f3_init = torch.zeros(bsz, device=device, dtype=e.dtype)
        else:
            f3_init = init_state.to(device=device, dtype=e.dtype).reshape(bsz)

        ctx = _baseflow_reservoir_scan(
            a,
            k,
            v,
            w_r,
            decay_bias,
            1.0 / math.sqrt(self.d_k),
            k_bf,
            f3_init,
            fast_backward=self.fast_scans,
        )

        return self.rate_head(self.drop(ctx)).squeeze(-1)


def _physical_series(
    value: Optional[torch.Tensor],
    name: str,
    batch_size: int,
    reference: torch.Tensor,
    length: Optional[int] = None,
    nonnegative: bool = False,
) -> torch.Tensor:
    """Validate shape, device, and finite values of a forcing series in physical units."""
    if value is None:
        raise ValueError(f"TriFlowMoE requires `{name}` in physical units")
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if (
        value.ndim != 2
        or value.shape[0] != batch_size
        or (length is not None and value.shape[1] != length)
    ):
        raise ValueError(
            f"{name} must have shape [B,T] (or [B,T,1]) with matching batch/time dimensions; "
            f"got {tuple(value.shape)}"
        )
    value = value.to(device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(
            f"{name} must be finite in physical units; missing forcing must be excluded by the dataset"
        )
    if nonnegative and bool((value < 0).any()):
        raise ValueError(
            f"{name} must be non-negative in physical units "
            f"(min={float(value.min()):.6g}); is it carrying standardized data?"
        )
    return value


SNOW_PARAMETER_NAMES: Tuple[str, ...] = (
    "rain_threshold_degC",
    "melt_threshold_degC",
    "ddf_max_mm_degC_d",
    "ddf_min_mm_degC_d",
    "elev_std_m",
    "precip_gradient_per_km",
    "thermal_inertia",
)


def _equal_area_band_offsets(n_bands: int) -> torch.Tensor:
    """Compute conditional standard-normal means for equal-probability elevation bands."""
    if n_bands == 1:
        return torch.zeros(1, dtype=torch.float64)
    prob = torch.arange(n_bands + 1, dtype=torch.float64) / n_bands
    edges = math.sqrt(2.0) * torch.erfinv(2.0 * prob - 1.0)
    pdf = torch.exp(-0.5 * edges**2) / math.sqrt(2.0 * math.pi)
    means = n_bands * (pdf[:-1] - pdf[1:])
    return 0.5 * (means - means.flip(0))


def _decaying_reflected_scan(
    x: torch.Tensor, log_a: torch.Tensor, chunk: int
) -> torch.Tensor:
    """Evaluate H[t] = max(0, a * H[t-1] + x[t]) with a reflected scan."""
    n_rows, length = x.shape
    state = x.new_zeros(n_rows)
    parts: List[torch.Tensor] = []
    for c0 in range(0, length, chunk):
        xc = x[:, c0 : c0 + chunk]
        steps = torch.arange(1, xc.shape[1] + 1, device=x.device, dtype=x.dtype)
        log_growth = steps.unsqueeze(0) * log_a.unsqueeze(1)
        z = state.unsqueeze(1) + torch.cumsum(xc * torch.exp(-log_growth), dim=1)
        floor = torch.cummin(z, dim=1).values.clamp(max=0.0)
        hc = (torch.exp(log_growth) * (z - floor)).clamp(min=0.0)
        parts.append(hc)
        state = hc[:, -1]
    return torch.cat(parts, dim=1)


class SharedSnowModule(nn.Module):
    """Partition rain and snow and advance SWE and thermal deficit across elevation bands."""

    N_PARAMS = len(SNOW_PARAMETER_NAMES)
    PARAMETER_NAMES = SNOW_PARAMETER_NAMES
    MELT_PHASE_DOY = 80.0
    DAYS_PER_YEAR = 365.25
    COVER_THRESHOLD_MM = 1.0

    RAW_LOWER = (-3.0, -3.0, 0.1, 0.05, 0.0, 0.0, 0.05)
    RAW_UPPER = (3.0, 3.0, 10.0, 1.0, 1000.0, 1.0, 0.98)

    def __init__(
        self,
        static_context_size: int = 0,
        transition_width: float = 1.0,
        initial_ddf: float = 3.0,
        n_bands: int = 5,
        seasonal_melt: bool = True,
        cold_content: bool = True,
        initial_ddf_ratio: float = 0.5,
        lapse_rate: float = 6.5,
        ripeness_scale: float = 1.0,
        initial_rain_threshold: float = 1.0,
        initial_elev_std: float = 200.0,
        initial_precip_gradient: float = 0.4,
        initial_thermal_inertia: float = 0.5,
    ) -> None:
        super().__init__()
        if not math.isfinite(transition_width) or transition_width <= 0:
            raise ValueError("snow_transition_width must be finite and > 0 degC")
        if not math.isfinite(initial_ddf) or not 0.1 < initial_ddf < 10.0:
            raise ValueError("snow_initial_ddf must be in (0.1, 10) mm/degC/day")
        if int(n_bands) != n_bands or not 1 <= int(n_bands) <= 20:
            raise ValueError("snow_n_bands must be an integer in [1, 20]")
        if not 0.05 < float(initial_ddf_ratio) < 1.0:
            raise ValueError("snow_initial_ddf_ratio must be in (0.05, 1)")
        if not math.isfinite(lapse_rate) or lapse_rate <= 0:
            raise ValueError("snow_lapse_rate must be finite and > 0 degC/km")
        if not math.isfinite(ripeness_scale) or ripeness_scale <= 0:
            raise ValueError("snow_ripeness_scale must be finite and > 0 degC")
        self.n_bands = int(n_bands)
        self.seasonal_melt = bool(seasonal_melt)
        self.cold_content = bool(cold_content)
        self.lapse_rate = float(lapse_rate)
        self.ripeness_scale = float(ripeness_scale)
        ratio0 = float(initial_ddf_ratio) if self.seasonal_melt else 0.5

        ddf_max0 = (
            2.0 * initial_ddf / (1.0 + ratio0)
            if self.seasonal_melt
            else float(initial_ddf)
        )
        initial = torch.tensor(
            [
                float(initial_rain_threshold),
                0.0,
                ddf_max0,
                ratio0,
                float(initial_elev_std),
                float(initial_precip_gradient),
                float(initial_thermal_inertia),
            ]
        )
        self.register_buffer("lower", torch.tensor(self.RAW_LOWER))
        self.register_buffer("upper", torch.tensor(self.RAW_UPPER))
        unit = (initial - self.lower) / (self.upper - self.lower)
        if not bool(((unit > 0.0) & (unit < 1.0)).all()):
            raise ValueError(
                f"snow initial values {initial.tolist()} must lie strictly inside "
                f"{list(zip(self.RAW_LOWER, self.RAW_UPPER))}"
            )
        self.register_buffer("transition_width", torch.tensor(float(transition_width)))
        self.register_buffer("band_offsets", _equal_area_band_offsets(self.n_bands))
        self.parameter_logits = nn.Parameter(torch.logit(unit))
        self.static_offsets = (
            nn.Linear(static_context_size, self.N_PARAMS, bias=False)
            if static_context_size > 0
            else None
        )
        if self.static_offsets is not None:
            nn.init.zeros_(self.static_offsets.weight)

        self.deficit_chunk = int(
            max(
                1,
                min(
                    128,
                    math.floor(100.0 * math.log(10.0) / -math.log(self.RAW_LOWER[6])),
                ),
            )
        )

    _series = staticmethod(_physical_series)

    def physical_parameters(
        self, batch_size: int, static_context: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """[B, 7] parameters in force, ordered as SNOW_PARAMETER_NAMES."""
        logits = self.parameter_logits.unsqueeze(0).expand(batch_size, -1)
        if self.static_offsets is not None and static_context is not None:
            logits = logits + self.static_offsets(static_context)
        raw = self.lower + (self.upper - self.lower) * torch.sigmoid(logits)
        t_rain, t_melt, ddf_max, ratio, elev_std, grad, inertia = raw.unbind(dim=1)
        ddf_min = ratio * ddf_max if self.seasonal_melt else ddf_max
        if self.n_bands == 1:
            elev_std, grad = torch.zeros_like(elev_std), torch.zeros_like(grad)
        if not self.cold_content:
            inertia = torch.zeros_like(inertia)
        return torch.stack(
            [t_rain, t_melt, ddf_max, ddf_min, elev_std, grad, inertia], dim=1
        )

    def forward(
        self,
        p_raw: torch.Tensor,
        t_raw: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
        warmup_p: Optional[torch.Tensor] = None,
        warmup_t: Optional[torch.Tensor] = None,
        warmup_mask: Optional[torch.Tensor] = None,
        initial_swe: Optional[torch.Tensor] = None,
        doy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if p_raw.ndim == 3 and p_raw.shape[-1] == 1:
            p_raw = p_raw.squeeze(-1)
        if p_raw.ndim != 2:
            raise ValueError("p_raw must have shape [B,T] or [B,T,1]")
        batch_size, seq_len = p_raw.shape
        p_raw = self._series(
            p_raw, "p_raw", batch_size, p_raw, seq_len, nonnegative=True
        )
        t_raw = self._series(t_raw, "t_raw", batch_size, p_raw, seq_len)
        if (warmup_p is None) != (warmup_t is None):
            raise ValueError(
                "snow_warmup_p and snow_warmup_t must be supplied together"
            )
        warmup_len = 0
        valid = torch.ones_like(p_raw, dtype=torch.bool)
        p_all, t_all = p_raw, t_raw
        if warmup_p is not None:
            warmup_p = self._series(
                warmup_p, "snow_warmup_p", batch_size, p_raw, nonnegative=True
            )
            warmup_len = warmup_p.shape[1]
            warmup_t = self._series(
                warmup_t, "snow_warmup_t", batch_size, p_raw, warmup_len
            )
            if warmup_mask is None:
                warmup_mask = torch.ones_like(warmup_p, dtype=torch.bool)
            elif warmup_mask.dtype != torch.bool or warmup_mask.shape != warmup_p.shape:
                raise ValueError("snow_warmup_mask must be bool [B,W]")
            warmup_mask = warmup_mask.to(device=p_raw.device)

            # The history mask may only exclude leading days.
            if bool((warmup_mask[:, :-1] & ~warmup_mask[:, 1:]).any()):
                raise ValueError("snow_warmup_mask may contain leading padding only")
            p_all = torch.cat([warmup_p, p_raw], dim=1)
            t_all = torch.cat([warmup_t, t_raw], dim=1)
            valid = torch.cat([warmup_mask, valid], dim=1)
        elif warmup_mask is not None:
            raise ValueError("snow_warmup_mask requires warmup meteorology")
        total_len = warmup_len + seq_len
        n_bands = self.n_bands
        f64 = torch.float64

        if self.seasonal_melt:
            if doy is None:
                raise ValueError(
                    "snow_seasonal_melt=True requires the day of year of the window (doy)"
                )
            if doy.ndim == 3 and doy.shape[-1] == 1:
                doy = doy.squeeze(-1)
            if doy.ndim != 2 or tuple(doy.shape) != (batch_size, seq_len):
                raise ValueError(
                    f"doy must have shape [B,T]=({batch_size},{seq_len}); got {tuple(doy.shape)}"
                )
            doy = doy.to(device=p_raw.device, dtype=f64)
            if not bool(torch.isfinite(doy).all()):
                raise ValueError("doy must be finite")

        if initial_swe is None:
            swe_init = torch.zeros(batch_size, n_bands, device=p_raw.device, dtype=f64)
        else:
            swe_init = (
                initial_swe.unsqueeze(-1) if initial_swe.ndim == 1 else initial_swe
            )
            if (
                swe_init.ndim != 2
                or swe_init.shape[0] != batch_size
                or swe_init.shape[1] not in (1, n_bands)
            ):
                raise ValueError(
                    f"initial_swe must be [B], [B,1] or [B,{n_bands}] (per band)"
                )
            swe_init = swe_init.to(device=p_raw.device, dtype=f64)
            if not bool(torch.isfinite(swe_init).all()) or bool((swe_init < 0).any()):
                raise ValueError("initial_swe must be finite and non-negative (mm)")
            swe_init = swe_init.expand(batch_size, n_bands)

        params = self.physical_parameters(batch_size, static_context)
        prm = params.to(f64)
        t_rain, t_melt = prm[:, 0:1].unsqueeze(1), prm[:, 1:2].unsqueeze(1)
        ddf_max, ddf_min = prm[:, 2:3], prm[:, 3:4]
        dz = prm[:, 4:5] * self.band_offsets.to(p_raw.device).unsqueeze(0)

        p64, t64 = p_all.to(f64), t_all.to(f64)
        valid3 = valid.unsqueeze(1)
        all_valid = bool(valid.all())

        def masked(v: torch.Tensor) -> torch.Tensor:
            return v if all_valid else torch.where(valid3, v, 0.0)

        t_band = t64.unsqueeze(1) - (self.lapse_rate / 1000.0) * dz.unsqueeze(2)
        if n_bands > 1:
            weight = torch.softmax(prm[:, 5:6] / 1000.0 * dz, dim=1) * n_bands
            p_band = p64.unsqueeze(1) * weight.unsqueeze(2)
        else:
            weight = torch.ones(batch_size, 1, device=p_raw.device, dtype=f64)
            p_band = p64.unsqueeze(1)
        rain_fraction = torch.sigmoid((t_band - t_rain) / self.transition_width.to(f64))
        rain = masked(rain_fraction * p_band)
        snow = masked(p_band - rain)

        if self.seasonal_melt:
            day = doy[:, :1] + torch.arange(
                -warmup_len, seq_len, device=p_raw.device, dtype=f64
            )
            season = torch.sin(
                2.0 * math.pi * (day - self.MELT_PHASE_DOY) / self.DAYS_PER_YEAR
            )
            ddf = 0.5 * (ddf_max + ddf_min) + 0.5 * (ddf_max - ddf_min) * season
        else:
            ddf = ddf_max.expand(batch_size, total_len)
        t_excess = t_band - t_melt
        potential = ddf.unsqueeze(1) * F.relu(t_excess)
        if self.cold_content:
            inertia = prm[:, 6]

            drive = masked((inertia - 1.0).view(-1, 1, 1) * t_excess)
            deficit = _decaying_reflected_scan(
                drive.reshape(batch_size * n_bands, total_len),
                torch.log(inertia).repeat_interleave(n_bands),
                self.deficit_chunk,
            ).reshape(batch_size, n_bands, total_len)
            ripeness = torch.exp(-deficit / self.ripeness_scale)
            potential = potential * ripeness
        else:
            deficit = torch.zeros_like(t_band)
            ripeness = torch.ones_like(t_band)
        potential = masked(potential)

        z = swe_init.unsqueeze(2) + torch.cumsum(snow - potential, dim=2)
        floor = torch.cummin(z, dim=2).values.clamp(max=0.0)
        swe = (z - floor).clamp(min=0.0)
        swe_before = torch.cat([swe_init.unsqueeze(2), swe[..., :-1]], dim=2)

        melt = torch.minimum(
            (swe_before + snow - swe).clamp(min=0.0),
            torch.minimum(potential, swe_before + snow),
        )
        liquid = (rain + melt).mean(dim=1)

        covered = (swe >= self.COVER_THRESHOLD_MM).to(f64).mean(dim=1)
        covered_before = (swe_before >= self.COVER_THRESHOLD_MM).to(f64).mean(dim=1)
        sl = slice(warmup_len, total_len)
        outputs = {
            "rainfall": rain.mean(dim=1)[:, sl],
            "snowfall": snow.mean(dim=1)[:, sl],
            "snowmelt": melt.mean(dim=1)[:, sl],
            "swe_before": swe_before.mean(dim=1)[:, sl],
            "swe": swe.mean(dim=1)[:, sl],
            "liquid_input": liquid[:, sl],
            "rain_fraction": (weight.unsqueeze(2) * rain_fraction).mean(dim=1)[:, sl],
            "t_raw": t_raw,
            "snow_balance_error": (swe - swe_before - snow + melt).mean(dim=1)[:, sl],
            "snow_cover_fraction": covered[:, sl],
            "snow_cover_before": covered_before[:, sl],
            "snow_thermal_deficit": deficit.mean(dim=1)[:, sl],
            "snow_ripeness": ripeness.mean(dim=1)[:, sl],
            "snow_ddf": ddf[:, sl],
            "swe_bands": swe[..., sl].transpose(1, 2),
            "snow_band_offsets_m": dz,
            "snow_parameters": params,
            "snow_warmup_days": valid[:, :warmup_len].sum(dim=1),
            "liquid_input_history": liquid[:, :warmup_len],
        }
        result = {key: value.to(dtype=p_raw.dtype) for key, value in outputs.items()}
        result["history_mask"] = valid[:, :warmup_len]

        result["swe_window_start"] = swe_before[:, :, warmup_len].mean(dim=1)
        result["swe_window_end"] = swe[:, :, -1].mean(dim=1)
        if not self.training:
            result["swe_target_start"] = swe_before[:, :, -1].mean(dim=1)
            result["swe_target_end"] = swe[:, :, -1].mean(dim=1)
        return result


def _water_ledger_scan_impl(
    inflow: torch.Tensor,
    et_ceiling: torch.Tensor,
    gen_logit: torch.Tensor,
    partition: torch.Tensor,
    kappa: torch.Tensor,
    s_cap: torch.Tensor,
    et_threshold: torch.Tensor,
    w_wet: torch.Tensor,
    b_wet: torch.Tensor,
    soil_init: torch.Tensor,
    cap_tau: float,
    beta_tau: float,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Advance the soil and three response stores with a daily conservative update."""
    seq_len = inflow.size(1)
    soil = soil_init
    stores = torch.zeros_like(partition[:, 0, :])
    aet_steps: List[torch.Tensor] = []
    recharge_steps: List[torch.Tensor] = []
    fraction_steps: List[torch.Tensor] = []
    soil_steps: List[torch.Tensor] = []
    outflow_steps: List[torch.Tensor] = []
    store_steps: List[torch.Tensor] = []
    for t in range(seq_len):
        water = soil + inflow[:, t]
        relative = water / (s_cap * et_threshold)
        beta = F.relu(1.0 - beta_tau * F.softplus((1.0 - relative) / beta_tau))
        demand = et_ceiling[:, t] * beta
        aet = -water * torch.expm1(-demand / (water + 1e-6))
        remaining = water - aet
        kept = F.relu(s_cap - cap_tau * F.softplus((s_cap - remaining) / cap_tau))
        kept = torch.minimum(kept, remaining)
        excess = remaining - kept
        fraction = torch.sigmoid(gen_logit[:, t] + w_wet * (kept / s_cap) + b_wet)
        drainage = fraction * kept
        recharge = excess + drainage
        soil = kept - drainage
        volume = stores + partition[:, t, :] * recharge.unsqueeze(1)
        outflow = kappa[:, t, :] * volume
        stores = volume - outflow
        aet_steps.append(aet)
        recharge_steps.append(recharge)
        fraction_steps.append(fraction)
        soil_steps.append(soil)
        outflow_steps.append(outflow)
        store_steps.append(stores)
    return (
        torch.stack(aet_steps, dim=1),
        torch.stack(recharge_steps, dim=1),
        torch.stack(fraction_steps, dim=1),
        torch.stack(soil_steps, dim=1),
        torch.stack(outflow_steps, dim=1),
        torch.stack(store_steps, dim=1),
    )


try:
    _water_ledger_scan = torch.jit.script(_water_ledger_scan_impl)
except Exception:
    _water_ledger_scan = _water_ledger_scan_impl


def _soil_step_impl(
    soil_prev: torch.Tensor,
    inflow: torch.Tensor,
    et_ceiling: torch.Tensor,
    gen_logit: torch.Tensor,
    s_cap: torch.Tensor,
    et_threshold: torch.Tensor,
    w_wet: torch.Tensor,
    b_wet: torch.Tensor,
    cap_tau: float,
    beta_tau: float,
    valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance soil water and compute evapotranspiration and recharge for one day."""
    water = soil_prev + inflow
    relative = water / (s_cap * et_threshold)
    beta = F.relu(1.0 - beta_tau * F.softplus((1.0 - relative) / beta_tau))
    demand = et_ceiling * beta
    aet = -water * torch.expm1(-demand / (water + 1e-6))
    remaining = water - aet
    kept = F.relu(s_cap - cap_tau * F.softplus((s_cap - remaining) / cap_tau))
    kept = torch.minimum(kept, remaining)
    excess = remaining - kept
    fraction = torch.sigmoid(gen_logit + w_wet * (kept / s_cap) + b_wet)
    drainage = fraction * kept
    recharge = excess + drainage
    soil = kept - drainage
    if valid is not None:
        aet = torch.where(valid, aet, torch.zeros_like(aet))
        recharge = torch.where(valid, recharge, torch.zeros_like(recharge))
        fraction = torch.where(valid, fraction, torch.zeros_like(fraction))
        soil = torch.where(valid, soil, soil_prev)
    return aet, recharge, fraction, soil


def _soil_prev_loop_impl(
    inflow: torch.Tensor,
    et_ceiling: torch.Tensor,
    gen_logit: torch.Tensor,
    s_cap: torch.Tensor,
    et_threshold: torch.Tensor,
    w_wet: torch.Tensor,
    b_wet: torch.Tensor,
    soil_init: torch.Tensor,
    cap_tau: float,
    beta_tau: float,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sequential soil recursion; returns only the LAGGED trajectory S_s,t-1 [B,T]."""
    seq_len = inflow.size(1)
    soil = soil_init
    prev: List[torch.Tensor] = []
    for t in range(seq_len):
        prev.append(soil)
        _, _, _, soil = _soil_step_impl(
            soil,
            inflow[:, t],
            et_ceiling[:, t],
            gen_logit[:, t],
            s_cap,
            et_threshold,
            w_wet,
            b_wet,
            cap_tau,
            beta_tau,
            valid[:, t] if valid is not None else None,
        )
    return torch.stack(prev, dim=1)


try:
    _soil_prev_loop = torch.jit.script(_soil_prev_loop_impl)
except Exception:
    _soil_prev_loop = _soil_prev_loop_impl


class _SoilTrajectoryFunction(torch.autograd.Function):
    """Use an exact adjoint for the lagged soil-water trajectory."""

    @staticmethod
    def forward(
        ctx,
        inflow,
        et_ceiling,
        gen_logit,
        s_cap,
        et_threshold,
        w_wet,
        b_wet,
        soil_init,
        cap_tau: float,
        beta_tau: float,
        valid,
    ):
        soil_prev = _soil_prev_loop(
            inflow,
            et_ceiling,
            gen_logit,
            s_cap,
            et_threshold,
            w_wet,
            b_wet,
            soil_init,
            cap_tau,
            beta_tau,
            valid,
        )
        ctx.save_for_backward(
            inflow,
            et_ceiling,
            gen_logit,
            s_cap,
            et_threshold,
            w_wet,
            b_wet,
            soil_prev,
            valid,
        )
        ctx.taus = (cap_tau, beta_tau)
        return soil_prev

    @staticmethod
    def backward(ctx, grad_prev):
        *inputs, soil_prev, valid = ctx.saved_tensors
        cap_tau, beta_tau = ctx.taus
        needs = ctx.needs_input_grad[:7]
        with torch.enable_grad():
            s_prev = soil_prev.detach().requires_grad_(True)
            leaves = [x.detach().requires_grad_(bool(n)) for x, n in zip(inputs, needs)]
            inflow, et_ceiling, gen_logit, s_cap, et_threshold, w_wet, b_wet = leaves
            soil_next = _soil_step_impl(
                s_prev,
                inflow,
                et_ceiling,
                gen_logit,
                s_cap.unsqueeze(1),
                et_threshold.unsqueeze(1),
                w_wet.unsqueeze(1),
                b_wet.unsqueeze(1),
                cap_tau,
                beta_tau,
                valid,
            )[3]
            (jac,) = torch.autograd.grad(
                soil_next, s_prev, torch.ones_like(soil_next), retain_graph=True
            )
            adj = _reverse_ltv_scan(grad_prev, jac)
            adj_next = torch.cat([adj[:, 1:], adj.new_zeros(adj.shape[0], 1)], dim=1)
            wanted = [x for x, n in zip(leaves, needs) if n]
            got = (
                list(torch.autograd.grad(soil_next, wanted, adj_next)) if wanted else []
            )
        grads: List[Optional[torch.Tensor]] = [got.pop(0) if n else None for n in needs]
        return (*grads, adj[:, 0], None, None, None)


class QuickStorageResponse(nn.Module):
    """Parameterize the fast reservoir’s storage-dependent outflow law."""

    def __init__(
        self,
        static_context_size: int = 0,
        weight_max: float = 4.0,
        weight_init: float = 1.0,
        storage_scale_bounds: Sequence[float] = (1.0, 100.0),
    ) -> None:
        super().__init__()
        lo, hi = (float(v) for v in storage_scale_bounds)
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
            raise ValueError(
                "quick_storage_scale_bounds must be finite 0 < low < high in mm"
            )
        if not math.isfinite(weight_max) or not 0.0 < weight_max <= 4.0:
            raise ValueError("quick_state_weight_max must be finite and in (0,4]")
        if not math.isfinite(weight_init) or not 0.0 < weight_init < weight_max:
            raise ValueError("quick_state_weight_init must satisfy 0 < init < max")
        self.weight_max = float(weight_max)
        start = math.log(weight_init / (weight_max - weight_init))
        self.param_logits = nn.Parameter(torch.tensor([start, 0.0]))
        self.register_buffer(
            "log_scale_bounds", torch.tensor([math.log(lo), math.log(hi)])
        )
        self.static_offsets = (
            nn.Linear(static_context_size, 2, bias=False)
            if static_context_size > 0
            else None
        )
        if self.static_offsets is not None:
            nn.init.zeros_(self.static_offsets.weight)

    def basin_parameters(
        self,
        batch_size: int,
        static_context: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = self.param_logits.unsqueeze(0).expand(batch_size, -1)
        if self.static_offsets is not None and static_context is not None:
            theta = theta + self.static_offsets(static_context)
        sig = torch.sigmoid(theta.to(dtype))
        bounds = self.log_scale_bounds.to(dtype)
        return (
            self.weight_max * sig[:, 0],
            torch.exp(bounds[0] + (bounds[1] - bounds[0]) * sig[:, 1]),
        )


def _quick_rate_impl(
    volume: torch.Tensor,
    meteo_logit: torch.Tensor,
    store_weight: torch.Tensor,
    scale: torch.Tensor,
    log_lo: torch.Tensor,
    log_hi: torch.Tensor,
) -> torch.Tensor:
    """Combine bounded meteorological modulation with monotone storage feedback."""
    loading = volume / (volume + scale)
    effective_logit = meteo_logit + store_weight * (loading - 0.5)
    return torch.exp(log_lo + (log_hi - log_lo) * torch.sigmoid(effective_logit))


def _quick_step_impl(
    previous: torch.Tensor,
    incoming: torch.Tensor,
    meteo_logit: torch.Tensor,
    store_weight: torch.Tensor,
    scale: torch.Tensor,
    log_lo: torch.Tensor,
    log_hi: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    volume = previous + incoming
    kappa = _quick_rate_impl(volume, meteo_logit, store_weight, scale, log_lo, log_hi)
    kappa = torch.where(valid, kappa, torch.zeros_like(kappa))
    outflow = kappa * volume
    return kappa, outflow, volume - outflow


def _quick_prev_loop_impl(
    incoming: torch.Tensor,
    meteo_logit: torch.Tensor,
    store_weight: torch.Tensor,
    scale: torch.Tensor,
    log_lo: torch.Tensor,
    log_hi: torch.Tensor,
    initial: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Only scalar-per-basin quick storage is recurrent; attention stays parallel."""
    state = initial
    previous: List[torch.Tensor] = []
    for t in range(incoming.shape[1]):
        previous.append(state)
        state = _quick_step_impl(
            state,
            incoming[:, t],
            meteo_logit[:, t],
            store_weight,
            scale,
            log_lo,
            log_hi,
            valid[:, t],
        )[2]
    return torch.stack(previous, dim=1)


try:
    _quick_prev_loop = torch.jit.script(_quick_prev_loop_impl)
except Exception:
    _quick_prev_loop = _quick_prev_loop_impl


class _QuickTrajectoryFunction(torch.autograd.Function):
    """Use an exact adjoint for the nonlinear fast reservoir recurrence."""

    @staticmethod
    def forward(
        ctx, incoming, meteo_logit, store_weight, scale, log_lo, log_hi, initial, valid
    ):
        previous = _quick_prev_loop(
            incoming, meteo_logit, store_weight, scale, log_lo, log_hi, initial, valid
        )
        ctx.save_for_backward(
            incoming, meteo_logit, store_weight, scale, log_lo, log_hi, previous, valid
        )
        return previous

    @staticmethod
    def backward(ctx, grad_previous):
        *inputs, previous, valid = ctx.saved_tensors
        needs = ctx.needs_input_grad[:6]
        with torch.enable_grad():
            state = previous.detach().requires_grad_(True)
            leaves = [x.detach().requires_grad_(bool(n)) for x, n in zip(inputs, needs)]
            incoming, meteo_logit, store_weight, scale, log_lo, log_hi = leaves
            next_state = _quick_step_impl(
                state,
                incoming,
                meteo_logit,
                store_weight.unsqueeze(1),
                scale.unsqueeze(1),
                log_lo,
                log_hi,
                valid,
            )[2]
            (jac,) = torch.autograd.grad(
                next_state, state, torch.ones_like(next_state), retain_graph=True
            )
            adj = _reverse_ltv_scan(grad_previous, jac)
            adj_next = torch.cat([adj[:, 1:], adj.new_zeros(adj.shape[0], 1)], dim=1)
            wanted = [x for x, n in zip(leaves, needs) if n]
            got = (
                list(torch.autograd.grad(next_state, wanted, adj_next))
                if wanted
                else []
            )
        grads: List[Optional[torch.Tensor]] = [got.pop(0) if n else None for n in needs]
        return (*grads, adj[:, 0], None)


def _decay_scan(u: torch.Tensor, log_keep: torch.Tensor, chunk: int) -> torch.Tensor:
    """Evaluate S[t] = keep[t] * (S[t-1] + inflow[t]) for a linear store."""
    n_rows, seq_len = u.shape
    state = u.new_zeros(n_rows)
    parts: List[torch.Tensor] = []
    for c0 in range(0, seq_len, chunk):
        uc = u[:, c0 : c0 + chunk]
        incl = torch.cumsum(log_keep[:, c0 : c0 + chunk], dim=1)
        excl = torch.cat([incl.new_zeros(n_rows, 1), incl[:, :-1]], dim=1)
        sc = torch.exp(incl) * (
            state.unsqueeze(1) + torch.cumsum(uc * torch.exp(-excl), dim=1)
        )
        parts.append(sc)
        state = sc[:, -1]
    return torch.cat(parts, dim=1)


class WaterLedger(nn.Module):
    """Route liquid input through soil and response storage with explicit water conservation."""

    N_STORES = 3
    N_PARAMS = 8
    DEFAULT_KAPPA_BOUNDS = ((0.25, 0.95), (0.03, 0.25), (0.0005, 0.03))

    def __init__(
        self,
        static_context_size: int = 0,
        kappa_bounds: Sequence[Sequence[float]] = DEFAULT_KAPPA_BOUNDS,
        soil_capacity_bounds: Sequence[float] = (25.0, 600.0),
        pet_scale_bounds: Sequence[float] = (0.5, 1.5),
        et_threshold_bounds: Sequence[float] = (0.3, 1.0),
        soil_init_fraction: float = 0.5,
        cap_tau: float = 1.0,
        use_float64: bool = True,
        fast_scans: bool = True,
        quick_state_dependent: bool = True,
        quick_state_weight_max: float = 4.0,
        quick_state_weight_init: float = 1.0,
        quick_storage_scale_bounds: Sequence[float] = (1.0, 100.0),
        quick_meteo_logit_limit: float = 0.5,
    ) -> None:
        super().__init__()
        kappa_bounds = [tuple(float(v) for v in pair) for pair in kappa_bounds]
        if len(kappa_bounds) != self.N_STORES or any(len(p) != 2 for p in kappa_bounds):
            raise ValueError(
                "ledger_kappa_bounds must be three (low, high) pairs: quick, inter, base"
            )
        for name, (lo, hi) in zip(("quick", "inter", "base"), kappa_bounds):
            if not 0.0 < lo < hi < 1.0:
                raise ValueError(
                    f"kappa bounds for the {name} store must satisfy 0 < low < high < 1"
                )
        for (lo_fast, _), (_, hi_slow) in zip(kappa_bounds[:-1], kappa_bounds[1:]):
            if hi_slow > lo_fast:
                warnings.warn(
                    "WaterLedger: overlapping kappa bands; the three stores are then no "
                    "longer separated by timescale and their labels become arbitrary.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        s_lo, s_hi = (float(v) for v in soil_capacity_bounds)
        c_lo, c_hi = (float(v) for v in pet_scale_bounds)
        e_lo, e_hi = (float(v) for v in et_threshold_bounds)
        if not 0.0 < s_lo < s_hi:
            raise ValueError("ledger_soil_capacity_bounds must satisfy 0 < low < high")
        if not 0.0 < c_lo < c_hi:
            raise ValueError("ledger_pet_scale_bounds must satisfy 0 < low < high")
        if not 0.0 < e_lo < e_hi <= 1.0:
            raise ValueError(
                "ledger_et_threshold_bounds must satisfy 0 < low < high <= 1"
            )
        if not 0.0 <= float(soil_init_fraction) <= 1.0:
            raise ValueError("ledger_soil_init_fraction must be in [0, 1]")
        if not float(cap_tau) > 0.0:
            raise ValueError("ledger_cap_tau must be > 0 mm")

        self.register_buffer(
            "log_kappa_lo", torch.log(torch.tensor([b[0] for b in kappa_bounds]))
        )
        self.register_buffer(
            "log_kappa_hi", torch.log(torch.tensor([b[1] for b in kappa_bounds]))
        )
        self.register_buffer("log_s_cap_bounds", torch.log(torch.tensor([s_lo, s_hi])))
        self.register_buffer("pet_scale_bounds", torch.tensor([c_lo, c_hi]))
        self.register_buffer("et_threshold_bounds", torch.tensor([e_lo, e_hi]))
        self.soil_init_fraction = float(soil_init_fraction)
        self.cap_tau = float(cap_tau)
        self.beta_tau = 0.05
        self.compute_dtype = torch.float64 if use_float64 else torch.float32

        self.fast_scans = bool(fast_scans)
        # Bound the scan chunk to keep reciprocal cumulative products finite.

        max_log_keep = -math.log1p(-max(b[1] for b in kappa_bounds))
        half_range = 0.5 * math.log(torch.finfo(torch.float64).max)
        self.store_scan_chunk = int(
            max(1, min(64, math.floor(half_range / max_log_keep)))
        )

        self.param_logits = nn.Parameter(
            torch.tensor([0.0, -3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        self.static_offsets = (
            nn.Linear(static_context_size, self.N_PARAMS, bias=False)
            if static_context_size > 0
            else None
        )
        if self.static_offsets is not None:
            nn.init.zeros_(self.static_offsets.weight)
        self.quick_state_dependent = bool(quick_state_dependent)
        self.quick_meteo_logit_limit = float(quick_meteo_logit_limit)
        if (
            not math.isfinite(self.quick_meteo_logit_limit)
            or self.quick_meteo_logit_limit < 0.0
        ):
            raise ValueError("quick_meteo_logit_limit must be finite and >=0")

        self.quick_response = (
            QuickStorageResponse(
                static_context_size,
                quick_state_weight_max,
                quick_state_weight_init,
                quick_storage_scale_bounds,
            )
            if self.quick_state_dependent
            else None
        )

    def basin_parameters(
        self, batch_size: int, static_context: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        theta = self.param_logits.unsqueeze(0).expand(batch_size, -1)
        if self.static_offsets is not None and static_context is not None:
            theta = theta + self.static_offsets(static_context)
        theta = theta.to(self.compute_dtype)
        sig = torch.sigmoid(theta)
        log_s = self.log_s_cap_bounds.to(self.compute_dtype)
        cb = self.pet_scale_bounds.to(self.compute_dtype)
        eb = self.et_threshold_bounds.to(self.compute_dtype)
        return {
            "w_wet": theta[:, 0],
            "b_wet": theta[:, 1],
            "s_cap": torch.exp(log_s[0] + (log_s[1] - log_s[0]) * sig[:, 2]),
            "pet_scale": cb[0] + (cb[1] - cb[0]) * sig[:, 3],
            "et_threshold": eb[0] + (eb[1] - eb[0]) * sig[:, 4],
            "kappa_bias": theta[:, 5:8],
        }

    def _fast_ledger(
        self,
        inflow: torch.Tensor,
        et_ceiling: torch.Tensor,
        gen_logit: torch.Tensor,
        part: torch.Tensor,
        kappa: torch.Tensor,
        prm: Dict[str, torch.Tensor],
        soil_init: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Resolve nonlinear soil storage before routing recharge through the linear stores."""
        batch_size, seq_len = inflow.shape
        s_cap, et_thr, w_wet, b_wet = (
            prm["s_cap"],
            prm["et_threshold"],
            prm["w_wet"],
            prm["b_wet"],
        )

        soil_prev = _SoilTrajectoryFunction.apply(
            inflow,
            et_ceiling,
            gen_logit,
            s_cap,
            et_thr,
            w_wet,
            b_wet,
            soil_init,
            self.cap_tau,
            self.beta_tau,
            None,
        )

        aet, recharge, drain_frac, soil = _soil_step_impl(
            soil_prev,
            inflow,
            et_ceiling,
            gen_logit,
            s_cap.unsqueeze(1),
            et_thr.unsqueeze(1),
            w_wet.unsqueeze(1),
            b_wet.unsqueeze(1),
            self.cap_tau,
            self.beta_tau,
        )

        inflow_k = part * recharge.unsqueeze(-1)
        u64 = inflow_k.double()

        def rows(v: torch.Tensor) -> torch.Tensor:
            return v.transpose(1, 2).reshape(batch_size * self.N_STORES, seq_len)

        s_scan = _decay_scan(
            rows(u64), rows(torch.log1p(-kappa.double())), self.store_scan_chunk
        )
        s_scan = s_scan.reshape(batch_size, self.N_STORES, seq_len).transpose(1, 2)
        volume = (
            torch.cat([torch.zeros_like(s_scan[:, :1]), s_scan[:, :-1]], dim=1) + u64
        )

        s_zero = torch.minimum(s_scan, volume)
        q_zero = volume - s_zero
        dt = inflow.dtype
        return aet, recharge, drain_frac, soil, q_zero.to(dt), s_zero.to(dt)

    def _state_soil_fluxes(self, inflow, et_ceiling, gen_logit, prm, soil_init, valid):
        args = (
            inflow,
            et_ceiling,
            gen_logit,
            prm["s_cap"],
            prm["et_threshold"],
            prm["w_wet"],
            prm["b_wet"],
            soil_init,
            self.cap_tau,
            self.beta_tau,
            valid,
        )
        soil_prev = (
            _SoilTrajectoryFunction.apply(*args)
            if self.fast_scans
            else _soil_prev_loop(*args)
        )
        fluxes = _soil_step_impl(
            soil_prev,
            inflow,
            et_ceiling,
            gen_logit,
            prm["s_cap"].unsqueeze(1),
            prm["et_threshold"].unsqueeze(1),
            prm["w_wet"].unsqueeze(1),
            prm["b_wet"].unsqueeze(1),
            self.cap_tau,
            self.beta_tau,
            valid,
        )
        return fluxes

    def _state_store_route(
        self, incoming, meteo_quick, kappa_meteo, store_init, quick_prm, valid
    ):
        """Nonlinear quick recurrence + unchanged linear inter/base recurrences."""
        store_weight, scale = quick_prm
        lo, hi = (
            self.log_kappa_lo[0].to(incoming.dtype),
            self.log_kappa_hi[0].to(incoming.dtype),
        )
        active = (
            torch.ones_like(meteo_quick, dtype=torch.bool) if valid is None else valid
        )
        args = (
            incoming[..., 0],
            meteo_quick,
            store_weight,
            scale,
            lo,
            hi,
            store_init[:, 0],
            active,
        )
        previous = (
            _QuickTrajectoryFunction.apply(*args)
            if self.fast_scans
            else _quick_prev_loop(*args)
        )
        volume = previous + incoming[..., 0]
        k_quick = _quick_rate_impl(
            volume, meteo_quick, store_weight.unsqueeze(1), scale.unsqueeze(1), lo, hi
        )
        k_quick = torch.where(active, k_quick, torch.zeros_like(k_quick))
        q_quick = k_quick * volume
        s_quick = volume - q_quick

        u, k, initial = (
            incoming[..., 1:].double(),
            kappa_meteo[..., 1:].double(),
            store_init[:, 1:].double(),
        )
        if self.fast_scans:
            batch_size, seq_len, n_linear = u.shape

            def rows(value):
                return value.transpose(1, 2).reshape(batch_size * n_linear, seq_len)

            zero = _decay_scan(rows(u), rows(torch.log1p(-k)), self.store_scan_chunk)
            zero = zero.reshape(batch_size, n_linear, seq_len).transpose(1, 2)
            volume_zero = (
                torch.cat([torch.zeros_like(zero[:, :1]), zero[:, :-1]], dim=1) + u
            )
            zero = torch.minimum(zero, volume_zero)
            survive = torch.cumprod(1.0 - k, dim=1)
            survive_prev = torch.cat(
                [torch.ones_like(survive[:, :1]), survive[:, :-1]], dim=1
            )
            q_linear = volume_zero - zero + initial.unsqueeze(1) * k * survive_prev
            s_linear = zero + initial.unsqueeze(1) * survive
        else:
            state = initial
            qs, ss = [], []
            for t in range(u.shape[1]):
                v = state + u[:, t]
                q = k[:, t] * v
                state = v - q
                qs.append(q)
                ss.append(state)
            q_linear, s_linear = torch.stack(qs, dim=1), torch.stack(ss, dim=1)
        outflow = torch.cat(
            [q_quick.unsqueeze(-1), q_linear.to(incoming.dtype)], dim=-1
        )
        storage = torch.cat(
            [s_quick.unsqueeze(-1), s_linear.to(incoming.dtype)], dim=-1
        )
        kappa = torch.cat([k_quick.unsqueeze(-1), kappa_meteo[..., 1:]], dim=-1)
        return outflow, storage, kappa

    def forward(
        self,
        inflow: torch.Tensor,
        pet: torch.Tensor,
        gen_logit: torch.Tensor,
        partition: torch.Tensor,
        rate_logits: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
        initial_soil: Optional[torch.Tensor] = None,
        initial_stores: Optional[torch.Tensor] = None,
        spinup_mask: Optional[torch.Tensor] = None,
        bootstrap_from_history: bool = False,
    ) -> Dict[str, torch.Tensor]:
        out_dtype = inflow.dtype
        dt = self.compute_dtype
        batch_size, seq_len = inflow.shape
        prm = self.basin_parameters(batch_size, static_context)

        inflow_c = inflow.to(dt)
        et_ceiling = pet.to(dt) * prm["pet_scale"].unsqueeze(1)
        part = partition.to(dt)
        part = part / part.sum(dim=-1, keepdim=True)
        residuals = rate_logits.to(dt)
        if self.quick_state_dependent:
            quick_residual = self.quick_meteo_logit_limit * torch.tanh(
                residuals[..., 0]
            )
            residuals = torch.cat(
                [quick_residual.unsqueeze(-1), residuals[..., 1:]], dim=-1
            )
        rate_logits_c = residuals + prm["kappa_bias"].unsqueeze(1)
        s = torch.sigmoid(rate_logits_c)
        log_lo = self.log_kappa_lo.to(dt)
        log_hi = self.log_kappa_hi.to(dt)
        kappa = torch.exp(log_lo + (log_hi - log_lo) * s)
        if spinup_mask is not None:
            if spinup_mask.shape != inflow.shape or spinup_mask.dtype != torch.bool:
                raise ValueError("spinup_mask must be a bool [B,T] tensor")
            valid = spinup_mask.to(device=inflow.device)

            inflow_c = torch.where(valid, inflow_c, 0.0)
            et_ceiling = torch.where(valid, et_ceiling, 0.0)
            kappa = torch.where(valid.unsqueeze(-1), kappa, 0.0)
            gen_logit = torch.where(valid, gen_logit.to(dt), -1e9)
        else:
            valid = None
            gen_logit = gen_logit.to(dt)
        soil_init = (
            self.soil_init_fraction * prm["s_cap"]
            if initial_soil is None
            else initial_soil.to(dt)
        )
        if (
            soil_init.shape != (batch_size,)
            or not bool(torch.isfinite(soil_init).all())
            or bool(((soil_init < 0) | (soil_init > prm["s_cap"])).any())
        ):
            raise ValueError(
                "initial_soil must be finite [B] within [0, soil capacity]"
            )
        if initial_stores is not None and bootstrap_from_history:
            raise ValueError("Specify either initial_stores or bootstrap_from_history")

        if self.quick_state_dependent:
            quick_prm = self.quick_response.basin_parameters(
                batch_size, static_context, dt
            )
            aet, recharge, drain_frac, soil = self._state_soil_fluxes(
                inflow_c, et_ceiling, gen_logit, prm, soil_init, valid
            )
        elif self.fast_scans:
            aet, recharge, drain_frac, soil, q_zero, s_zero = self._fast_ledger(
                inflow_c, et_ceiling, gen_logit, part, kappa, prm, soil_init
            )
        else:
            aet, recharge, drain_frac, soil, q_zero, s_zero = _water_ledger_scan(
                inflow_c,
                et_ceiling,
                gen_logit,
                part,
                kappa,
                prm["s_cap"],
                prm["et_threshold"],
                prm["w_wet"],
                prm["b_wet"],
                soil_init,
                self.cap_tau,
                self.beta_tau,
            )

        if initial_stores is not None:
            store_init = initial_stores.to(dt)
            if (
                store_init.shape != (batch_size, self.N_STORES)
                or not bool(torch.isfinite(store_init).all())
                or bool((store_init < 0).any())
            ):
                raise ValueError(
                    "initial_stores must be finite nonnegative [B,3] in mm"
                )
        elif bootstrap_from_history:
            count = (
                valid.sum(dim=1)
                if valid is not None
                else torch.full((batch_size,), seq_len, device=inflow.device)
            )
            count_f = count.clamp(min=1).to(dt).unsqueeze(-1)
            u_bar = (part * recharge.unsqueeze(-1)).sum(dim=1) / count_f
            k_bar = kappa.sum(dim=1) / count_f
            store_init = torch.where(
                k_bar > 0,
                (1.0 - k_bar) / k_bar.clamp(min=1e-12) * u_bar,
                torch.zeros_like(u_bar),
            )
            if self.quick_state_dependent:
                store_init = torch.cat(
                    [torch.zeros_like(store_init[:, :1]), store_init[:, 1:]], dim=-1
                )
        else:
            # Initialize stores from antecedent forcing.
            store_init = torch.zeros_like(part[:, 0, :])

        if self.quick_state_dependent:
            store_outflow, store_storage, kappa = self._state_store_route(
                part * recharge.unsqueeze(-1),
                rate_logits_c[..., 0],
                kappa,
                store_init,
                quick_prm,
                valid,
            )
        else:
            survive = torch.cumprod(1.0 - kappa, dim=1)
            survive_prev = torch.cat(
                [torch.ones_like(survive[:, :1]), survive[:, :-1]], dim=1
            )
            store_outflow = q_zero + store_init.unsqueeze(1) * kappa * survive_prev
            store_storage = s_zero + store_init.unsqueeze(1) * survive
        runoff = store_outflow.sum(dim=-1)

        storage_start = soil_init + store_init.sum(dim=-1)
        storage_end = soil[:, -1] + store_storage[:, -1, :].sum(dim=-1)
        input_total = inflow_c.sum(dim=1) + storage_start
        closure = input_total - aet.sum(dim=1) - runoff.sum(dim=1) - storage_end

        params = torch.stack(
            [
                prm["s_cap"],
                prm["pet_scale"],
                prm["et_threshold"],
                prm["w_wet"],
                prm["b_wet"],
            ],
            dim=-1,
        )
        result = {
            "runoff": runoff,
            "store_outflow": store_outflow,
            "partition": part,
            "kappa": kappa,
            "aet": aet,
            "et_ceiling": et_ceiling,
            "recharge": recharge,
            "drainage_fraction": drain_frac,
            "soil_storage": soil,
            "store_storage": store_storage,
            "soil_storage_init": soil_init,
            "store_storage_init": store_init,
            "ledger_parameters": params,
        }
        result = {k: v.to(out_dtype) for k, v in result.items()}

        result["closure_residual"] = closure
        result["ledger_input_total"] = input_total
        if not self.training:
            result.update(
                {
                    "window_liquid_sum": inflow_c.double().sum(dim=1),
                    "window_aet_sum": aet.double().sum(dim=1),
                    "window_outflow_sums": store_outflow.double().sum(dim=1),
                    "window_runoff_sum": runoff.double().sum(dim=1),
                    "window_soil_start": soil_init.double(),
                    "window_soil_end": soil[:, -1].double(),
                    "window_stores_start": store_init.double(),
                    "window_stores_end": store_storage[:, -1].double(),
                    "target_soil_start": (
                        soil[:, -2] if seq_len > 1 else soil_init
                    ).double(),
                    "target_soil_end": soil[:, -1].double(),
                    "target_stores_start": (
                        store_storage[:, -2] if seq_len > 1 else store_init
                    ).double(),
                    "target_stores_end": store_storage[:, -1].double(),
                    "target_outflows": store_outflow[:, -1].double(),
                    "target_aet": aet[:, -1].double(),
                    "target_recharge": recharge[:, -1].double(),
                    "target_partition": part[:, -1].double(),
                    "target_kappa": kappa[:, -1].double(),
                }
            )
        return result


class GatingRouter(nn.Module):
    """Partition soil recharge across the three response reservoirs using a softmax."""

    API_RHO_INIT = 2.2
    N_EXPERTS = 3

    N_SNOW_FEATURES = 3

    def __init__(
        self,
        d_hidden: int = 32,
        use_doy_feature: bool = False,
        static_context_size: int = 0,
        use_snow_features: bool = False,
    ) -> None:
        super().__init__()
        self.use_doy_feature = use_doy_feature
        self.use_snow_features = bool(use_snow_features)
        self.rho_raw = nn.Parameter(torch.tensor(self.API_RHO_INIT))

        feat_dim = 8 if use_doy_feature else 6
        feat_dim += self.N_SNOW_FEATURES if self.use_snow_features else 0
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(feat_dim, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, self.N_EXPERTS),
        )

        self._physical_init()

    def _physical_init(self) -> None:

        out_layer = self.dynamic_mlp[-1]
        with torch.no_grad():
            out_layer.bias[2] += 1.0

    def _api30(self, p: torch.Tensor) -> torch.Tensor:
        return antecedent_index(p, self.rho_raw, window=30)

    @staticmethod
    def _p5(p: torch.Tensor) -> torch.Tensor:
        """5-day rainfall total: a causal FIR with an all-ones kernel."""
        k_len = min(5, p.shape[1])
        kernel = torch.ones(k_len, device=p.device, dtype=p.dtype)
        return _causal_fir(p, kernel)

    def forward(
        self,
        p_raw: torch.Tensor,
        t_avg: torch.Tensor,
        vp: torch.Tensor,
        sw: torch.Tensor,
        doy: Optional[torch.Tensor] = None,
        static_bias_logits: Optional[torch.Tensor] = None,
        snow_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_avg = t_avg.squeeze(-1)
        vp = vp.squeeze(-1)
        sw = sw.squeeze(-1)

        p_feat = torch.log1p(p_raw)
        p5 = torch.log1p(self._p5(p_raw))
        api30 = torch.log1p(self._api30(p_raw))

        if self.use_doy_feature:
            if doy is None:
                raise ValueError(
                    "use_doy_feature=True, but no DOY tensor was provided to GatingRouter."
                )
            sin_doy = torch.sin(2 * math.pi * doy.float() / 365.0)
            cos_doy = torch.cos(2 * math.pi * doy.float() / 365.0)
            feat = torch.stack(
                [sin_doy, cos_doy, p5, api30, p_feat, t_avg, vp, sw], dim=-1
            )
        else:
            feat = torch.stack([p5, api30, p_feat, t_avg, vp, sw], dim=-1)

        if self.use_snow_features:
            if snow_features is None or snow_features.shape != (
                *p_raw.shape,
                self.N_SNOW_FEATURES,
            ):
                raise ValueError(
                    "Router requires [log1p(SWE_before), log1p(melt), snow cover] "
                    f"with shape [B,T,{self.N_SNOW_FEATURES}]"
                )
            feat = torch.cat([feat, snow_features], dim=-1)
        logits = self.dynamic_mlp(feat)
        if static_bias_logits is not None:
            logits = logits + static_bias_logits.unsqueeze(1)

        # The partition allocates all generated recharge.
        return torch.softmax(logits, dim=-1)


class TriFlowCore(nn.Module):
    """Couple shared snow and neural rates to the state–flux backbone."""

    def __init__(
        self,
        d_model: int = 64,
        d_k: int = 32,
        d_hidden_gate: int = 32,
        feature_order: str = "P_SW_TMAX_TMIN_VP",
        use_doy_feature: bool = False,
        aux_input_size: int = 0,
        static_input_size: int = 0,
        static_hidden_size: int = 64,
        static_context_size: int = 64,
        static_dropout: float = 0.0,
        quick_window: int = 7,
        inter_window: int = 30,
        dropout: float = 0.0,
        use_snow_module: bool = False,
        precip_mean: Optional[float] = None,
        precip_std: Optional[float] = None,
        snow_transition_width: float = 1.0,
        snow_initial_ddf: float = 3.0,
        snow_kwargs: Optional[Dict] = None,
        snow_detach_features: bool = True,
        ledger_kwargs: Optional[Dict] = None,
        fast_scans: bool = True,
    ) -> None:
        super().__init__()
        self.layout = MeteoFeatureLayout(feature_order)
        self.embed = VariableEmbedding(d_model=d_model, aux_input_size=aux_input_size)
        self.film = FiLM()

        self.dropout_rate = float(dropout)
        self.drop = nn.Dropout(self.dropout_rate)
        self.expert1 = QuickFlowExpert(
            d_model=d_model,
            d_k=d_k,
            d_p=self.embed.d_p,
            window=quick_window,
            dropout=self.dropout_rate,
        )
        self.expert2 = InterflowExpert(
            d_model=d_model,
            d_k=d_k,
            d_other=self.embed.d_other,
            window=inter_window,
            dropout=self.dropout_rate,
        )
        self.expert3 = BaseflowExpert(
            d_model=d_model, d_k=d_k, dropout=self.dropout_rate, fast_scans=fast_scans
        )

        self.gen_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gen_head.weight)
        nn.init.zeros_(self.gen_head.bias)

        if static_input_size > 0:
            self.static_encoder = StaticFeatureEncoder(
                input_size=static_input_size,
                context_size=static_context_size,
                hidden_size=static_hidden_size,
                dropout=static_dropout,
            )
            self.static_conditioner = StaticConditioner(
                context_size=static_context_size,
                d_p=self.embed.d_p,
                d_other=self.embed.d_other,
                d_model=d_model,
                n_experts=GatingRouter.N_EXPERTS,
            )
            router_static_context_size = static_context_size
        else:
            self.static_encoder = None
            self.static_conditioner = None
            router_static_context_size = 0

        self.use_snow_module = bool(use_snow_module)
        if self.use_snow_module:
            if (
                precip_mean is None
                or precip_std is None
                or not math.isfinite(float(precip_mean))
                or not math.isfinite(float(precip_std))
                or float(precip_std) <= 0
            ):
                raise ValueError(
                    "Snow module requires finite precip_mean and positive precip_std from TRAINING data"
                )
            self.register_buffer("precip_mean", torch.tensor(float(precip_mean)))
            self.register_buffer("precip_std", torch.tensor(float(precip_std)))
            self.snow = SharedSnowModule(
                static_context_size=router_static_context_size,
                transition_width=snow_transition_width,
                initial_ddf=snow_initial_ddf,
                **(snow_kwargs or {}),
            )

            self.snow_embedding = nn.Linear(4, d_model, bias=False)
            nn.init.zeros_(self.snow_embedding.weight)

        self.snow_detach_features = bool(snow_detach_features)

        self.router = GatingRouter(
            d_hidden=d_hidden_gate,
            use_doy_feature=use_doy_feature,
            static_context_size=router_static_context_size,
            use_snow_features=self.use_snow_module,
        )
        self.ledger = WaterLedger(
            static_context_size=router_static_context_size,
            fast_scans=fast_scans,
            **(ledger_kwargs or {}),
        )

    def _neural_rates(
        self,
        x_meteo: torch.Tensor,
        x_aux: Optional[torch.Tensor],
        liquid: torch.Tensor,
        cond: Dict[str, Optional[torch.Tensor]],
        doy: Optional[torch.Tensor],
        snow_fields: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict dynamic rates for the current window (once per forward pass)."""
        p, sw, tmax, tmin, vp, t_avg = self.layout.split(x_meteo)
        neural_liquid = liquid.detach() if self.snow_detach_features else liquid
        router_snow_features = None
        if snow_fields is not None:
            p = ((neural_liquid - self.precip_mean) / self.precip_std).unsqueeze(-1)
            snowfall, melt, swe_before, cover_before = snow_fields
            if self.snow_detach_features:
                snowfall, melt, swe_before, cover_before = (
                    v.detach() for v in snow_fields
                )
            log_swe, log_melt = torch.log1p(swe_before), torch.log1p(melt)
            router_snow_features = torch.stack(
                [log_swe, log_melt, cover_before], dim=-1
            )
            snow_embed_features = torch.stack(
                [torch.log1p(snowfall), log_swe, log_melt, cover_before], dim=-1
            )
        else:
            snow_embed_features = None

            neural_liquid = liquid

        other = self.layout.build_other(sw, tmax, tmin, vp)
        e, h_p, h_other = self.embed(p=p, other=other, x_aux=x_aux)
        if snow_embed_features is not None:
            e = e + self.snow_embedding(snow_embed_features)
        h_p = self.film(h_p, cond["gamma_p"], cond["beta_p"])
        h_other = self.film(h_other, cond["gamma_other"], cond["beta_other"])
        e = self.drop(self.film(e, cond["gamma_e"], cond["beta_e"]))

        rates = torch.stack(
            [
                self.expert1(e=e, h_p=h_p, p_std=p, t_avg=t_avg, sw=sw),
                self.expert2(e=e, h_other=h_other),
                self.expert3(e=e, init_state=cond["baseflow_init"]),
            ],
            dim=-1,
        )
        partition = self.router(
            p_raw=neural_liquid,
            t_avg=t_avg,
            vp=vp,
            sw=sw,
            doy=doy,
            static_bias_logits=cond["router_bias"],
            snow_features=router_snow_features,
        )
        return self.gen_head(e).squeeze(-1), partition, rates

    def forward(
        self,
        x_meteo: torch.Tensor,
        p_raw: torch.Tensor,
        pet_raw: torch.Tensor,
        x_aux: Optional[torch.Tensor] = None,
        x_static: Optional[torch.Tensor] = None,
        doy: Optional[torch.Tensor] = None,
        t_raw: Optional[torch.Tensor] = None,
        warmup_p: Optional[torch.Tensor] = None,
        warmup_pet: Optional[torch.Tensor] = None,
        warmup_t: Optional[torch.Tensor] = None,
        warmup_mask: Optional[torch.Tensor] = None,
        initial_swe: Optional[torch.Tensor] = None,
        snow_doy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        static_context = (
            self.static_encoder(x_static) if self.static_encoder is not None else None
        )
        cond = (
            self.static_conditioner(static_context)
            if self.static_conditioner is not None
            else {
                "gamma_p": None,
                "beta_p": None,
                "gamma_other": None,
                "beta_other": None,
                "gamma_e": None,
                "beta_e": None,
                "baseflow_init": None,
                "router_bias": None,
            }
        )

        snow_out: Dict[str, torch.Tensor] = {}
        liquid_input = p_raw
        history_liquid = None
        current_snow_fields = None
        if self.use_snow_module:
            snow_out = self.snow(
                p_raw,
                t_raw,
                static_context=static_context,
                warmup_p=warmup_p,
                warmup_t=warmup_t,
                warmup_mask=warmup_mask,
                initial_swe=initial_swe,
                doy=snow_doy,
            )
            history_liquid = snow_out.pop("liquid_input_history")
            warmup_mask = snow_out.pop("history_mask")
            current_snow_fields = (
                snow_out["snowfall"],
                snow_out["snowmelt"],
                snow_out["swe_before"],
                snow_out.pop("snow_cover_before"),
            )
            liquid_input = snow_out["liquid_input"]
        elif warmup_p is not None:
            history_liquid = warmup_p

        gen_logit, partition, rates = self._neural_rates(
            x_meteo, x_aux, liquid_input, cond, doy, current_snow_fields
        )

        initial_soil = initial_stores = None
        spinup_days = torch.zeros(p_raw.shape[0], device=p_raw.device, dtype=torch.long)
        if warmup_p is not None:
            if warmup_pet is None or warmup_mask is None:
                raise ValueError(
                    "History spin-up requires warmup_p, warmup_pet and warmup_mask"
                )
            if history_liquid is None or history_liquid.shape != warmup_p.shape:
                raise ValueError("Warmup liquid history has the wrong shape")
            if bool((warmup_mask[:, :-1] & ~warmup_mask[:, 1:]).any()):
                raise ValueError("warmup_mask may contain leading padding only")
            spinup_days = warmup_mask.sum(dim=1)
            n_batch, n_hist = warmup_p.shape
            if n_hist:
                # Advance physical stores with basin-conditioned rates and neutral neural residuals.

                prior_logits = (
                    self.router.dynamic_mlp[-1].bias.unsqueeze(0).expand(n_batch, -1)
                )
                if cond["router_bias"] is not None:
                    prior_logits = prior_logits + cond["router_bias"]
                hist_partition = (
                    torch.softmax(prior_logits, dim=-1)
                    .unsqueeze(1)
                    .expand(-1, n_hist, -1)
                )
                hist_gen = torch.zeros_like(warmup_p)
                hist_rates = warmup_p.new_zeros(n_batch, n_hist, 3)
                initial = self.ledger(
                    inflow=history_liquid,
                    pet=warmup_pet,
                    gen_logit=hist_gen,
                    partition=hist_partition,
                    rate_logits=hist_rates,
                    static_context=static_context,
                    spinup_mask=warmup_mask,
                    bootstrap_from_history=True,
                )
                initial_soil = initial["soil_storage"][:, -1]
                initial_stores = initial["store_storage"][:, -1]

        ledger = self.ledger(
            inflow=liquid_input,
            pet=pet_raw,
            gen_logit=gen_logit,
            partition=partition,
            rate_logits=rates,
            static_context=static_context,
            initial_soil=initial_soil,
            initial_stores=initial_stores,
        )
        runoff = ledger["runoff"]

        # Check whole-system closure across snow, soil, and all response stores.

        p_sum = p_raw.double().sum(dim=1)
        liquid_sum = liquid_input.double().sum(dim=1)
        if self.use_snow_module:
            swe_start = snow_out.pop("swe_window_start")
            swe_end = snow_out.pop("swe_window_end")
            snow_handover = p_sum + swe_start - swe_end - liquid_sum
        else:
            swe_start = torch.zeros_like(p_sum)
            swe_end = torch.zeros_like(p_sum)
            snow_handover = p_sum - liquid_sum
        system_residual = ledger["closure_residual"].double() + snow_handover
        system_input_total = (
            ledger["ledger_input_total"].double() - liquid_sum + p_sum + swe_start
        )

        if not self.training:
            water_start = (
                swe_start
                + ledger["window_soil_start"]
                + ledger["window_stores_start"].sum(-1)
            )
            water_end = (
                swe_end
                + ledger["window_soil_end"]
                + ledger["window_stores_end"].sum(-1)
            )
            available = p_sum + water_start
            saved_residual = (
                available
                - ledger["window_runoff_sum"]
                - ledger["window_aet_sum"]
                - water_end
            )
            swe_target_start = snow_out.pop("swe_target_start", torch.zeros_like(p_sum))
            swe_target_end = snow_out.pop("swe_target_end", torch.zeros_like(p_sum))
            target_start = (
                swe_target_start
                + ledger["target_soil_start"]
                + ledger["target_stores_start"].sum(-1)
            )
            target_end = (
                swe_target_end
                + ledger["target_soil_end"]
                + ledger["target_stores_end"].sum(-1)
            )
            ledger.update(
                {
                    "window_precip_sum": p_sum,
                    "window_swe_start": swe_start,
                    "window_swe_end": swe_end,
                    "window_water_start": water_start,
                    "window_water_end": water_end,
                    "window_available_water": available,
                    "window_balance_residual": saved_residual,
                    "target_swe_start": swe_target_start,
                    "target_swe_end": swe_target_end,
                    "target_balance_residual": (
                        p_raw[:, -1].double()
                        + target_start
                        - ledger["target_outflows"].sum(-1)
                        - ledger["target_aet"]
                        - target_end
                    ),
                }
            )

        return {
            **ledger,
            "system_closure_residual": system_residual,
            "system_input_total": system_input_total,
            "y_seq": runoff,
            "p_raw": p_raw,
            "pet_raw": pet_raw,
            "spinup_days": spinup_days,
            **snow_out,
            "liquid_input": liquid_input,
        }


class TriFlowMoE(nn.Module):
    """TriFlow-MoE for daily runoff prediction.

    Pass standardized dynamic inputs [B, T, F] and basin attributes [B, S].
    Precipitation and PET are in mm/day. Mean air temperature is in degrees Celsius.
    Preceding raw forcing initializes storage. The history mask marks available days.
    The final daily runoff prediction, y_hat, has shape [B, 1] and units of mm/day.
    """

    CORE_METEO_SIZE = 5

    def __init__(
        self, hyperparameters: Dict[str, Union[int, float, str, dict, bool, list]]
    ):
        super().__init__()
        self.input_size_dynamic = int(hyperparameters["input_size_dynamic"])
        self.static_input_size = int(hyperparameters.get("static_input_size", 0))

        self.d_model = int(hyperparameters["d_model"])
        self.d_k = int(hyperparameters["d_k"])
        self.d_hidden_gate = int(hyperparameters["d_hidden_gate"])
        self.static_hidden_size = int(hyperparameters.get("static_hidden_size", 64))
        self.static_context_size = int(
            hyperparameters.get("static_context_size", self.d_model)
        )
        self.feature_order = str(
            hyperparameters.get("feature_order", "P_SW_TMAX_TMIN_VP")
        )
        self.use_doy_feature = bool(hyperparameters.get("use_doy_feature", True))
        self.static_dropout = float(hyperparameters.get("static_dropout", 0.0))
        self.drop_out_rate = float(hyperparameters.get("drop_out_rate", 0.0))
        if not 0.0 <= self.drop_out_rate < 1.0:
            raise ValueError("drop_out_rate must be in [0, 1)")

        self.use_snow_module = bool(hyperparameters.get("use_snow_module", False))
        self.snow_transition_width = float(
            hyperparameters.get("snow_transition_width", 1.0)
        )
        self.snow_initial_ddf = float(hyperparameters.get("snow_initial_ddf", 3.0))

        self.snow_n_bands = int(hyperparameters.get("snow_n_bands", 5))
        self.snow_seasonal_melt = bool(hyperparameters.get("snow_seasonal_melt", True))
        self.snow_cold_content = bool(hyperparameters.get("snow_cold_content", True))
        self.snow_initial_ddf_ratio = float(
            hyperparameters.get("snow_initial_ddf_ratio", 0.5)
        )
        self.snow_lapse_rate = float(hyperparameters.get("snow_lapse_rate", 6.5))
        self.snow_ripeness_scale = float(
            hyperparameters.get("snow_ripeness_scale", 1.0)
        )
        self.snow_detach_features = bool(
            hyperparameters.get("snow_detach_features", True)
        )
        self.quick_window = int(hyperparameters.get("quick_window", 7))
        self.inter_window = int(hyperparameters.get("inter_window", 30))
        self.quick_state_dependent = bool(
            hyperparameters.get("quick_state_dependent", True)
        )
        self.quick_state_weight_max = float(
            hyperparameters.get("quick_state_weight_max", 4.0)
        )
        self.quick_state_weight_init = float(
            hyperparameters.get("quick_state_weight_init", 1.0)
        )
        self.quick_meteo_logit_limit = float(
            hyperparameters.get("quick_meteo_logit_limit", 0.5)
        )
        self.quick_storage_scale_bounds = tuple(
            float(v)
            for v in hyperparameters.get("quick_storage_scale_bounds", (1.0, 100.0))
        )

        self.ledger_kappa_bounds = tuple(
            tuple(float(v) for v in pair)
            for pair in hyperparameters.get(
                "ledger_kappa_bounds", WaterLedger.DEFAULT_KAPPA_BOUNDS
            )
        )
        self.ledger_soil_capacity_bounds = tuple(
            float(v)
            for v in hyperparameters.get("ledger_soil_capacity_bounds", (25.0, 600.0))
        )
        self.ledger_pet_scale_bounds = tuple(
            float(v) for v in hyperparameters.get("ledger_pet_scale_bounds", (0.5, 1.5))
        )
        self.ledger_et_threshold_bounds = tuple(
            float(v)
            for v in hyperparameters.get("ledger_et_threshold_bounds", (0.3, 1.0))
        )
        self.ledger_soil_init_fraction = float(
            hyperparameters.get("ledger_soil_init_fraction", 0.5)
        )
        self.ledger_cap_tau = float(hyperparameters.get("ledger_cap_tau", 1.0))
        self.ledger_float64 = bool(hyperparameters.get("ledger_float64", True))

        self.fast_scans = bool(hyperparameters.get("fast_scans", True))

        if self.input_size_dynamic < self.CORE_METEO_SIZE:
            raise ValueError(
                "TriFlowMoE requires at least five meteorological inputs. "
                "The default order is [P, SW, T_max, T_min, VP]."
            )
        if self.d_model < 2:
            raise ValueError("d_model must be >= 2")

        aux_input_size = max(0, self.input_size_dynamic - self.CORE_METEO_SIZE)
        self.core = TriFlowCore(
            d_model=self.d_model,
            d_k=self.d_k,
            d_hidden_gate=self.d_hidden_gate,
            feature_order=self.feature_order,
            use_doy_feature=self.use_doy_feature,
            aux_input_size=aux_input_size,
            static_input_size=self.static_input_size,
            static_hidden_size=self.static_hidden_size,
            static_context_size=self.static_context_size,
            static_dropout=self.static_dropout,
            quick_window=self.quick_window,
            inter_window=self.inter_window,
            dropout=self.drop_out_rate,
            use_snow_module=self.use_snow_module,
            precip_mean=hyperparameters.get("precip_mean"),
            precip_std=hyperparameters.get("precip_std"),
            snow_transition_width=self.snow_transition_width,
            snow_initial_ddf=self.snow_initial_ddf,
            snow_kwargs={
                "n_bands": self.snow_n_bands,
                "seasonal_melt": self.snow_seasonal_melt,
                "cold_content": self.snow_cold_content,
                "initial_ddf_ratio": self.snow_initial_ddf_ratio,
                "lapse_rate": self.snow_lapse_rate,
                "ripeness_scale": self.snow_ripeness_scale,
            },
            snow_detach_features=self.snow_detach_features,
            ledger_kwargs={
                "kappa_bounds": self.ledger_kappa_bounds,
                "soil_capacity_bounds": self.ledger_soil_capacity_bounds,
                "pet_scale_bounds": self.ledger_pet_scale_bounds,
                "et_threshold_bounds": self.ledger_et_threshold_bounds,
                "soil_init_fraction": self.ledger_soil_init_fraction,
                "cap_tau": self.ledger_cap_tau,
                "use_float64": self.ledger_float64,
                "quick_state_dependent": self.quick_state_dependent,
                "quick_state_weight_max": self.quick_state_weight_max,
                "quick_state_weight_init": self.quick_state_weight_init,
                "quick_storage_scale_bounds": self.quick_storage_scale_bounds,
                "quick_meteo_logit_limit": self.quick_meteo_logit_limit,
            },
            fast_scans=self.fast_scans,
        )

    @staticmethod
    def _get_doy(doy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if doy.dim() == 3:
            if doy.shape[-1] != 1:
                raise ValueError(
                    f"doy must have shape [B,T] or [B,T,1]; got {tuple(doy.shape)}"
                )
            doy = doy.squeeze(-1)
        if doy.dim() != 2:
            raise ValueError(
                f"doy must have shape [B,T] or [B,T,1]; got {tuple(doy.shape)}"
            )
        if tuple(doy.shape) != tuple(x.shape[:2]):
            raise ValueError(
                f"doy shape {tuple(doy.shape)} does not match x batch/time "
                f"shape {tuple(x.shape[:2])}"
            )
        return torch.round(doy.float()).clamp(min=1, max=365)

    def forward(
        self,
        x: torch.Tensor,
        x_static: Optional[torch.Tensor] = None,
        doy: Optional[torch.Tensor] = None,
        p_raw: Optional[torch.Tensor] = None,
        pet_raw: Optional[torch.Tensor] = None,
        t_raw: Optional[torch.Tensor] = None,
        warmup_p: Optional[torch.Tensor] = None,
        warmup_pet: Optional[torch.Tensor] = None,
        warmup_t: Optional[torch.Tensor] = None,
        warmup_mask: Optional[torch.Tensor] = None,
        initial_swe: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B,T,F]; got {tuple(x.shape)}")
        if x.shape[-1] != self.input_size_dynamic:
            raise ValueError(
                f"Dynamic input has {x.shape[-1]} features, but input_size_dynamic="
                f"{self.input_size_dynamic}. Check static-feature concatenation and "
                "feature_order."
            )
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Stateful updates require finite raw forcing.
        p_raw = _physical_series(
            p_raw, "p_raw", batch_size, x, seq_len, nonnegative=True
        )
        pet_raw = _physical_series(
            pet_raw, "pet_raw", batch_size, x, seq_len, nonnegative=True
        )
        if warmup_p is not None and not self.use_snow_module:
            warmup_p = _physical_series(
                warmup_p, "warmup_p", batch_size, x, nonnegative=True
            )
            if warmup_mask is None:
                warmup_mask = torch.ones_like(warmup_p, dtype=torch.bool)
            elif warmup_mask.dtype != torch.bool or warmup_mask.shape != warmup_p.shape:
                raise ValueError("warmup_mask must be bool [B,W]")
        if warmup_p is not None:
            if warmup_pet is None or warmup_mask is None:
                raise ValueError(
                    "History spin-up requires warmup_p, warmup_pet and warmup_mask"
                )
            warmup_pet = _physical_series(
                warmup_pet,
                "warmup_pet",
                batch_size,
                x,
                warmup_p.shape[1],
                nonnegative=True,
            )
        elif warmup_pet is not None:
            raise ValueError("warmup_pet requires warmup_p")

        x_meteo = x[..., : self.CORE_METEO_SIZE]
        x_aux = (
            x[..., self.CORE_METEO_SIZE :]
            if x.shape[-1] > self.CORE_METEO_SIZE
            else None
        )

        # Seasonal snowmelt uses the actual day of year, including leap days.

        snow_doy = None
        if self.use_snow_module and self.snow_seasonal_melt:
            if doy is None:
                raise ValueError(
                    "snow_seasonal_melt=True, but doy was not provided "
                    "(the dataset must be built with return_doy=True)"
                )
            snow_doy = doy.squeeze(-1) if doy.dim() == 3 else doy
            if tuple(snow_doy.shape) != (batch_size, seq_len):
                raise ValueError(
                    f"doy shape {tuple(doy.shape)} does not match x batch/time shape "
                    f"{(batch_size, seq_len)}"
                )

        if self.use_doy_feature:
            if doy is None:
                raise ValueError("use_doy_feature=True, but doy was not provided")
            doy = self._get_doy(doy, x=x)
        else:
            doy = None

        core_out = self.core(
            x_meteo=x_meteo,
            p_raw=p_raw,
            pet_raw=pet_raw,
            x_aux=x_aux,
            x_static=x_static,
            doy=doy,
            t_raw=t_raw,
            warmup_p=warmup_p,
            warmup_pet=warmup_pet,
            warmup_t=warmup_t,
            warmup_mask=warmup_mask,
            initial_swe=initial_swe,
            snow_doy=snow_doy,
        )
        y_hat = core_out["y_seq"][:, -1:]
        return {**core_out, "y_hat": y_hat}
