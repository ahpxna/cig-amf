"""
structural_proxy.py — Local counterfactual proxy.

=============================================================================
BỐN LỖI CỦA BẢN v1 ĐƯỢC SỬA Ở ĐÂY
=============================================================================

[L1] DẤU BỊ HUỶ NGAY TRONG ESTIMATOR.
     v1: abs_effect = torch.abs(alt_preds - base_preds)  -> mu LUÔN >= 0.
     Hệ quả dây chuyền:
       - p_core buộc phải dùng |mu_bar| (vì mu đã không âm sẵn)
       - không thể phân biệt "thằng ngáng đường" với "thằng hỗ trợ"
       - KHÔNG THỂ khớp với oracle của env (oracle CÓ DẤU) -> Exp3 bất khả thi
     Hiện tại: giữ dấu. Expose 4 chế độ (xem effect_mode) để chạy ablation.

[L2] ƯỚC LƯỢNG PLUG-IN, THỪA HƯỞNG TOÀN BỘ BIAS CỦA REWARD MODEL.
     v1: w = f(a') - f(a). Nếu f lệch thì w lệch. Mà a_j không ngẫu nhiên
     nên f học được từ dữ liệu confounded.
     Hiện tại: doubly-robust. Vì trong MARL ta BIẾT CHÍNH XÁC pi_j (ta tự
     train nó), propensity là exact -> DR không chệch ngay cả khi f sai.

[L3] ENSEMBLE GIẢ — cả 3 member train trên CÙNG batch, cùng thứ tự.
     v1: `for model, optim in zip(self.models, self.optims)` nằm TRONG vòng
     lặp batch -> mọi member thấy đúng cùng dữ liệu -> hội tụ về gần cùng
     một hàm -> sigma = 0.000.
     Hiện tại: mỗi member có bootstrap mask riêng + batch riêng + khởi tạo
     riêng — VÀ toàn bộ ensemble forward/backward/step chạy như MỘT lệnh
     GPU duy nhất qua torch.func.vmap, không còn vòng lặp Python qua từng
     model (xem "TỐI ƯU GPU" bên dưới).

[L4] CHỈ MỘT HORIZON -> không có chiều "độ trễ" cho influence signature.
     Multi-horizon head, dự đoán R^(1), R^(2), ..., R^(H) cùng lúc.

=============================================================================
TỐI ƯU GPU (torch.func.vmap ensemble)
=============================================================================
Vấn đề gốc: n_ensemble model riêng biệt, mỗi model là một nn.Module có bộ
trọng số và optimizer của riêng nó. Chạy forward/backward cho ensemble theo
kiểu `for model in self.models: ...` nghĩa là GPU thấy n_ensemble lần launch
kernel TUẦN TỰ cho những phép tính có shape GIỐNG HỆT NHAU — không có gì để
GPU song song hoá giữa các member, và mỗi lần `.item()`/`.cpu()` trong vòng
lặp đó là một lần đồng bộ CPU<->GPU (stall pipeline).

Cách sửa — "vectorize the ensemble dimension":
    1. `torch.func.stack_module_state(models)` gộp trọng số của n_ensemble
       model thành MỘT cây tensor, mỗi lá có thêm chiều đầu E = n_ensemble.
    2. `torch.func.functional_call` + `torch.func.vmap` chạy forward cho
       CẢ n_ensemble model trong MỘT lệnh, y hệt cách vmap chạy batch trong
       chiều batch — chỉ khác là chiều được vmap ở đây là chiều ENSEMBLE.
    3. Một Adam DUY NHẤT tối ưu toàn bộ tensor đã stack (Adam là elementwise
       theo tham số nên vẫn cho mỗi member một moment estimate độc lập —
       không có rò rỉ gradient giữa các member).
    4. Grad-clip vẫn phải tính NORM RIÊNG cho từng member (nếu tính chung một
       norm toàn cục thì một member có gradient lớn sẽ kéo cả ensemble bị
       clip theo, làm mất tính độc lập) — cài trong `_clip_grad_norm_per_member`,
       vector hoá theo chiều E, không có vòng lặp Python qua từng member.
    5. Kết quả: n_ensemble forward + n_ensemble backward + n_ensemble
       optimizer update chỉ còn là 1 forward + 1 backward + 1 update trên
       tensor có thêm chiều E — đúng những gì torch.vmap sinh ra để làm.

`self.buffer` CỐ TÌNH vẫn là `deque` các dict Python — `drift_probe.py`
(`DriftDetector`) đọc trực tiếp cấu trúc này (`buf["obs_i"]`, v.v.) nên đổi
kiểu dữ liệu ở đây sẽ làm gãy module đó. Việc lấy mẫu (`_sample_for_member`)
được viết lại để KHÔNG quét toàn bộ buffer bằng vòng lặp Python + hash thủ
công như trước (O(buffer_size) mỗi member mỗi bước train — với buffer 200k
phần tử và 4 member thì đó là 800k phép tính Python thuần mỗi train_step).

Thay vào đó mỗi member có một PERMUTATION MASK CỐ ĐỊNH (tính một lần ở
__init__, numpy — O(buffer_size) một lần duy nhất, không lặp lại mỗi
train_step) đánh dấu ~bootstrap_ratio phần trăm vị trí (rank) mà member đó
được phép thấy; `_sample_for_member` chỉ lọc theo mask (numpy, C-level) rồi
`random.choices` có trọng số để oversample mẫu can thiệp. Mask KHÔNG được
rút lại ngẫu nhiên mỗi lần gọi — xem [BB1] trong `GPU_OPTIMIZATION_CONTRACT.md`:
rút ngẫu nhiên mỗi lần khiến mọi member dần "thấy" gần hết buffer, xoá mất
khác biệt hệ thống giữa các member, và ensemble hội tụ về nhau y hệt lỗi v1.

=============================================================================
BACKWARD COMPATIBILITY
=============================================================================
Giữ nguyên chữ ký các hàm runner đang gọi:
    add_sample(...)  -> thêm tham số optional, có default
    train_step(...)  -> giữ nguyên
    score_batch(...) -> giữ nguyên, VẪN trả (mu_arr, sigma_arr)
    score_pair(...)  -> giữ nguyên
    score_batch_full(...) -> trả dict đầy đủ để xây influence signature
    self.buffer      -> vẫn là deque[dict], drift_probe.py phụ thuộc vào nó
=============================================================================
"""

import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call, stack_module_state, vmap
    _HAS_TORCH_FUNC = True
except ImportError:  # torch < 2.0 fallback — hiếm, nhưng đừng chết cứng
    _HAS_TORCH_FUNC = False


# =============================================================================
# Mạng proxy
# =============================================================================

class LocalCounterfactualProxyNet(nn.Module):
    """
    Một member trong ensemble.

    Input (đúng conditioning set của paper, Eq. 5):
        obs_i, a_i, a_j, Z_i^{-j}, M_i^{-j}, B_i

    Output:
        [B, n_horizons] — dự đoán R_i^(1), R_i^(2), ..., R_i^(H)

    Vì sao multi-horizon:
        Ảnh hưởng của một neighbour có ĐỘ TRỄ. Blocker chặn đường tác động
        tức thì (h=1). Relay/signaller phát tín hiệu thì lợi ích chỉ hiện ra
        sau vài bước (h=3). Nếu chỉ dự đoán R^(H) tổng, hai loại này trông
        giống hệt nhau. Tách theo horizon là chiều thứ 6 của signature.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        core_dim: int,
        periph_dim: int,
        belief_dim: int,
        hidden: int = 160,
        n_horizons: int = 8,
        use_belief_input: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.core_dim = int(core_dim)
        self.periph_dim = int(periph_dim)
        self.belief_dim = int(belief_dim)
        self.hidden = int(hidden)
        self.n_horizons = int(n_horizons)

        # ---------------------------------------------------------------
        # [H6] CẮT VÒNG LẶP PHẢN HỒI belief -> proxy -> belief
        #
        # B_i được SINH RA TỪ chính w_hat mà proxy này tạo ra. Nếu đưa B_i
        # vào input, proxy có thể "tự xác nhận" -> confounder do chính
        # kiến trúc tạo ra. Mặc định TẮT belief input.
        # ---------------------------------------------------------------
        self.use_belief_input = bool(use_belief_input)

        self.in_dim = (
            self.obs_dim
            + self.action_dim   # a_i one-hot
            + self.action_dim   # a_j one-hot
            + self.core_dim
            + self.periph_dim
            + (self.belief_dim if self.use_belief_input else 0)
        )

        layers = [
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))

        layers += [
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))

        # Output n_horizons thay vì 1.
        layers.append(nn.Linear(self.hidden, self.n_horizons))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        obs_i: torch.Tensor,            # [..., B, obs_dim]
        action_i_onehot: torch.Tensor,  # [..., B, action_dim]
        action_j_onehot: torch.Tensor,  # [..., B, action_dim]
        z_core_excl_j: torch.Tensor,    # [..., B, core_dim]
        m_periph_excl_j: torch.Tensor,  # [..., B, periph_dim]
        belief_summary: torch.Tensor,   # [..., B, belief_dim]
    ) -> torch.Tensor:
        """
        Returns:
            [..., B, n_horizons]

        Chấp nhận cả input không có chiều ensemble (gọi trực tiếp một
        model, dùng ở smoke test / debug) lẫn input có sẵn chiều batch
        (đường vmap chuẩn — vmap tự thêm/bỏ chiều được map, hàm forward
        không cần biết gì về chiều E).
        """
        parts = [
            obs_i,
            action_i_onehot,
            action_j_onehot,
            z_core_excl_j,
            m_periph_excl_j,
        ]

        if self.use_belief_input:
            parts.append(belief_summary)

        x = torch.cat(parts, dim=-1)  # [..., in_dim]

        return self.net(x)  # [..., n_horizons]


# =============================================================================
# Grad-clip riêng cho từng member (vector hoá theo chiều E)
# =============================================================================

def _clip_grad_norm_per_member(
    stacked_params: Dict[str, torch.Tensor],
    max_norm: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Clip gradient-norm ĐỘC LẬP cho từng ensemble member.

    Nếu dùng `torch.nn.utils.clip_grad_norm_` trực tiếp trên list các tensor
    đã stack, nó sẽ tính MỘT norm toàn cục qua cả chiều E lẫn chiều tham số
    -> một member có gradient bùng nổ sẽ kéo tất cả member khác bị clip
    theo, phá vỡ tính độc lập vốn là lý do ensemble tồn tại. Hàm này tính
    norm riêng cho từng lát cắt E rồi scale riêng — không có vòng lặp Python
    qua từng member, chỉ có một vòng lặp (nhỏ, cố định) qua các THAM SỐ
    (weight/bias của từng layer), không phải qua agents/ensemble.

    Returns:
        [E] — norm gradient trước khi clip (để log/chẩn đoán nếu cần).
    """
    grads = [p.grad for p in stacked_params.values() if p.grad is not None]

    if len(grads) == 0:
        return torch.zeros(0)

    E = int(grads[0].shape[0])
    device = grads[0].device

    sq_sum = torch.zeros(E, device=device, dtype=grads[0].dtype)

    for g in grads:
        sq_sum = sq_sum + g.reshape(E, -1).pow(2).sum(dim=1)

    norm = torch.sqrt(sq_sum)                                    # [E]
    coef = (float(max_norm) / (norm + eps)).clamp(max=1.0)       # [E]

    for g in grads:
        shape = [E] + [1] * (g.dim() - 1)
        g.mul_(coef.view(*shape))

    return norm


# =============================================================================
# Ensemble
# =============================================================================

class LocalCounterfactualProxyEnsemble:
    """
    Ensemble proxy có dấu, doubly-robust, đa horizon, chạy trên GPU dưới
    dạng MỘT tensor có chiều ensemble (không phải n_ensemble model rời rạc
    được lặp bằng Python).

    ---------------------------------------------------------------------
    BỐN CHẾ ĐỘ TÍNH EFFECT (effect_mode)
    ---------------------------------------------------------------------
    "signed_aristocrat"  (MẶC ĐỊNH — dùng cho gán vai trò Thiện/Ác)
        w = f(s, a_j_obs) - E_{a' ~ pi_j}[ f(s, a') ]
        w > 0: hành động thực của j TỐT cho i hơn mức trung bình -> j GIÚP
        w < 0: j đang HẠI

    "signed_oracle_matched"  (dùng cho Exp3 calibration)
        w = mean_{a in candidates}[ f(s,a) ] - f(s, a_j_obs)
        Khớp ĐÚNG công thức oracle trong env.

    "range"  (Pieroth ICML'24 style — baseline đối chứng)
        w = max_a f(s,a) - min_a f(s,a)   (luôn >= 0)

    "mean_abs"  (bản v1 — giữ để chạy ablation "trước/sau khi vá")
        w = mean_{a != a_obs} |f(s,a) - f(s,a_obs)|

    ---------------------------------------------------------------------
    DOUBLY ROBUST
    ---------------------------------------------------------------------
        psi_DR(a) = f_hat(s,a) + (1{a_obs = a} / b_j(a|s)) * (R_obs - f_hat(s,a_obs))

    Chỉ cần MỘT trong hai (outcome model HOẶC propensity) đúng là không chệch.
    Trong MARL ta biết chính xác propensity -> luôn có sẵn một cái đúng.
    Importance weight 1/b bị CLIP để tránh nổ phương sai khi b nhỏ.
    """

    # Bốn chế độ hợp lệ
    MODES = (
        "signed_aristocrat",
        "signed_oracle_matched",
        "range",
        "mean_abs",
    )
    MODE_ALIASES = {
        "unsigned_range": "range",
    }

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        core_dim: int,
        periph_dim: int,
        belief_dim: int,
        n_ensemble: int = 4,
        hidden: int = 160,
        lr: float = 1e-3,
        buffer_size: int = 200000,
        device: str = "cpu",
        grad_clip: float = 1.0,
        eps: float = 1e-8,
        n_horizons: int = 8,
        effect_mode: str = "signed_aristocrat",
        use_doubly_robust: bool = True,
        iw_clip: float = 10.0,
        bootstrap_ratio: float = 0.8,
        use_belief_input: bool = False,
        candidate_actions: Optional[List[int]] = None,
        ensemble_dropout: float = 0.0,
        seed: int = 0,
        use_vmap_ensemble: bool = True,
        compile_ensemble: bool = False,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.core_dim = int(core_dim)
        self.periph_dim = int(periph_dim)
        self.belief_dim = int(belief_dim)
        self.n_ensemble = int(n_ensemble)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.buffer_size = int(buffer_size)
        self.device = device
        self.grad_clip = float(grad_clip)
        self.eps = float(eps)

        self.n_horizons = int(n_horizons)

        effect_mode = self.MODE_ALIASES.get(str(effect_mode), str(effect_mode))
        if effect_mode not in self.MODES:
            raise ValueError(
                f"effect_mode phải thuộc {self.MODES}, nhận '{effect_mode}'"
            )
        self.effect_mode = str(effect_mode)

        self.use_doubly_robust = bool(use_doubly_robust)
        self.iw_clip = float(iw_clip)
        self.bootstrap_ratio = float(np.clip(bootstrap_ratio, 0.1, 1.0))
        self.use_belief_input = bool(use_belief_input)

        self.candidate_actions = (
            list(range(self.action_dim))
            if candidate_actions is None
            else [int(a) for a in candidate_actions]
        )

        # ---------------------------------------------------------------
        # [L3] ENSEMBLE ĐA DẠNG THẬT, chạy như MỘT tensor có chiều E.
        # Ba nguồn đa dạng:
        #   1. Khởi tạo khác nhau (torch seed khác nhau mỗi member)
        #   2. Bootstrap mask khác nhau (mỗi member chỉ thấy 1 phần buffer)
        #   3. Batch khác nhau (sample riêng cho từng member)
        # ---------------------------------------------------------------
        self.models: List[LocalCounterfactualProxyNet] = []

        for k in range(self.n_ensemble):
            torch.manual_seed(int(seed) * 1000 + k)

            self.models.append(
                LocalCounterfactualProxyNet(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    core_dim=self.core_dim,
                    periph_dim=self.periph_dim,
                    belief_dim=self.belief_dim,
                    hidden=self.hidden,
                    n_horizons=self.n_horizons,
                    use_belief_input=self.use_belief_input,
                    dropout=float(ensemble_dropout),
                ).to(self.device)
            )

        self.use_vmap_ensemble = bool(use_vmap_ensemble) and _HAS_TORCH_FUNC
        self._compile_ensemble_flag = bool(compile_ensemble)

        if self.use_vmap_ensemble:
            self._setup_vmap_ensemble()
        else:
            # Fallback cho torch < 2.0 (không có torch.func): giữ hành vi
            # kiểu vòng lặp cũ, đúng nhưng không tận dụng GPU triệt để.
            self.optims = [
                torch.optim.Adam(m.parameters(), lr=self.lr) for m in self.models
            ]

        # RNG riêng cho từng member để oversample có trọng số độc lập.
        self._member_rngs = [
            random.Random(int(seed) * 7919 + k) for k in range(self.n_ensemble)
        ]

        # ---------------------------------------------------------------
        # [BB1 — GPU_OPTIMIZATION_CONTRACT.md] Bootstrap mask CỐ ĐỊNH,
        # KHÔNG rút lại ngẫu nhiên mỗi lần gọi.
        #
        # Nếu mỗi train_step() tự rút một pool con MỚI cho từng member (bản
        # trước làm vậy để tránh quét toàn buffer bằng Python), thì về lâu
        # dài mọi member đều đã "thấy" gần như toàn bộ buffer, chỉ khác
        # nhau ở minibatch cụ thể tại mỗi bước — không còn sự khác biệt HỆ
        # THỐNG giữa các member -> chúng hội tụ về gần cùng một hàm và
        # sigma (Eq. 10) suy biến về 0, đúng lỗi của bản v1.
        #
        # Sửa: mỗi member có một permutation CỐ ĐỊNH (tính một lần ở đây,
        # seed riêng theo member) đánh dấu ~bootstrap_ratio phần trăm VỊ TRÍ
        # (rank trong buffer, không phải sample cụ thể — buffer là deque
        # maxlen nên sample dịch chuyển qua vị trí theo thời gian, nhưng
        # tập RANK mà một member luôn quan sát thì cố định) mà member đó
        # được phép thấy. Member k luôn loại trừ đúng phần buffer đó, mọi
        # lúc mọi nơi -> khác biệt hệ thống được duy trì, không phai theo
        # thời gian.
        # ---------------------------------------------------------------
        self._member_pool_mask: List[np.ndarray] = []
        keep_n = max(1, int(self.buffer_size * self.bootstrap_ratio))

        for k in range(self.n_ensemble):
            rng_k = np.random.RandomState(int(seed) * 7919 + k)
            perm = rng_k.permutation(self.buffer_size)
            mask = np.zeros(self.buffer_size, dtype=bool)
            mask[perm[:keep_n]] = True
            self._member_pool_mask.append(mask)

        self.buffer = deque(maxlen=self.buffer_size)

        # Diagnostics (runner cũ đang đọc các field này)
        self.last_train_called = False
        self.last_train_batch_count = 0
        self.latest_residual = 0.0
        self.latest_train_residual = 0.0
        self.latest_holdout_residual = 0.0
        self.latest_loss = 0.0

        # Diagnostics mới
        self.latest_ensemble_disagreement = 0.0
        self.latest_dr_correction_magnitude = 0.0
        self.n_interventional_samples = 0

        # [BB3 — GPU_OPTIMIZATION_CONTRACT.md] loss RIÊNG từng member, để
        # test T3 so được "thang gradient member 0" giữa E=1 và E=4 mà
        # không cần lục lại đồ thị autograd. None cho tới lần train đầu.
        self.latest_loss_per_member: Optional[np.ndarray] = None

    # =====================================================================
    # Thiết lập đường vmap cho ensemble
    # =====================================================================

    def _setup_vmap_ensemble(self):
        """
        Gộp trọng số của n_ensemble model thành MỘT cây tensor có thêm
        chiều đầu E, rồi định nghĩa 2 hàm forward vmap:

          _vmap_forward_shared:
              params/buffers có chiều E, DỮ LIỆU DÙNG CHUNG cho mọi member
              (in_dims data = None -> broadcast). Dùng khi ta muốn ensemble
              cùng đánh giá trên một batch — inference (_predict_all_actions),
              holdout eval.

          _vmap_forward_per_member:
              params/buffers có chiều E, DỮ LIỆU CŨNG có chiều E riêng
              (mỗi member một batch, cho bootstrap diversity thật). Dùng
              khi train.

        `self._base_model` chỉ đóng vai trò KIẾN TRÚC MẪU cho functional_call
        — trọng số thật nằm hết trong self._stacked_params (yêu cầu grad).
        self.models[k].parameters() sau bước này không còn là nguồn sự thật
        (không ai đọc lại chúng ở nơi khác trong codebase, đã kiểm tra).
        """
        self._base_model = self.models[0]

        stacked_params, stacked_buffers = stack_module_state(self.models)

        self._stacked_params: Dict[str, torch.Tensor] = {
            k: v.detach().clone().requires_grad_(True)
            for k, v in stacked_params.items()
        }
        self._stacked_buffers: Dict[str, torch.Tensor] = dict(stacked_buffers)

        self.optim = torch.optim.Adam(
            list(self._stacked_params.values()), lr=self.lr
        )

        def _fmodel(params, buffers, obs_i, a_i_oh, a_j_oh, z, m, belief):
            return functional_call(
                self._base_model,
                (params, buffers),
                args=(),
                kwargs=dict(
                    obs_i=obs_i,
                    action_i_onehot=a_i_oh,
                    action_j_onehot=a_j_oh,
                    z_core_excl_j=z,
                    m_periph_excl_j=m,
                    belief_summary=belief,
                ),
            )

        # randomness="different": nếu sau này ensemble_dropout > 0, mỗi
        # member phải rút mask dropout ĐỘC LẬP (đúng ý nghĩa ensemble).
        # Mặc định vmap sẽ raise lỗi nếu gặp toán tử ngẫu nhiên mà không
        # khai báo rõ randomness -> khai báo sẵn ở đây để không sập khi có
        # người bật ensemble_dropout > 0 sau này.
        self._vmap_forward_shared = vmap(
            _fmodel, in_dims=(0, 0, None, None, None, None, None, None),
            randomness="different",
        )
        self._vmap_forward_per_member = vmap(
            _fmodel, in_dims=(0, 0, 0, 0, 0, 0, 0, 0),
            randomness="different",
        )

        # ---------------------------------------------------------------
        # torch.compile — TẮT MẶC ĐỊNH, bật thủ công qua compile_ensemble=True.
        #
        # vmap ở chế độ eager KHÔNG tự gộp kernel: mỗi Linear/ReLU bên trong
        # vẫn launch kernel CUDA riêng, chỉ thêm một chiều batch. Với mạng
        # nhỏ (hidden=160) thì overhead LAUNCH kernel (vài chục µs cố định
        # mỗi lần, không phụ thuộc kích thước tensor) áp đảo thời gian tính
        # thật -> GPU có thể CHẬM HƠN CPU, đúng triệu chứng đang gặp
        # (12.5 trên CUDA vs ~50 trên Mac). torch.compile(mode="reduce-
        # overhead") dùng CUDA graphs để gộp toàn bộ chuỗi kernel thành một
        # lần launch, đây là cách sửa chuẩn cho đúng triệu chứng này.
        #
        # KHÔNG bật mặc định vì: (a) cần shape batch CỐ ĐỊNH (batch_size,
        # holdout_size trong cfg không đổi giữa các lần gọi — kiểm tra
        # trước khi bật), batch B*A trong _predict_all_actions ĐỔI theo B
        # nên nhánh đó dễ bị recompile liên tục nếu bật; (b) chưa verify
        # được trên GPU thật trong môi trường này. Bật thử ở máy bạn, đo
        # throughput trước/sau, và trước hết nên compile riêng
        # _vmap_forward_per_member (dùng cho train_step, batch cố định)
        # chứ đừng compile _vmap_forward_shared (dùng cho
        # _predict_all_actions, batch đổi theo B).
        # ---------------------------------------------------------------
        if bool(getattr(self, "_compile_ensemble_flag", False)):
            self._vmap_forward_per_member = torch.compile(
                self._vmap_forward_per_member, mode="reduce-overhead"
            )

    def _ensemble_train_mode(self, training: bool):
        """Bật/tắt training mode (ảnh hưởng Dropout) cho kiến trúc mẫu.
        Dùng chung một flag cho mọi member vì kiến trúc giống hệt nhau."""
        if self.use_vmap_ensemble:
            self._base_model.train(training)
        else:
            for m in self.models:
                m.train(training)

    # =====================================================================
    # Helper tensor
    # =====================================================================

    def _one_hot(self, actions) -> torch.Tensor:
        """actions: array-like [B] -> [B, action_dim] float32"""
        if isinstance(actions, torch.Tensor):
            a = actions.to(device=self.device, dtype=torch.long)
        else:
            a = torch.tensor(
                np.asarray(actions, dtype=np.int64),
                dtype=torch.long,
                device=self.device,
            )

        if a.dim() == 0:
            a = a.unsqueeze(0)

        a = a.clamp(min=0, max=self.action_dim - 1)

        return F.one_hot(a, num_classes=self.action_dim).to(dtype=torch.float32)

    def _to_float_tensor(self, x, expected_dim=None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x.to(device=self.device, dtype=torch.float32)
        else:
            t = torch.tensor(
                np.asarray(x, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )

        if t.dim() == 1:
            t = t.unsqueeze(0)

        if expected_dim is not None and t.shape[-1] != int(expected_dim):
            raise ValueError(
                f"Expected last dim={expected_dim}, got {t.shape[-1]}"
            )

        return t

    def _normalise_vector(self, x, expected_dim) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)

        if arr.shape[0] != int(expected_dim):
            raise ValueError(
                f"Expected vector dim={expected_dim}, got {arr.shape[0]}"
            )

        return arr.astype(np.float32)

    # =====================================================================
    # Buffer
    # =====================================================================

    def add_sample(
        self,
        ego_id,
        neighbor_id,
        obs_i,
        action_i,
        observed_action_j,
        z_core_excl_j,
        m_periph_excl_j,
        belief_summary,
        target_return_h,
        target_returns_multi=None,
        behaviour_prob_j=None,
        was_forced=False,
        state_key=None,
    ):
        """
        Thêm một mẫu supervised.

        Args:
            target_returns_multi:
                list/array length n_horizons = [R^(1), R^(2), ..., R^(H)].
                Nếu None, broadcast target_return_h ra mọi horizon (kém
                chính xác, chỉ để tương thích ngược — nên truyền vào).
            behaviour_prob_j:
                b_j(a_j_obs | s) tại thời điểm thu thập — CẦN cho DR.
                Nếu None, DR sẽ tự tắt cho mẫu này (fallback plug-in).
            was_forced:
                True nếu hành động của j bị eps-forcing ép -> can thiệp
                THẬT, đáng giá hơn -> được oversample khi train.
            state_key:
                định danh ngữ cảnh (zone id / hash vị trí thô).
        """
        if target_returns_multi is None:
            multi = np.full(
                (self.n_horizons,), float(target_return_h), dtype=np.float32
            )
        else:
            multi = np.asarray(target_returns_multi, dtype=np.float32).reshape(-1)

            if multi.shape[0] != self.n_horizons:
                raise ValueError(
                    f"target_returns_multi phải có length {self.n_horizons}, "
                    f"nhận {multi.shape[0]}"
                )

        sample = {
            "ego_id": int(ego_id),
            "neighbor_id": int(neighbor_id),
            "obs_i": self._normalise_vector(obs_i, self.obs_dim),
            "action_i": int(action_i),
            "observed_action_j": int(observed_action_j),
            "z_core_excl_j": self._normalise_vector(z_core_excl_j, self.core_dim),
            "m_periph_excl_j": self._normalise_vector(
                m_periph_excl_j, self.periph_dim
            ),
            "belief_summary": self._normalise_vector(
                belief_summary, self.belief_dim
            ),
            "target_return_h": float(target_return_h),
            "target_returns_multi": multi,                      # [n_horizons]
            "behaviour_prob_j": (
                None if behaviour_prob_j is None else float(behaviour_prob_j)
            ),
            "was_forced": bool(was_forced),
            "state_key": state_key,
        }

        self.buffer.append(sample)

        if bool(was_forced):
            self.n_interventional_samples += 1

    def add_sample_batch(self, samples: List[dict]):
        """
        Thêm nhiều mẫu cùng lúc — giảm overhead gọi hàm khi runner đẩy cả
        một trajectory (O(n_agents^2) mẫu mỗi timestep). Mỗi phần tử phải
        có đúng các key mà add_sample nhận (dùng **s cho gọn ở call site).

        Đây KHÔNG phải "đổi buffer sang tensor" — buffer vẫn là deque[dict]
        vì drift_probe.py phụ thuộc trực tiếp vào định dạng đó. Cái được
        loại bỏ ở đây là chi phí gọi hàm Python `add_sample(...)` riêng lẻ
        n_agents^2 lần mỗi bước — thay bằng một vòng lặp append() rẻ.
        """
        for s in samples:
            self.add_sample(**s)

    def get_buffer_size(self) -> int:
        return int(len(self.buffer))

    def get_last_train_called(self) -> bool:
        return bool(self.last_train_called)

    def get_last_train_batch_count(self) -> int:
        return int(self.last_train_batch_count)

    def get_latest_residual(self) -> float:
        return float(self.latest_residual)

    def get_latest_train_residual(self) -> float:
        return float(self.latest_train_residual)

    def get_latest_holdout_residual(self) -> float:
        return float(self.latest_holdout_residual)

    # =====================================================================
    # Train
    # =====================================================================

    def _batch_to_tensors(self, batch):
        """batch: list[dict] length B -> tuple tensors"""
        obs = np.stack([b["obs_i"] for b in batch], axis=0)          # [B, obs_dim]
        action_i = np.asarray([b["action_i"] for b in batch], np.int64)      # [B]
        action_j = np.asarray(
            [b["observed_action_j"] for b in batch], np.int64
        )                                                                     # [B]
        z = np.stack([b["z_core_excl_j"] for b in batch], axis=0)    # [B, core_dim]
        m = np.stack([b["m_periph_excl_j"] for b in batch], axis=0)  # [B, periph_dim]
        belief = np.stack(
            [b["belief_summary"] for b in batch], axis=0
        )                                                            # [B, belief_dim]
        target_multi = np.stack(
            [b["target_returns_multi"] for b in batch], axis=0
        )                                                            # [B, n_horizons]

        return (
            torch.tensor(obs, dtype=torch.float32, device=self.device),
            self._one_hot(action_i),
            self._one_hot(action_j),
            torch.tensor(z, dtype=torch.float32, device=self.device),
            torch.tensor(m, dtype=torch.float32, device=self.device),
            torch.tensor(belief, dtype=torch.float32, device=self.device),
            torch.tensor(target_multi, dtype=torch.float32, device=self.device),
        )

    def _sample_for_member(self, buf_list: list, member_idx: int, n: int,
                            forced_boost: float = 3.0):
        """
        [L3] Sample RIÊNG cho từng ensemble member — KHÔNG quét toàn bộ
        buffer bằng vòng lặp Python (bản cũ hash từng phần tử, O(buffer_size)
        mỗi member mỗi train_step, tức O(n_ensemble * buffer_size) mỗi
        train_step — với buffer 200k và 4 member là 800k lượt tính Python
        thuần, KHÔNG liên quan gì tới GPU nhưng chiếm phần lớn wall-clock).

        [BB1] Pool con của mỗi member được rút từ `self._member_pool_mask
        [member_idx]` — một permutation CỐ ĐỊNH tính một lần ở __init__,
        KHÔNG rút lại ngẫu nhiên ở đây. Nếu rút ngẫu nhiên mỗi lần gọi, về
        lâu dài mọi member "thấy" gần hết buffer và không còn khác biệt hệ
        thống -> ensemble hội tụ về nhau, sigma (Eq. 10) suy biến về 0 —
        đúng lỗi của bản v1 mà toàn bộ nhánh uncertainty-aware của paper
        (LCB, selectivity, targeted-eps, bơm phồng) phụ thuộc vào việc
        tránh được. Mask cố định giữ chi phí O(pool_size) (numpy, không
        phải Python loop) NHƯNG vẫn đảm bảo member k luôn loại trừ đúng
        một tập rank cố định của buffer, mọi lúc.

        `buf_list` được truyền vào từ ngoài (list(self.buffer) một lần duy
        nhất mỗi train_step, dùng chung cho mọi member) vì `deque` không
        hỗ trợ random-access O(1).
        """
        if len(buf_list) == 0:
            return []

        rng = self._member_rngs[member_idx]

        mask = self._member_pool_mask[member_idx][: len(buf_list)]
        pool_positions = np.nonzero(mask)[0]

        if pool_positions.size == 0:
            # Chỉ xảy ra khi buffer còn rất nhỏ (đầu training) và permutation
            # tình cờ không rơi vào prefix ngắn đó -> fallback dùng cả
            # buf_list hiện có, KHÔNG bao giờ trả rỗng.
            pool_positions = np.arange(len(buf_list))

        pool = [buf_list[int(i)] for i in pool_positions]
        weights = [forced_boost if s["was_forced"] else 1.0 for s in pool]

        # random.choices lấy có hoàn lại, có trọng số -> luôn trả đúng n
        # mẫu (kể cả khi pool nhỏ hơn n, ví dụ đầu training) -> mọi member
        # luôn có cùng batch size -> stack được thành tensor [E, B, ...].
        # Trọng số oversample cho was_forced=True (BB2) được giữ nguyên ở
        # đây — đừng thay bằng random.sample/slicing thuần, sẽ mất oversample.
        return rng.choices(pool, weights=weights, k=int(n))

    def train_step(
        self,
        n_steps: int = 1,
        batch_size: int = 256,
        holdout_size: int = 0,
    ) -> float:
        """
        Train ensemble. Giữ nguyên chữ ký v1.

        Khác trước ở chỗ: cả n_ensemble member forward/backward/update
        trong MỘT lệnh vmap thay vì vòng lặp Python `for model in models`,
        và mọi số liệu chẩn đoán được tích trên GPU rồi CHỈ đồng bộ về CPU
        MỘT LẦN ở cuối hàm (thay vì mỗi bước mỗi member một lần `.item()`).
        """
        self.last_train_called = True
        self.last_train_batch_count = 0

        if len(self.buffer) == 0:
            self.latest_loss = 0.0
            self.latest_residual = 0.0
            self.latest_train_residual = 0.0
            self.latest_holdout_residual = 0.0
            return 0.0

        n_steps = int(max(0, n_steps))
        batch_size = int(max(1, batch_size))
        holdout_size = int(max(0, holdout_size))

        if n_steps == 0:
            self.latest_loss = 0.0
            return 0.0

        if not self.use_vmap_ensemble:
            return self._train_step_fallback(n_steps, batch_size, holdout_size)

        self._ensemble_train_mode(True)

        E = self.n_ensemble
        per_step_losses = []       # list of [E] tensors, sync một lần ở cuối
        per_step_residuals = []    # list of [E] tensors

        for _ in range(n_steps):
            buf_list = list(self.buffer)  # một lần copy deque->list mỗi step

            member_batches = [
                self._sample_for_member(buf_list, k, batch_size)
                for k in range(E)
            ]

            print(f"[TRAIN-DEBUG] batch_size={batch_size} " f"member_batch_lens={[len(b) for b in member_batches]}")

            if any(len(b) == 0 for b in member_batches):
                print("[TRAIN-DEBUG] SKIPPED — empty batch this step")
                continue

            if any(len(b) == 0 for b in member_batches):
                continue

            obs_l, ai_l, aj_l, z_l, m_l, bl_l, tgt_l = [], [], [], [], [], [], []

            for b in member_batches:
                (obs_t, a_i_oh, a_j_oh, z_t, m_t, belief_t, target_multi_t) = (
                    self._batch_to_tensors(b)
                )
                obs_l.append(obs_t)
                ai_l.append(a_i_oh)
                aj_l.append(a_j_oh)
                z_l.append(z_t)
                m_l.append(m_t)
                bl_l.append(belief_t)
                tgt_l.append(target_multi_t)

            obs_e = torch.stack(obs_l, dim=0)    # [E, B, obs_dim]
            ai_e = torch.stack(ai_l, dim=0)      # [E, B, A]
            aj_e = torch.stack(aj_l, dim=0)      # [E, B, A]
            z_e = torch.stack(z_l, dim=0)        # [E, B, core_dim]
            m_e = torch.stack(m_l, dim=0)        # [E, B, periph_dim]
            bel_e = torch.stack(bl_l, dim=0)     # [E, B, belief_dim]
            tgt_e = torch.stack(tgt_l, dim=0)    # [E, B, H]

            # MỘT lệnh vmap = n_ensemble forward pass chạy song song trên GPU.
            preds = self._vmap_forward_per_member(
                self._stacked_params, self._stacked_buffers,
                obs_e, ai_e, aj_e, z_e, m_e, bel_e,
            )  # [E, B, H]

            per_member_loss = F.mse_loss(preds, tgt_e, reduction="none").mean(
                dim=(1, 2)
            )  # [E] — mỗi member có loss riêng, không trộn lẫn

            loss = per_member_loss.sum()  # backward của tổng các thành phần
            # ĐỘC LẬP <=> mỗi thành phần chỉ lan gradient về đúng member đó
            # (vmap đảm bảo không có phép toán nào trộn chiều E).

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            _clip_grad_norm_per_member(self._stacked_params, self.grad_clip)
            self.optim.step()

            per_step_losses.append(per_member_loss.detach())

            with torch.no_grad():
                res = torch.mean(
                    torch.abs(preds[:, :, -1] - tgt_e[:, :, -1]), dim=1
                )  # [E] — residual đo trên horizon cuối (khớp R^(H) của v1)
                forced_mask = torch.tensor([b["was_forced"] for b in member_batches[0]], device=preds.device)
                if forced_mask.any():
                    res_forced = torch.mean(torch.abs(preds[:, forced_mask, -1] - tgt_e[:, forced_mask, -1]))
                    res_control = torch.mean(torch.abs(preds[:, ~forced_mask, -1] - tgt_e[:, ~forced_mask, -1])) if (~forced_mask).any() else torch.tensor(0.0)
                    print(f"[RESIDUAL-SPLIT] forced_n={forced_mask.sum().item()} res_forced={res_forced.item():.4e} res_control={res_control.item():.4e}")
            per_step_residuals.append(res)

            self.last_train_batch_count += 1

        # ---- holdout residual: batch chung, KHÔNG dùng để update ----------
        # [H7] Residual phải đo trên dữ liệu không tham gia gradient, nếu
        # không nó phản ánh chính sự thay đổi của mình chứ không phải
        # structural shift.
        holdout_residual_t = None

        if holdout_size > 0 and len(self.buffer) > holdout_size:
            ho_batch = random.sample(list(self.buffer), int(holdout_size))

            (ho_obs, ho_ai, ho_aj, ho_z, ho_m, ho_b, ho_target) = (
                self._batch_to_tensors(ho_batch)
            )

            self._ensemble_train_mode(False)

            with torch.no_grad():
                stacked = self._vmap_forward_shared(
                    self._stacked_params, self._stacked_buffers,
                    ho_obs, ho_ai, ho_aj, ho_z, ho_m, ho_b,
                )  # [E, B, H] — MỘT lệnh vmap, dữ liệu dùng chung mọi member

                pred_mean = stacked.mean(dim=0)  # [B, H]

                holdout_residual_t = torch.mean(
                    torch.abs(pred_mean[:, -1] - ho_target[:, -1])
                )

                # [L3] Chẩn đoán: ensemble có thật sự bất đồng không?
                # Nếu số này ~ 0 thì ensemble đang giả, sigma vô nghĩa.
                if stacked.shape[0] > 1:
                    self.latest_ensemble_disagreement = float(
                        torch.mean(torch.std(stacked, dim=0)).item()
                    )  # 1 sync duy nhất cho cả holdout eval

        if len(per_step_losses) == 0:
            print("[TRAIN-DEBUG] ALL n_steps SKIPPED — per_step_losses rỗng")
            self.latest_loss = 0.0
            self.latest_residual = 0.0
            self.latest_train_residual = 0.0
            self.latest_holdout_residual = 0.0
            return 0.0

        # ---- MỘT sync CPU<->GPU cho toàn bộ train_step() call --------------
        # (bản cũ: n_steps * n_ensemble lần .item() cho loss + như vậy nữa
        # cho residual; ở đây: 1 lần cho loss, 1 lần cho residual, bất kể
        # n_steps/n_ensemble lớn cỡ nào.)
        losses_stacked = torch.stack(per_step_losses)  # [n_steps_thật, E]
        self.latest_loss = float(losses_stacked.mean().item())
        # [BB3] Loss RIÊNG từng member (mean qua các step) — để test T3 xác
        # nhận thang gradient của member 0 không đổi khi thêm member khác.
        self.latest_loss_per_member = (
            losses_stacked.mean(dim=0).detach().cpu().numpy()
        )
        self.latest_train_residual = float(
            torch.stack(per_step_residuals).mean().item()
        )

        if holdout_residual_t is not None:
            self.latest_holdout_residual = float(holdout_residual_t.item())
            self.latest_residual = self.latest_holdout_residual
        else:
            self.latest_holdout_residual = self.latest_train_residual
            self.latest_residual = self.latest_train_residual

        return float(self.latest_loss)

    # ---- fallback nếu torch < 2.0 (không có torch.func) -------------------
    def _train_step_fallback(self, n_steps, batch_size, holdout_size):
        """Đường vòng lặp Python cũ — chỉ dùng khi môi trường không có
        torch.func (torch < 2.0). Đúng về mặt số học, không tối ưu GPU."""
        all_losses = []
        train_residuals = []

        for _ in range(n_steps):
            buf_list = list(self.buffer)
            for k, (model, optim) in enumerate(zip(self.models, self.optims)):
                batch = self._sample_for_member(buf_list, k, batch_size)

                if len(batch) == 0:
                    continue

                (obs_t, a_i_oh, a_j_oh, z_t, m_t, belief_t, target_multi_t) = (
                    self._batch_to_tensors(batch)
                )

                model.train()

                pred = model(
                    obs_i=obs_t,
                    action_i_onehot=a_i_oh,
                    action_j_onehot=a_j_oh,
                    z_core_excl_j=z_t,
                    m_periph_excl_j=m_t,
                    belief_summary=belief_t,
                )

                loss = F.mse_loss(pred, target_multi_t)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optim.step()

                all_losses.append(loss.detach())

                with torch.no_grad():
                    res = torch.mean(torch.abs(pred[:, -1] - target_multi_t[:, -1]))
                train_residuals.append(res)

            self.last_train_batch_count += 1

        holdout_residual = None

        if holdout_size > 0 and len(self.buffer) > holdout_size:
            ho_batch = random.sample(list(self.buffer), int(holdout_size))
            (ho_obs, ho_ai, ho_aj, ho_z, ho_m, ho_b, ho_target) = (
                self._batch_to_tensors(ho_batch)
            )

            with torch.no_grad():
                preds = []
                for model in self.models:
                    model.eval()
                    preds.append(
                        model(
                            obs_i=ho_obs, action_i_onehot=ho_ai,
                            action_j_onehot=ho_aj, z_core_excl_j=ho_z,
                            m_periph_excl_j=ho_m, belief_summary=ho_b,
                        )
                    )

                stacked = torch.stack(preds, dim=0)
                pred_mean = stacked.mean(dim=0)

                holdout_residual = float(
                    torch.mean(
                        torch.abs(pred_mean[:, -1] - ho_target[:, -1])
                    ).item()
                )

                if stacked.shape[0] > 1:
                    self.latest_ensemble_disagreement = float(
                        torch.mean(torch.std(stacked, dim=0)).item()
                    )

        if len(all_losses) == 0:
            self.latest_loss = 0.0
            self.latest_residual = 0.0
            return 0.0

        self.latest_loss = float(torch.stack(all_losses).mean().item())
        self.latest_train_residual = (
            float(torch.stack(train_residuals).mean().item())
            if train_residuals else 0.0
        )

        if holdout_residual is not None:
            self.latest_holdout_residual = holdout_residual
            self.latest_residual = holdout_residual
        else:
            self.latest_holdout_residual = self.latest_train_residual
            self.latest_residual = self.latest_train_residual

        return float(self.latest_loss)

    # =====================================================================
    # Dự đoán mọi hành động thay thế
    # =====================================================================

    def _predict_all_actions(
        self,
        obs,       # [B, obs_dim]
        action_i,  # [B]
        z,         # [B, core_dim]
        m,         # [B, periph_dim]
        belief,    # [B, belief_dim]
    ) -> torch.Tensor:
        """
        Dự đoán return cho MỌI hành động khả dĩ của j, với mọi ensemble
        member, trong ĐÚNG MỘT forward pass GPU.

        Hai tầng gộp batch:
          1. (đã có từ trước) Gộp mọi hành động thay thế của j vào một
             batch lớn thay vì gọi forward action_dim lần.
          2. (mới) Gộp luôn cả chiều ensemble bằng vmap thay vì lặp qua
             từng model.

        Returns:
            [E, B, A, n_horizons]
        """
        obs_t = self._to_float_tensor(obs, self.obs_dim)        # [B, obs_dim]
        z_t = self._to_float_tensor(z, self.core_dim)           # [B, core_dim]
        m_t = self._to_float_tensor(m, self.periph_dim)         # [B, periph_dim]
        belief_t = self._to_float_tensor(belief, self.belief_dim)  # [B, belief_dim]

        B = int(obs_t.shape[0])
        A = int(self.action_dim)

        a_i_oh = self._one_hot(np.asarray(action_i).reshape(-1))  # [B, A]

        # Nhân bản mỗi sample A lần, mỗi bản gán một hành động khác của j.
        obs_rep = obs_t.repeat_interleave(A, dim=0)        # [B*A, obs_dim]
        z_rep = z_t.repeat_interleave(A, dim=0)            # [B*A, core_dim]
        m_rep = m_t.repeat_interleave(A, dim=0)            # [B*A, periph_dim]
        belief_rep = belief_t.repeat_interleave(A, dim=0)  # [B*A, belief_dim]
        a_i_rep = a_i_oh.repeat_interleave(A, dim=0)       # [B*A, A]

        eye = torch.eye(A, dtype=torch.float32, device=self.device)  # [A, A]
        a_j_rep = eye.repeat(B, 1)                                    # [B*A, A]

        self._ensemble_train_mode(False)

        with torch.no_grad():
            if self.use_vmap_ensemble:
                preds = self._vmap_forward_shared(
                    self._stacked_params, self._stacked_buffers,
                    obs_rep, a_i_rep, a_j_rep, z_rep, m_rep, belief_rep,
                )  # [E, B*A, n_horizons] — MỘT lệnh cho cả ensemble
                out = preds.view(self.n_ensemble, B, A, self.n_horizons)
            else:
                out = self._predict_all_actions_loop(
                    obs_rep, a_i_rep, a_j_rep, z_rep, m_rep, belief_rep, B, A,
                )

        return out  # [E, B, A, n_horizons]

    def _predict_all_actions_loop(
        self, obs_rep, a_i_rep, a_j_rep, z_rep, m_rep, belief_rep, B, A,
    ) -> torch.Tensor:
        """Vòng lặp Python qua từng model — dùng làm fallback (torch<2.0)
        VÀ làm bản tham chiếu cho test T1 (xem _predict_all_actions_reference).
        Không dùng trong đường nóng khi use_vmap_ensemble=True."""
        outs = []
        for model in self.models:
            model.eval()
            pred = model(
                obs_i=obs_rep, action_i_onehot=a_i_rep,
                action_j_onehot=a_j_rep, z_core_excl_j=z_rep,
                m_periph_excl_j=m_rep, belief_summary=belief_rep,
            )
            outs.append(pred.view(B, A, self.n_horizons))
        return torch.stack(outs, dim=0)

    def _sync_stacked_to_models(self):
        """Copy trọng số từ self._stacked_params (nguồn sự thật khi
        use_vmap_ensemble=True) ngược lại vào self.models[k]. CHỈ dùng cho
        test/debug (_predict_all_actions_reference) — không nằm trong
        đường train/inference thật, chấp nhận vòng lặp Python ở đây."""
        if not self.use_vmap_ensemble:
            return
        with torch.no_grad():
            for k, model in enumerate(self.models):
                sd = model.state_dict()
                for name, p in self._stacked_params.items():
                    if name in sd:
                        sd[name].copy_(p[k])

    def _predict_all_actions_reference(
        self, obs, action_i, z, m, belief,
    ) -> torch.Tensor:
        """
        [GPU_OPTIMIZATION_CONTRACT.md — quy tắc vàng ở Phần 3] Bản THAM
        CHIẾU chậm (vòng lặp Python qua từng model, không vmap), đồng bộ
        đúng trọng số hiện tại từ self._stacked_params. Dùng trong smoke
        test để assert allclose với _predict_all_actions (đường vmap
        nhanh) — đây là cách duy nhất bắt được lỗi kiểu BB4 (thứ tự
        repeat_interleave/repeat/view bị đảo): cả hai đường đều cho số
        thực "trông hợp lý", sai lệch chỉ lộ ra khi so hai bản với nhau.

        KHÔNG dùng hàm này trong production — nó tồn tại chỉ để test.
        """
        self._sync_stacked_to_models()

        obs_t = self._to_float_tensor(obs, self.obs_dim)
        z_t = self._to_float_tensor(z, self.core_dim)
        m_t = self._to_float_tensor(m, self.periph_dim)
        belief_t = self._to_float_tensor(belief, self.belief_dim)

        B = int(obs_t.shape[0])
        A = int(self.action_dim)

        a_i_oh = self._one_hot(np.asarray(action_i).reshape(-1))

        obs_rep = obs_t.repeat_interleave(A, dim=0)
        z_rep = z_t.repeat_interleave(A, dim=0)
        m_rep = m_t.repeat_interleave(A, dim=0)
        belief_rep = belief_t.repeat_interleave(A, dim=0)
        a_i_rep = a_i_oh.repeat_interleave(A, dim=0)

        eye = torch.eye(A, dtype=torch.float32, device=self.device)
        a_j_rep = eye.repeat(B, 1)

        with torch.no_grad():
            out = self._predict_all_actions_loop(
                obs_rep, a_i_rep, a_j_rep, z_rep, m_rep, belief_rep, B, A,
            )

        return out  # [E, B, A, n_horizons]

    # =====================================================================
    # Tính effect
    # =====================================================================

    def _compute_effects(
        self,
        preds_all: torch.Tensor,        # [E, B, A, n_horizons]
        action_j_obs: np.ndarray,       # [B]
        policy_probs_j: Optional[np.ndarray] = None,   # [B, A]
        observed_returns: Optional[np.ndarray] = None,  # [B]
        behaviour_probs_obs: Optional[np.ndarray] = None,  # [B]
        mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tính effect theo mode, có/không DR. Toàn bộ đã vector hoá bằng
        einsum/gather từ trước — giữ nguyên, chỉ dọn lại comment.

        Returns dict:
            effect:        [E, B]              effect ở horizon cuối
            effect_per_h:  [E, B, n_horizons]  effect tách theo từng horizon
            dr_correction: [B]                 độ lớn số hạng hiệu chỉnh DR
        """
        mode = self.effect_mode if mode is None else str(mode)

        E, B, A, H = preds_all.shape

        idx_obs = torch.tensor(
            np.asarray(action_j_obs, dtype=np.int64).reshape(-1),
            dtype=torch.long,
            device=self.device,
        ).clamp(0, A - 1)  # [B]

        idx_exp = (
            idx_obs.view(1, B, 1, 1).expand(E, B, 1, H)
        )  # [E, B, 1, H]

        f_obs = torch.gather(preds_all, dim=2, index=idx_exp).squeeze(2)
        # f_obs: [E, B, H]

        dr_correction = torch.zeros(B, dtype=torch.float32, device=self.device)

        # ---------------------------------------------------------------
        if mode == "signed_aristocrat":
            if policy_probs_j is None:
                w = torch.full(
                    (B, A), 1.0 / float(A), dtype=torch.float32, device=self.device
                )
            else:
                w = torch.tensor(
                    np.asarray(policy_probs_j, dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )  # [B, A]

            baseline = torch.einsum("ebah,ba->ebh", preds_all, w)
            effect_per_h = f_obs - baseline  # [E, B, H]

        elif mode == "signed_oracle_matched":
            cand = torch.tensor(
                [a for a in self.candidate_actions if 0 <= a < A],
                dtype=torch.long,
                device=self.device,
            )  # [n_cand]

            if cand.numel() == 0:
                cand = torch.arange(A, dtype=torch.long, device=self.device)

            cand_preds = preds_all[:, :, cand, :]      # [E, B, n_cand, H]
            cand_mean = cand_preds.mean(dim=2)         # [E, B, H]

            effect_per_h = cand_mean - f_obs           # [E, B, H]

        elif mode == "range":
            max_f = preds_all.max(dim=2).values        # [E, B, H]
            min_f = preds_all.min(dim=2).values        # [E, B, H]
            effect_per_h = max_f - min_f               # [E, B, H]

        elif mode == "mean_abs":
            diff = torch.abs(preds_all - f_obs.unsqueeze(2))  # [E, B, A, H]

            mask = torch.ones(B, A, dtype=torch.float32, device=self.device)
            mask.scatter_(1, idx_obs.view(B, 1), 0.0)          # [B, A]

            denom = torch.clamp(mask.sum(dim=1), min=1.0)      # [B]

            effect_per_h = (
                torch.einsum("ebah,ba->ebh", diff, mask) / denom.view(1, B, 1)
            )  # [E, B, H]

        else:
            raise ValueError(f"mode không hợp lệ: {mode}")

        # ---------------------------------------------------------------
        # DOUBLY ROBUST CORRECTION — chỉ áp cho mode có dấu.
        # ---------------------------------------------------------------
        apply_dr = (
            self.use_doubly_robust
            and mode in ("signed_aristocrat", "signed_oracle_matched")
            and observed_returns is not None
            and behaviour_probs_obs is not None
        )

        if apply_dr:
            R_obs = torch.tensor(
                np.asarray(observed_returns, dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )  # [B]

            b_obs = torch.tensor(
                np.asarray(behaviour_probs_obs, dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            ).clamp(min=1.0 / self.iw_clip, max=1.0)  # [B]

            residual = R_obs.view(1, B) - f_obs[:, :, -1]  # [E, B]

            iw_minus_one = torch.clamp(
                1.0 / b_obs - 1.0, min=0.0, max=self.iw_clip
            )  # [B]

            correction = residual * iw_minus_one.view(1, B)  # [E, B]

            if mode == "signed_oracle_matched":
                correction = -correction

            effect_per_h = effect_per_h.clone()
            effect_per_h[:, :, -1] = effect_per_h[:, :, -1] + correction

            dr_correction = torch.mean(torch.abs(correction), dim=0)  # [B]

        return {
            "effect": effect_per_h[:, :, -1],   # [E, B]
            "effect_per_h": effect_per_h,       # [E, B, H]
            "dr_correction": dr_correction,     # [B]
        }

    # =====================================================================
    # API scoring
    # =====================================================================

    def score_batch(
        self,
        obs_i_batch,
        action_i_batch,
        observed_action_j_batch,
        z_core_excl_j_batch,
        m_periph_excl_j_batch,
        belief_summary_batch,
        policy_probs_j_batch=None,
        observed_returns_batch=None,
        behaviour_probs_obs_batch=None,
    ):
        """
        GIỮ NGUYÊN chữ ký v1 -> runner cũ gọi được ngay.

        Returns:
            mu_arr:    np.ndarray [B]  — CÓ DẤU nếu effect_mode là signed_*
            sigma_arr: np.ndarray [B]  — std across ensemble (bất định epistemic)
        """
        out = self.score_batch_full(
            obs_i_batch=obs_i_batch,
            action_i_batch=action_i_batch,
            observed_action_j_batch=observed_action_j_batch,
            z_core_excl_j_batch=z_core_excl_j_batch,
            m_periph_excl_j_batch=m_periph_excl_j_batch,
            belief_summary_batch=belief_summary_batch,
            policy_probs_j_batch=policy_probs_j_batch,
            observed_returns_batch=observed_returns_batch,
            behaviour_probs_obs_batch=behaviour_probs_obs_batch,
        )

        return out["mu"], out["sigma"]

    def score_batch_full(
        self,
        obs_i_batch,
        action_i_batch,
        observed_action_j_batch,
        z_core_excl_j_batch,
        m_periph_excl_j_batch,
        belief_summary_batch,
        policy_probs_j_batch=None,
        observed_returns_batch=None,
        behaviour_probs_obs_batch=None,
    ) -> Dict[str, np.ndarray]:
        """
        Phiên bản đầy đủ — cung cấp mọi thứ influence_signature.py cần.

        Returns dict of np.ndarray:
            mu            [B]    effect trung bình qua ensemble (CÓ DẤU)
            sigma         [B]    std qua ensemble = bất định epistemic
            mu_per_h      [B, H] effect theo từng horizon -> chiều LATENCY
            latency       [B]    trọng tâm horizon của |effect|, trong [0, H-1]
            mu_range      [B]    impact kiểu Pieroth (luôn >= 0) -> baseline
            dr_correction [B]    độ lớn hiệu chỉnh DR (chẩn đoán bias model)
        """
        B = len(obs_i_batch)

        if B == 0:
            z = np.zeros((0,), dtype=np.float32)
            return {
                "mu": z,
                "sigma": z,
                "mu_per_h": np.zeros((0, self.n_horizons), dtype=np.float32),
                "latency": z,
                "mu_range": z,
                "dr_correction": z,
            }

        obs = np.asarray(obs_i_batch, dtype=np.float32)
        z_arr = np.asarray(z_core_excl_j_batch, dtype=np.float32)
        m_arr = np.asarray(m_periph_excl_j_batch, dtype=np.float32)
        belief = np.asarray(belief_summary_batch, dtype=np.float32)
        a_i = np.asarray(action_i_batch, dtype=np.int64).reshape(-1)
        a_j = np.asarray(observed_action_j_batch, dtype=np.int64).reshape(-1)

        # MỘT forward pass duy nhất (cả action thay thế lẫn cả ensemble).
        preds_all = self._predict_all_actions(
            obs=obs, action_i=a_i, z=z_arr, m=m_arr, belief=belief
        )  # [E, B, A, H]

        res = self._compute_effects(
            preds_all=preds_all,
            action_j_obs=a_j,
            policy_probs_j=policy_probs_j_batch,
            observed_returns=observed_returns_batch,
            behaviour_probs_obs=behaviour_probs_obs_batch,
            mode=self.effect_mode,
        )

        effect = res["effect"]              # [E, B]
        effect_per_h = res["effect_per_h"]  # [E, B, H]

        mu = torch.mean(effect, dim=0)      # [B]

        if effect.shape[0] <= 1:
            sigma = torch.zeros_like(mu)
        else:
            sigma = torch.sqrt(
                torch.var(effect, dim=0, unbiased=True) + self.eps
            )  # [B]

        mu_per_h = torch.mean(effect_per_h, dim=0)  # [B, H]

        # ---- LATENCY: trọng tâm horizon của |effect| --------------------
        abs_h = torch.abs(mu_per_h)                              # [B, H]
        h_idx = torch.arange(
            self.n_horizons, dtype=torch.float32, device=self.device
        ).view(1, -1)                                            # [1, H]

        denom = torch.clamp(abs_h.sum(dim=1), min=self.eps)      # [B]
        latency = (abs_h * h_idx).sum(dim=1) / denom             # [B]

        # ---- mu_range: luôn tính, dùng làm baseline Pieroth -------------
        res_range = self._compute_effects(
            preds_all=preds_all,
            action_j_obs=a_j,
            mode="range",
        )
        mu_range = torch.mean(res_range["effect"], dim=0)        # [B]

        self.latest_dr_correction_magnitude = float(
            torch.mean(res["dr_correction"]).item()
        )

        # ---- MỘT lần .cpu().numpy() cho mỗi tensor output, ở biên API ----
        # (đây là nơi bắt buộc phải rời GPU: influence_signature.py và
        # belief_layer.py phía dưới vẫn nhận numpy, chưa được vector hoá
        # trong đợt này — xem giải thích cuối README_INTEGRATION.md).
        to_np = lambda t: t.detach().cpu().numpy().astype(np.float32)

        return {
            "mu": to_np(mu),
            "sigma": to_np(sigma),
            "mu_per_h": to_np(mu_per_h),
            "latency": to_np(latency),
            "mu_range": to_np(mu_range),
            "dr_correction": to_np(res["dr_correction"]),
        }

    def score_pair(
        self,
        obs_i,
        action_i,
        observed_action_j,
        z_core_excl_j,
        m_periph_excl_j,
        belief_summary,
        **kwargs,
    ):
        """Wrapper một cặp — giữ nguyên chữ ký v1."""
        mu_arr, sigma_arr = self.score_batch(
            obs_i_batch=[obs_i],
            action_i_batch=[int(action_i)],
            observed_action_j_batch=[int(observed_action_j)],
            z_core_excl_j_batch=[z_core_excl_j],
            m_periph_excl_j_batch=[m_periph_excl_j],
            belief_summary_batch=[belief_summary],
            **kwargs,
        )

        if len(mu_arr) == 0:
            return 0.0, 0.0

        return float(mu_arr[0]), float(sigma_arr[0])

    # =====================================================================
    # Chẩn đoán
    # =====================================================================

    def get_diagnostics(self) -> Dict[str, float]:
        """
        Số liệu chẩn đoán. Ba con số quan trọng nhất:

        ensemble_disagreement:
            Nếu ~ 0 -> ensemble đang GIẢ, sigma vô nghĩa (bệnh của v1).
            Kỳ vọng sau khi vá: > 0 và giảm dần khi học tốt lên.

        dr_correction_magnitude:
            Đo mức độ reward model bị lệch. Lớn -> model sai nhiều,
            DR đang gánh. Nhỏ -> model tốt.

        interventional_fraction:
            Tỷ lệ mẫu đến từ eps-forcing (can thiệp thật).
        """
        n = max(1, len(self.buffer))

        return {
            "buffer_size": int(len(self.buffer)),
            "n_interventional_samples": int(self.n_interventional_samples),
            "interventional_fraction": float(self.n_interventional_samples) / float(n),
            "latest_loss": float(self.latest_loss),
            "latest_train_residual": float(self.latest_train_residual),
            "latest_holdout_residual": float(self.latest_holdout_residual),
            "ensemble_disagreement": float(self.latest_ensemble_disagreement),
            "dr_correction_magnitude": float(self.latest_dr_correction_magnitude),
            "effect_mode": self.effect_mode,
            "use_doubly_robust": bool(self.use_doubly_robust),
            "n_ensemble": int(self.n_ensemble),
            "n_horizons": int(self.n_horizons),
            "use_vmap_ensemble": bool(self.use_vmap_ensemble),
        }
