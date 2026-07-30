# HANDOFF NOTE — Tìm novelty cho paper CIG-AMF (MARL)
*Mục đích file: nếu session/credit hết, model khác đọc file này là tiếp tục được nguyên mạch suy nghĩ. Người dùng (S) là NEWBIE về MARL — mọi giải thích phải ở mức người mới, định nghĩa mọi thuật ngữ tiếng Anh/viết tắt khi dùng. Ngôn ngữ trao đổi: tiếng Việt.*

## 1. Bối cảnh
- S có draft paper (file `CYB_490__3_.pdf`): **CIG-AMF — Causal Influence Graph with Adaptive Mean Field**, một framework MARL xử lý non-stationarity bằng cách tách 2 tầng (structural: AI ảnh hưởng ai / behavioural: họ đang hành xử thế nào), dùng counterfactual proxy ŵ_ij (Eq.7: hiệu 2 lần dự đoán của reward-model khi đổi action của neighbour j) + Bayes-light belief + hysteresis để chia neighbour thành CORE (model kỹ bằng pair latent z_ij) và PERIPHERAL (nén bằng multi-memory slots, Eq.20-25). Two-timescale + EWMA/CUSUM trigger.
- Draft còn chứa nguyên các chỉ dẫn của giáo sư nhúng trong text (kiểu "make it shorter", "Discuss the shortcomings..."). Results chỉ có Exp1-2 (behavioural drift / structural shift); **Exp3 (tiny-oracle calibration), Exp4 (peripheral gap), Exp5 (scalability) CHƯA có số liệu**.
- S cần: MỘT điểm novelty để tiếp tục research.

## 2. Ba lỗ hổng đã chẩn đoán từ số liệu trong draft (turn 1)
1. **Null result của module chính**: NoMultiMemory (bỏ multi-memory) Core F1 = 0.251/0.259 ≈ hoặc > Final CIG-AMF (0.244/0.262), reward ngang, nhanh gấp ~3 (241.8 vs 74.7 agent-steps/s). → module multi-memory chưa chứng minh được giá trị. Chẩn đoán kỹ thuật: **slot collapse** — slot gán bằng softmax trên feature học tự do, không có tín hiệu huấn luyện riêng ép slot chuyên môn hoá → sụp về ~single mean (đồng phục hoá trọng số hoặc 1 slot độc quyền).
2. **Chữ "Causal" trong tên chưa được kiểm chứng**: Exp3 (so proxy với clone-state intervention thật) trống; Eq.7 là plug-in estimator, action của j không randomized → ŵ_ij có thể chỉ là correlation; bài phải tự thú "không claim ŵ=w".
3. **Throughput thảm**: chậm ~45x so với Pure Mean Field (74.7 vs 3405.8); core size chỉ nở tới trần cứng 6 rồi đứng — "allocate capacity" chưa thật sự adaptive; Exp5 trống.

## 3. Ba hướng novelty đã đề xuất (turn 1)
- **Flagship**: Debias ŵ_ij bằng doubly-robust (outcome model + propensity model của action j, Neyman-orthogonal) + dùng SIGN của influence (helpful/harmful, mượn insight MAGIC "mạnh ≠ có ích") cho core selection; mượn nền identifiability từ paper Agent-Specific Effects (noise monotonicity). Biến Exp3 thành headline. Mạnh về lý thuyết nhưng NẶNG cho newbie.
- **Option B**: Core size = learned budget có ràng buộc compute (Lagrangian/meta-controller, marginal value per compute); vẽ reward–compute Pareto frontier; cứu Exp5 + throughput.
- **Option C (KHUYẾN NGHỊ cho S vì khả thi nhất với newbie)**: **Role-structured peripheral memory từ counterfactual influence signatures** — chi tiết mục 5.

## 4. Turn 2: S upload sách MARL (Albrecht/Christianos/Schäfer, MIT Press 2024, 395tr, path `/mnt/user-data/uploads/marl-book.pdf`), yêu cầu:
(a) tóm "nhân loại đã làm gì" từ sách; (b) giải thích MỌI thuật ngữ EN + viết tắt đã dùng (cf-ASE, noise monotonicity, doubly robust, ...) cho newbie; (c) giải thích câu "slot softmax sụp về mean"; (d) validate Option C cụ thể, ai làm chưa. → ĐÃ TRẢ LỜI ĐẦY ĐỦ trong turn này (glossary ~30 thuật ngữ; đừng lặp lại, chỉ bổ sung nếu S hỏi thêm).

### Tóm sách đã trích (các mốc để tái sử dụng):
- Ch2-4: RL cơ bản; game models (normal-form → stochastic → POSG; env của S là POSG); solution concepts (Nash, Pareto, correlated eq; PPAD-complete).
- **Mục 5.4 — 4 thách thức**: non-stationarity (định nghĩa stationary process; cyclic co-adaptation), equilibrium selection, credit assignment, scaling (|A| exponential). Sách coi non-stationarity là 1 khối; việc S tách structural/behavioural là góc mới.
- Ch6: value iteration cho game, Minimax-Q/Nash-Q/Correlated-Q, **fictitious play** (tổ tiên opponent modelling), WoLF, no-regret.
- Ch9: CTDE (paradigm thống trị); **mục 9.4.4 = họ hàng gần nhất của Eq.7**: difference rewards d_i = R(s,⟨a_i,a₋ᵢ⟩) − R(s,⟨ã_i,a₋ᵢ⟩) (Wolpert&Tumer), aristocrat utility, **COMA** counterfactual baseline — khác biệt: COMA can thiệp action CỦA MÌNH (credit assignment), S can thiệp action CỦA NEIGHBOUR (influence). S PHẢI cite COMA + difference rewards. VDN/QMIX (value decomposition); 9.6 agent modelling (global repr — S làm pair-specific, điểm khác thật); parameter sharing; AlphaZero/PSRO/AlphaStar.
- Sách KHÔNG có: causal inference cho inter-agent influence, capacity allocation, structural/behavioural split, role discovery → đất của S nằm ngoài giáo trình chuẩn.

## 5. Option C chi tiết (đã validate turn 2)
**Ý tưởng**: giữ hồ sơ ŵ_ij theo thời gian/ngữ cảnh thành **influence signature** vector (dấu; độ lớn; variance; tính điều-kiện-theo-ngữ-cảnh; độ trễ trong horizon H). Cluster signatures (k-means ở slow timescale trên μ̄_ij đã smooth, hoặc auxiliary loss cho slot head) → slot = role chức năng ego-centric: blocker (âm mạnh, conditional, tức thì), relay/signaller (dương, trễ 3-5 bước), consumer (âm nhẹ đều), inert (~0). Thay Eq.21. Peripheral summary giữ heterogeneity mà mean vứt (mean chỉ giữ trung bình, mất dấu-theo-ngữ-cảnh/độ trễ/variance).
**Prior art đã tra** (không ai làm đúng ô này):
- Role-based MARL: ROMA (2020, role embedding từ observation/history, mutual information), RODE (2021)/SIRD (2023) (role = subset action space), LDSA/ALMA (sub-task assignment), ACORM, CORD, R3DM. TẤT CẢ: role suy từ dữ liệu QUAN SÁT (correlational) + dùng để điều kiện hoá POLICY CỦA CHÍNH AGENT, role TOÀN CỤC.
- Causal-influence MARL: Jaques 2019 (social influence), SCIC (Du 2024, single-step interventional influence, CTDE), MAGIC (2026, multi-step + advantage gate, insight "influence mạnh ≠ có ích") — TẤT CẢ dùng influence làm INTRINSIC REWARD.
- ASE/cf-ASE (Triantafyllou et al. 2023-24, arXiv 2310.11334): lý thuyết identifiability cho agent-to-agent effect, cần noise monotonicity, không cần bijectiveness; offline, MMDP nhỏ, mục đích accountability (sepsis testbed) — S nên cite làm nền lý thuyết.
**Ô trống của S** = (chữ ký từ counterfactual/interventional, không phải hành vi) × (dùng để cấu trúc hoá BIỂU DIỄN VỀ NGƯỜI KHÁC trong peripheral memory, không phải policy mình) × (ego-centric: j là blocker với i nhưng relay với i′).
**Rủi ro + thuốc giải**: ŵ nhiễu → cluster trên belief-smoothed μ̄_ij; env phải có role thật (S tự thiết kế env nên kiểm soát được); PHẢI thêm baseline "cluster trên observed behaviour" (ROMA-lite) để chứng minh causal signature > correlational signature — thí nghiệm then chốt.
**Thí nghiệm**: chạy Exp4 sẵn thiết kế (signature-slots vs softmax-slots vs single mean vs behaviour-cluster); hình heatmap cluster centroids (interpretability); kỳ vọng lật null result → positive. Fallback: kể cả reward không tăng, role-recovery khớp ground-truth vẫn là kết quả interpretability đăng được.
**Novelty statement gợi ý**: "First peripheral memory mechanism where slots are organized by ego-centric functional roles discovered from counterfactual influence signatures — distinct from role-based MARL (roles from observed behaviour, for the agent's own policy) and from causal-influence MARL (influence as intrinsic reward)."

## 6. Việc bắt buộc bất kể chọn hướng nào
Related work đang THIẾU: COMA + difference rewards (có trong sách 9.4.4), Jaques 2019, SCIC (Du 2024), MAGIC (2026, arXiv 2605.01805), ASE (arXiv 2310.11334), ROMA/RODE. Câu phòng thủ chuẩn: "họ dùng influence làm reward/để chuyên môn hoá policy; chúng tôi dùng làm tiêu chí phân bổ + tổ chức dung lượng mô hình hoá dưới bất định."

## 7. Trạng thái hội thoại & việc tiếp theo khả dĩ
- Đã hỏi S (turn 1, chưa trả lời trực tiếp): chọn flagship / B / C; compute budget (N≤96, 5 seeds?).
- Turn 2 S nghiêng về tìm hiểu C. Việc tiếp theo hợp lý: (i) chốt hướng; (ii) nếu C: thiết kế cụ thể signature vector + pseudo-code clustering + sửa Eq.20-25 + viết lại RQ3/H3 + kế hoạch Exp4; (iii) sửa related work; (iv) làm sạch các chỉ dẫn giáo sư còn nhúng trong draft.
- Bối cảnh cá nhân S: newbie MARL, nền Java/DSA cơ bản, nói tiếng Việt, đang làm song song project CAROECT-D (event camera). Giải thích phải cụ thể, có ví dụ giao thông, tránh trừu tượng.
