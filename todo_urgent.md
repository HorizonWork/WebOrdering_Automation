# TODO_URGENT – WOA Agent

Danh sách các việc cần làm / nâng cấp quan trọng để đưa agent về đúng kiến trúc “Planner + Controller + Observer + Executor”, chạy ổn định trên 2 sàn TMĐT VN và an toàn hơn với người dùng.

---

## P0 – Ổn định code hiện tại (blocking issues)

- [ ] **Sửa lỗi cú pháp / encoding trong code Python**
  - Rà toàn bộ `src/orchestrator/agent_orchestrator.py`, `src/planning/react_engine.py`, `src/execution/browser_manager.py`, `src/orchestrator/safety_guardrails.py`, v.v. để:
    - Loại bỏ các ký tự lạ trong string log (`"�o"`, `"dY"`, …) nếu làm hỏng cú pháp.
    - Sửa các lệnh log sai cú pháp kiểu `logger.info(f"dY", Step {step}/{self.max_steps}")` → dùng 1 string format hợp lệ.
  - Chạy lại import + unit tests cơ bản (`tests/unit`) để đảm bảo code chạy được.

- [ ] **Đồng bộ `run_agent.py` với API mới của `AgentOrchestrator`**
  - Hiện `AgentOrchestrator.__init__` nhận `phobert_checkpoint`/`vit5_checkpoint` chứ không nhận `phobert_encoder`/`vit5_planner` như trong `run_agent.py`.
  - Cập nhật `run_agent.py` cho đúng (hoặc dùng luôn `tests/full_pipeline_test.py` làm entrypoint chính).

- [ ] **Kích hoạt lại các test quan trọng**
  - Chạy và sửa các lỗi cơ bản (không cần hoàn hảo) cho:
    - `tests/unit/test_perception.py`
    - `tests/unit/test_execution.py`
    - `tests/unit/test_planning.py`
  - Mục tiêu: đảm bảo pipeline hiện tại ít nhất chạy được end-to-end với `example.com` và không crash.

---

## P1 – Khớp kiến trúc Planner + Controller + DSL action

- [ ] **Thiết kế action DSL chuẩn (NAVIGATE / CLICK / FILL / PRESS / SELECT_DROPDOWN / WAIT_FOR / CLICK_AT)**
  - Định nghĩa schema JSON cho 1 action:
    - `{ "id": "INPUT_MAX_PRICE", "type": "FILL", "description": "...", "dom_selector": "...", "vision_ref": "..." }`
  - Xây `Executor` (hoặc map trong `SkillExecutor`) từ `action_id` → Playwright:
    - `NAVIGATE(url)` → skill `goto`
    - `CLICK(action_id)` → skill `click` với selector map sẵn
    - `FILL(action_id, text)` → skill `fill`/`type`
    - `WAIT_FOR(selector_id|event)` → skill `wait_for_selector`/`wait_for_navigation`
    - Fallback `CLICK_AT(bbox)` (dùng khi không có selector, optional).

- [ ] **Observer → `page_state` + `available_actions`**
  - Viết module Observer riêng (có thể dựa trên `SceneRepresentation`, `UIDetector`, `DOMDistiller`, `VisionEnhancer`) để build:
    - `page_type` (rule: URL pattern + presence selector per-sàn).
    - `dom_state` có `products`, `filters`, `cart`, `checkout`.
    - `vision_state` (nếu bật vision).
    - `available_actions`: danh sách action DSL như spec.
  - Ưu tiên implement cho 1 sàn (Shopee) trước, sau đó nhân bản rule cho Lazada.

- [ ] **Planner mới (LLM-1, chỉ sinh bước kế hoạch cấp cao)**
  - Định nghĩa interface Planner:
    - Input: `{ goal, high_level_state, history_summary }`
    - Output: `{ "next_plan_step": {...} }` với các `step_id` hợp lệ: `SEARCH_PRODUCT`, `APPLY_FILTER`, `SELECT_PRODUCT`, `GO_TO_CART`, `GO_TO_CHECKOUT`, `FILL_CHECKOUT_INFO`, `REVIEW_ORDER`, `TERMINATE`.
  - Viết prompt template ở dạng system + few-shot, trả JSON thuần.
  - Giai đoạn đầu: có thể dùng chính ViT5 hoặc model nhỏ hơn / state machine rule-based cho 2 sàn.

- [ ] **Controller mới (LLM-2, chọn action cụ thể)**
  - Định nghĩa interface Controller:
    - Input: `{ goal_summary, current_plan_step, page_state, available_actions_flat, last_action_result, short_history }`
    - Output: `{"chosen_action": {"action_id": "...", "type": "...", "text": "...?"}}` hoặc `CLICK_AT(bbox)` fallback.
  - Xây prompt system cho Controller:
    - Chỉ được chọn 1 action từ `available_actions`.
    - Không được click nút đặt hàng cuối cùng.

- [ ] **Tích hợp Planner + Controller vào main loop**
  - Cập nhật `AgentOrchestrator.execute_task`:
    - Thay vì cho `ReActEngine` sinh trực tiếp skill, dùng flow:
      - Observer → `page_state`
      - Planner → `next_plan_step`
      - Controller → `chosen_action`
      - Executor → thực thi action, rồi quay lại Observer.
  - Giai đoạn chuyển tiếp: có thể giữ ReAct cũ cho các task generic, dùng Planner/Controller mới cho task shopping.

---

## P2 – Guardrail & UI tương tác với người dùng

- [ ] **Xác nhận với người dùng ở các bước quan trọng (UI/UX)**
  - Thiết kế một lớp `UserInteraction` (hoặc callback) để:
    - Hỏi lại user trước khi:
      - Đăng nhập / đăng ký tài khoản.
      - Thêm phương thức thanh toán mới.
      - Thực hiện bước cuối “Đặt hàng” / “Place Order” / “Pay now”.
    - Khi cần nhập thông tin nhạy cảm (địa chỉ nhận hàng, số điện thoại, email):
      - Agent đề xuất giá trị hoặc parse từ history, nhưng vẫn **hỏi lại user** để confirm / chỉnh sửa.
  - Ở phiên bản CLI:
    - Dùng prompt kiểu: `Agent sẽ đăng nhập bằng email X – bạn có đồng ý? [y/N]`
    - Cho phép user nhập / sửa địa chỉ, tên người nhận, số điện thoại ngay trong terminal.
  - Chuẩn bị interface để sau này có thể gắn vào UI web (tối thiểu là tách logic hỏi/confirm khỏi core).

- [ ] **Nâng cấp `SafetyGuardrails` đúng rule “không tự đặt hàng”**
  - Thêm rule nhận diện nút cuối: text chứa các pattern:
    - `Đặt hàng`, `Thanh toán`, `Place Order`, `Pay now`, `Thanh toán ngay`, `Pay`, `Checkout now`, v.v.
  - Kết hợp với `page_type == 'checkout'` hoặc `REVIEW_ORDER`:
    - Nếu Controller chọn action click vào nút này:
      - Không thực thi ngay; thay bằng trạng thái “REVIEW_ORDER” + gửi yêu cầu confirm sang `UserInteraction`.
      - Chỉ thực thi khi user xác nhận rõ ràng.

- [ ] **Siết domain whitelist cho TMĐT**
  - Từ `SafetyGuardrails.get_allowed_domains()`:
    - Chỉ whitelist bắt buộc: `shopee.vn`, `lazada.vn` (và domain phụ nếu cần).
    - Các domain khác (`google.com`, `facebook.com`, `example.com`) chỉ dùng cho test/dev, có flag cấu hình riêng.
  - Trong orchestrator:
    - Nếu `start_url` không thuộc whitelist khi chạy ở chế độ “shopping”, từ chối và báo lại user.

- [ ] **Policy retry / timeout thân thiện**
  - Khi selector không thấy hoặc trang load chậm:
    - Hiển thị thông tin ngắn gọn cho user (ở CLI).
    - Hỏi user có muốn thử lại / bỏ qua bước / dừng agent.

---

## P3 – Logging, episode dataset & training

- [ ] **Chuẩn hóa episode log**
  - Thiết kế cấu trúc episode như spec:
    ```json
    {
      "episode_id": "ep_xxxx",
      "goal": "...",
      "steps": [
        {
          "t": 0,
          "page_state": { ... },
          "planner_input": { ... }, "planner_output": { ... },
          "controller_input": { ... }, "controller_output": { ... },
          "env_feedback": {"success": true, "new_url": "..."}
        }
      ],
      "final_status": "SUCCESS|FAIL|TIMEOUT"
    }
    ```
  - Gắn pipeline log này vào `AgentOrchestrator` (mode debug/collect).

- [ ] **Hoàn thiện `scripts/collect_trajectories.py`**
  - Viết script chạy agent trên một tập task demo (Shopee/Lazada), lưu episode log theo format trên vào `data/trajectories/`.
  - Phân tách ngay từ đầu:
    - Mẫu `Planner`: `(goal + high_level_state + history_summary) → next_plan_step`.
    - Mẫu `Controller`: `(goal + plan_step + page_state + available_actions) → chosen_action_id`.

- [ ] **Chuẩn bị pipeline fine-tune (Controller trước, Planner sau)**
  - Bước 1 (no-train): chạy bằng prompt + few-shot, action space nhỏ.
  - Bước 2: fine-tune Controller (CE trên `chosen_action_id`) từ human demo + rule-based + optional teacher.
  - Giai đoạn sau: cân nhắc fine-tune Planner hoặc chuyển Planner thành state machine cho từng sàn.

---

## P4 – Model & offline inference

- [ ] **Tùy chọn backend LLM local**
  - Thiết kế abstraction cho LLM backend (thay vì gắn chặt vào ViT5):
    - Cho phép dùng: Qwen2 7B Instruct, Mistral 7B Instruct, LLaMA3 8B Instruct… thông qua:
      - Ollama, llama.cpp, vLLM (tùy môi trường).
  - Cấu hình qua `config/models.yaml` + `.env`:
    - Mode “HF online” vs “local backend”.

- [ ] **Lite mode / Controller nhỏ**
  - Thiết kế để Controller có thể dùng:
    - Model 1–3B local (hoặc distilled) chỉ làm multiple-choice trên `available_actions`.
  - Planner có thể chuyển sang state machine đơn giản per-sàn (Shopee / Lazada) cho flow mua hàng tiêu chuẩn.

---

## P5 – Test, docs & UI nâng cao

- [ ] **Viết integration test thật cho flow mua hàng**
  - `tests/integration/test_shopee_workflow.py`: flow chọn sản phẩm, apply filter, thêm vào giỏ, đi tới trang checkout (dừng trước nút “Đặt hàng”).
  - `tests/integration/test_agent_flow.py`: test generic AgentOrchestrator với Planner/Controller mới.

- [ ] **Cập nhật docs kiến trúc**
  - Viết lại `docs/architecture.md` cho khớp với:
    - Planner (LLM-1) / Controller (LLM-2).
    - `page_state`, DSL, Observer, Executor.
    - Guardrail & UserInteraction (confirm các bước quan trọng).

- [ ] **Nâng cấp UI/UX (tùy chọn, sau khi core ổn định)**
  - Thiết kế giao diện CLI “đẹp hơn”:
    - Hiển thị plan step hiện tại, action tiếp theo, hỏi `[y/N]` cho bước quan trọng.
    - Cho phép user nhập nhanh địa chỉ / thông tin giao hàng trong cùng flow.
  - Về lâu dài: cân nhắc web dashboard nhỏ (FastAPI + frontend đơn giản) để:
    - Hiển thị screenshot + highlight element sắp click.
    - Cho phép user approve / sửa actions bằng chuột thay vì chỉ text.

---

**Gợi ý thứ tự thực hiện**  
1) P0 → chạy được code, test unit cơ bản.  
2) P1 + P2 → kiến trúc Planner/Controller + guardrail/confirm user.  
3) P3 → logging + dataset.  
4) P4/P5 → tối ưu model, UI/UX, test & docs. 

