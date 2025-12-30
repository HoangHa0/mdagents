# Tài liệu sửa lỗi logic: MDAgents Implementation

**Tham chiếu:** MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making (NeurIPS 2024)

**File phân tích:** `utils_paper.py`

---

## Mức độ 1: Lỗi nghiêm trọng (Critical)

---

### 1.3 Hàm `determine_difficulty` có thể trả về `None`

**Vị trí:** Lines 344-349

**Mô tả lỗi:** Nếu response từ LLM không chứa keywords "basic", "intermediate", "advanced" hoặc "1)", "2)", "3)", hàm không có return statement cuối và trả về `None` implicitly.

```python
if 'basic' in response.lower() or '1)' in response.lower():
    return 'basic', None
elif 'intermediate' in response.lower() or '2)' in response.lower():
    return 'intermediate', moderator
elif 'advanced' in response.lower() or '3)' in response.lower():
    return 'advanced', None
# THIẾU: else clause hoặc default return
```

**Xung đột với paper:** Algorithm 1 (Line 1) yêu cầu moderator phải trả về một complexity level. Trả về `None` sẽ gây crash ở `main.py`.

**Cách sửa:**
```python
# Thêm default fallback cuối hàm
return 'basic', None  # Default to basic if unclear
```

## Mức độ 2: Lỗi logic quan trọng (High Priority)

### 2.2 Intermediate không sử dụng Hierarchy đúng cách

**Vị trí:** Lines 440-444, Lines 618-666

**Mô tả lỗi:** Code parse hierarchy từ LLM output nhưng không thực sự sử dụng nó để điều khiển communication flow. Tất cả agents vẫn có thể message nhau tự do.

**Xung đột với paper:** Figure 2 và Section 3.4 mô tả:
- "Pathologist -> Radiologist -> GP Moderator" hierarchy
- Agents communicate based on their hierarchical relationship
- "for every round, consensus within the MDT is determined by parsing and comparing their opinions"

**Thực tế trong code:** `parse_hierarchy()` chỉ tạo tree structure để hiển thị, không ảnh hưởng đến logic thảo luận.

---

### 2.3 External communication trong Group chưa implement

**Vị trí:** Lines 204-207

**Mô tả lỗi:** `comm_type='external'` chỉ return `None`, không có logic thực sự.

```python
elif comm_type == 'external':
    return None  # Not implemented
```

**Xung đột với paper:** Section 3.4 (High complexity) mô tả ICT có inter-team communication:
- "Beginning with the Initial Assessment Team, moving through various diagnostic teams"
- Teams build upon foundation laid by previous teams

**Hiện tại:** FRDT nhận reports qua tham số `message`, nhưng external communication pattern chưa được implement đầy đủ.

---

## Mức độ 3: Lỗi trung bình (Medium Priority)

### 3.1 Không lưu Conversation History cho Decision Maker

**Vị trí:** Lines 767-786 (Intermediate), Lines 946-965 (Advanced)

**Mô tả lỗi:** Final decision maker không nhận được full conversation history như paper yêu cầu.

**Xung đột với paper:** Section 3.5 mô tả:
- "Moderate: Incorporates the **conversation history (Interaction)** between the recruited agents"
- Decision maker cần hiểu nuances và disagreements

**Code hiện tại:**
```python
# Line 775: Chỉ truyền final_answers, không có interaction history
answers_text = "".join(f"[{role}] {ans}\n" for role, ans in final_answers.items())
```

**Cách sửa:** Truyền thêm `interaction_log` vào prompt của decision maker.

---

### 3.2 Thứ tự Few-shot khác với CoT-SC trong paper

**Vị trí:** Lines 69-72

**Mô tả lỗi:** Few-shot format là `reason + answer`, nhưng paper sử dụng CoT-SC (Chain-of-Thought with Self-Consistency).

**Code:**
```python
self.messages.append({"role": "assistant", "content": "Let's think step by step. " + exampler['reason'] + " " + exampler['answer']})
```

**Xung đột với paper:** Section 4.1 và baseline methods:
- "Few-shot CoT-SC [82] builds upon Few-shot CoT by **sampling multiple chains** to yield the majority answer"
- Code hiện tại không implement self-consistency (multiple sampling)

---

### 3.3 Advanced query không có inter-round refinement

**Vị trí:** Lines 897-944

**Mô tả lỗi:** Mỗi team (IAT, MDTs, FRDT) chỉ chạy một lần `interact()`, không có refinement rounds như intermediate.

**Xung đột với paper:** Algorithm 1 Lines 17-24:
- "each team, led by a lead clinician, collaboratively produces a comprehensive report synthesizing their findings"
- "This phased approach... ensures a meticulous and refined examination"

**Hiện tại:** Không có multiple rounds hoặc feedback loop trong ICT internal discussions.

---

## Mức độ 4: Cải thiện và tối ưu (Optimization)

### 4.1 Không tracking API calls

**Mô tả:** Paper phân tích efficiency thông qua số API calls (Section 4.4), nhưng code không count/log số calls.

**Khuyến nghị:** Thêm counter trong `Agent.chat()` và `Agent.temp_responses()`.

---

### 4.3 Không lưu Interaction Logs ra file

**Mô tả:** `interaction_log` và `feedback_log` chỉ được print, không được persist để phân tích sau.

**Khuyến nghị:** Lưu logs vào JSON file cùng với results.

---

### 4.4 Thiếu temperature variation

**Vị trí:** Line 105

**Mô tả:** `temp_responses()` mặc định chỉ dùng `temperatures=[0.0]`.

**Xung đột với paper:** Section 4.5 thử nghiệm với T=0.3 và T=1.2
- "Our Adaptive approach shows resilience to changes in temperature"

**Khuyến nghị:** Cho phép truyền temperatures từ arguments.

---

## Tóm tắt ưu tiên sửa lỗi

| STT | Lỗi | Line | Mức độ |
|-----|-----|------|--------|
| 1 | determine_difficulty trả về None | 344-349 | Critical |
| 2 | Hierarchy không được sử dụng | 440-444 | High |
| 3 | External communication chưa implement | 204-207 | High |
| 4 | Thiếu conversation history cho decision maker | 767-786 | Medium |
| 8 | Không implement CoT-SC | 69-72 | Medium |
| 9 | ICT thiếu refinement rounds | 897-944 | Medium |
| 10 | Không tracking API calls | - | Low |
| 11 | Không tính entropy | 686-712 | Low |

---

Code `utils_paper.py` đã implement cấu trúc chính nhưng còn thiếu một số chi tiết như hierarchy enforcement và complete conversation history tracki ng.
