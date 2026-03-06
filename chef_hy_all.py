from abc import ABC, abstractmethod
from PIL import Image
import torch

# ── Abstract Base ──────────────────────────────────────────────────────────────

class BaseVLM(ABC):
    @abstractmethod
    def load_model(self, model_id: str): ...

    @abstractmethod
    def predict(self, image: Image.Image, prompt: str) -> str: ...


# ── Qwen2.5-VL Implementation ──────────────────────────────────────────────────

class Qwen25VL(BaseVLM):
    def __init__(self, model_id: str, device: str = "auto", token: str = None):
        self.model_id = model_id
        self.device = device
        self.token = token
        self.model = None
        self.processor = None
        self.load_model(model_id)

    def load_model(self, model_id: str):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=self.token
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
            token=self.token
        )

        self.model.eval()
        print(f"[Qwen2.5-VL] Loaded '{model_id}'")

    def predict(self, image: Image.Image, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


# ══════════════════════════════════════════════════════════════════════════════
# CHAIN-OF-THOUGHT PROMPT — STRICT SOP ENFORCEMENT
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_CHAIN_OF_THOUGHT = """
You are a senior kitchen hygiene compliance officer.
Inspect this image of kitchen staff carefully.

Check this in order:
Step 1 — Are the torso/body visible in the image? Yes or No
Step 2 — Apron: Is an apron worn? Yes or No
Step 3 — Is the head/hair visible in the image? Yes or No
Step 4 — Hairnet: Is a hairnet or head covering worn? Yes or No
Step 5 — Are the hands/wrists visible in the image? Yes or No
Step 6 — Gloves: Are gloves worn on the hands? Yes or No

Reply in exactly this format:
Torso visible: Yes/No
Apron: Yes/No
Head visible: Yes/No
Hairnet: Yes/No
Hands visible: Yes/No
Gloves: Yes/No
Verdict: APPROVED or REJECTED

Decision logic - Report the FIRST failure in order:
- If torso is NOT visible → Verdict: REJECTED | Torso not visible
- Else if apron is NO → Verdict: REJECTED | Apron not detected
- Else if head is NOT visible → Verdict: REJECTED | Head not visible
- Else if hairnet is NO → Verdict: REJECTED | Hairnet not detected
- Else if hands are NOT visible → Verdict: REJECTED | Hands not visible
- Else if gloves is NO → Verdict: REJECTED | Gloves not detected
- Else (all YES) → Verdict: APPROVED | Apron, hairnet and gloves worn correctly

IMPORTANT: Report only the FIRST failure you find when checking top to bottom. Do not report later failures.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Chef Hygiene Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class ChefHygienePipeline:
    def __init__(self, vlm: BaseVLM):
        self.vlm = vlm

    def run(self, image: Image.Image):
        self._run_chain_of_thought(image)

    def _run_chain_of_thought(self, image: Image.Image):
        print(f"\n{'=' * 80}")
        print("👨‍🍳  CHEF HYGIENE REPORT  |  Strategy: CHAIN-OF-THOUGHT (STRICT)")
        print(f"{'=' * 80}\n")

        response = self.vlm.predict(image, PROMPT_CHAIN_OF_THOUGHT)
        print(response)
        print(f"\n{'=' * 80}")


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    print("=" * 80)
    print("👨‍🍳  MEALAWE — CHEF HYGIENE ASSESSMENT PIPELINE")
    print("=" * 80)
    print(f"Model: {MODEL_ID}")
    print("Loading model...")

    try:
        vlm = Qwen25VL(MODEL_ID)
    except Exception as e:
        print(f"\n❌ Model loading failed: {str(e)[:300]}")
        print("Try: pip3 install --user qwen-vl-utils")
        sys.exit(1)

    print("✅ Model loaded!\n")

    image_path = input("📸 Enter chef image path: ").strip()
    if not image_path:
        print("No path provided.")
        sys.exit(1)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Could not open image: {e}")
        sys.exit(1)

    pipeline = ChefHygienePipeline(vlm)
    pipeline.run(image)
