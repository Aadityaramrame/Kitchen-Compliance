# pipeline.py
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


# ── Food Quality Prompts ───────────────────────────────────────────────────────

FOOD_PROMPTS = {
    "cooking_quality": """
Inspect this food image.
Only flag if you see a problem: burnt, undercooked, or poor quality.
If food looks fine, reply: PASS
If not, reply: FAIL | (reason in 5 words max, e.g. "food appears burnt")
""",

    "veg_nonveg": """
Does this food contain meat, fish, or eggs?
Reply: Veg or Non-Veg only. No explanation.
""",

    "jain_check": """
Does this food visibly contain onion, garlic, or root vegetables?
If not visible: PASS
If yes: FAIL | contains non-jain ingredients
""",

    "oil_colour_check": """
Does this food look excessively oily or have unnatural colour?
If fine: PASS
If not: FAIL | (e.g. "food looks oily" or "colour looks off")
""",

    "salad_check": """
Check if this salad has all 5: onion, cucumber, tomato, carrot, lemon.
If all present: PASS
If missing anything: FAIL | missing (list what's missing)
""",

    "garnishing_check": """
Is any garnishing visible on this dish?
If yes: PASS
If no: FAIL | no garnishing
""",

    "sop_compliance": """
Inspect this food image against these rules:
1. Not burnt or undercooked
2. Not excessively oily
3. Natural colour
4. Garnishing present
5. Clean plating

If all good: PASS
If any fail: FAIL | (only list what's wrong, max 10 words, e.g. "food is oily, no garnishing")
"""
}

# ── Chef Hygiene Prompts ───────────────────────────────────────────────────────

CHEF_PROMPTS = {
    "apron_check": """
Is the person in this image wearing an apron?
If yes: PASS
If no: FAIL | no apron
""",

    "hairnet_check": """
Is the person in this image wearing a hairnet or hair covering?
If yes: PASS
If no: FAIL | no hairnet
""",

    "gloves_check": """
Is the person in this image wearing gloves?
If yes: PASS
If no: FAIL | no gloves
""",

    "jewellery_check": """
Is the person wearing any jewellery on hands or wrists (rings, bangles, bracelets, watch)?
If none visible: PASS
If yes: FAIL | jewellery detected on hands
""",

    "cleanliness_check": """
Does the person look clean, neat and professionally presented for a kitchen environment?
Check: clean uniform, no visible dirt, tidy appearance.
If clean and neat: PASS
If not: FAIL | (reason in 5 words, e.g. "uniform looks dirty/shabby")
""",

    "chef_sop_compliance": """
Check this chef/kitchen staff image against hygiene SOPs:
1. Apron worn
2. Hairnet or hair covering worn
3. Gloves worn
4. No jewellery on hands or wrists
5. Clean and neat appearance

If all good: PASS | Great job, keep it up!
If any fail: FAIL | (list only what's missing or wrong, e.g. "no hairnet, jewellery on wrist")
"""
}


# ── Food Quality Pipeline ──────────────────────────────────────────────────────

class FoodQualityPipeline:
    def __init__(self, vlm: BaseVLM):
        self.vlm = vlm
        self.contradictions = []

    def evaluate(self, image_path: str, checks: list = None) -> dict:
        image = Image.open(image_path).convert("RGB")
        checks = checks or list(FOOD_PROMPTS.keys())
        results = {}
        for check in checks:
            prompt = FOOD_PROMPTS[check]
            results[check] = self.vlm.predict(image, prompt)
        return results

    def parse_result(self, check_name: str, response: str) -> dict:
        lines = response.strip().split('\n')
        parsed = {'check': check_name, 'raw_response': response}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip().lower()] = value.strip()
        return parsed

    def validate_consistency(self, parsed_results: list) -> tuple:
        contradictions = []
        oil_check = next((p for p in parsed_results if p['check'] == 'oil_colour_check'), None)
        sop_check = next((p for p in parsed_results if p['check'] == 'sop_compliance'), None)
        garnish_check = next((p for p in parsed_results if p['check'] == 'garnishing_check'), None)

        if oil_check and sop_check:
            oil_result = oil_check.get('result', '').lower()
            sop_oil = sop_check.get('oil', '').lower()
            oil_level = oil_check.get('oil_level', '').lower()
            if ('not good' in oil_result or 'excessive' in oil_level) and 'pass' in sop_oil:
                contradictions.append({
                    'type': 'Oil Level Contradiction',
                    'detail': f'Oil check says "{oil_check.get("result")}" but SOP says Oil: Pass',
                    'decision': 'OVERRIDE TO FAIL (trust oil_colour_check)'
                })

        if garnish_check and sop_check:
            garnish_result = garnish_check.get('garnishing', '').lower()
            sop_garnish = sop_check.get('garnishing', '').lower()
            if 'absent' in garnish_result and 'pass' in sop_garnish:
                contradictions.append({
                    'type': 'Garnishing Contradiction',
                    'detail': 'Garnishing check says "Absent" but SOP says Garnishing: Pass',
                    'decision': 'OVERRIDE TO FAIL (garnishing is mandatory)'
                })

        return len(contradictions) == 0, contradictions

    def is_quality_good(self, parsed_results: list) -> bool:
        is_consistent, contradictions = self.validate_consistency(parsed_results)
        self.contradictions = contradictions

        if contradictions:
            for result in parsed_results:
                check = result.get('check', '')
                if check in ['oil_colour_check', 'garnishing_check']:
                    if 'not good' in result.get('result', '').lower() or \
                       'fail' in result.get('result', '').lower():
                        return False
                if check == 'sop_compliance':
                    if 'rejected' in result.get('overall', '').lower():
                        return False
        else:
            critical_checks = {
                'cooking_quality': ['good'],
                'oil_colour_check': ['good'],
                'sop_compliance': ['approved']
            }
            for result in parsed_results:
                check = result.get('check', '')
                if check in critical_checks:
                    found_acceptable = any(
                        v.lower() in result.get('result', '').lower()
                        for v in critical_checks[check]
                    )
                    if not found_acceptable:
                        return False

        return True

    def format_output(self, results: dict) -> None:
        parsed_results = [self.parse_result(c, r) for c, r in results.items()]
        is_good = self.is_quality_good(parsed_results)

        print("\n" + "=" * 80)
        print("🍽️  FOOD QUALITY ASSESSMENT REPORT")
        print("=" * 80)

        if self.contradictions:
            print("\n⚠️  CONTRADICTION DETECTION SYSTEM ACTIVATED")
            for i, c in enumerate(self.contradictions, 1):
                print(f"\n{i}. {c['type']}")
                print(f"   Issue: {c['detail']}")
                print(f"   Action: {c['decision']}")

        if is_good:
            print("\n✅ OVERALL STATUS: APPROVED - EXCELLENT QUALITY!")
            print("\n🎉 Outstanding work! Food meets all quality standards. Keep it up! 👨‍🍳")
        else:
            print("\n❌ OVERALL STATUS: NEEDS IMPROVEMENT")
            self._print_detailed_issues(parsed_results)

        print("\n" + "=" * 80)

    def _print_detailed_issues(self, parsed_results: list) -> None:
        print("\n⚠️  ISSUES FOUND:")
        for parsed in parsed_results:
            raw = parsed.get('raw_response', '')
            if 'FAIL' in raw:
                check = parsed.get('check', '').replace('_', ' ').title()
                reason = raw.split('|')[-1].strip() if '|' in raw else raw
                print(f"  ❌ {check}: {reason}")


# ── Chef Hygiene Pipeline ──────────────────────────────────────────────────────

class ChefHygienePipeline:
    def __init__(self, vlm: BaseVLM):
        self.vlm = vlm

    def evaluate(self, image_path: str, checks: list = None) -> dict:
        image = Image.open(image_path).convert("RGB")
        checks = checks or list(CHEF_PROMPTS.keys())
        results = {}
        for check in checks:
            prompt = CHEF_PROMPTS[check]
            results[check] = self.vlm.predict(image, prompt)
        return results

    def format_output(self, results: dict) -> None:
        print("\n" + "=" * 80)
        print("👨‍🍳  CHEF HYGIENE ASSESSMENT REPORT")
        print("=" * 80)

        all_pass = all("PASS" in r for r in results.values())

        if all_pass:
            print("\n✅ OVERALL STATUS: HYGIENE APPROVED")
            print("\n🎉 Great job! You're properly dressed and following all hygiene SOPs. Keep it up!")
        else:
            print("\n❌ OVERALL STATUS: HYGIENE CHECK FAILED")
            print("\n⚠️  ISSUES FOUND — Please correct before entering the kitchen:")
            for check, response in results.items():
                if "FAIL" in response:
                    name = check.replace('_', ' ').title()
                    reason = response.split('|')[-1].strip() if '|' in response else response
                    print(f"  ❌ {name}: {reason}")

        print("\n" + "-" * 80)
        print("📋 DETAILED BREAKDOWN")
        print("-" * 80)

        check_icons = {
            "apron_check": "🥼 Apron",
            "hairnet_check": "🧢 Hairnet",
            "gloves_check": "🧤 Gloves",
            "jewellery_check": "💍 Jewellery",
            "cleanliness_check": "🧼 Cleanliness",
            "chef_sop_compliance": "✅ Overall SOP"
        }

        for check, response in results.items():
            icon_name = check_icons.get(check, check.replace('_', ' ').title())
            status = "✅ PASS" if "PASS" in response else "❌ FAIL"
            reason = response.split('|')[-1].strip() if '|' in response else ""
            print(f"\n{icon_name}: {status}")
            if reason and "FAIL" in response:
                print(f"   → {reason}")

        print("\n" + "=" * 80)


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    print("=" * 80)
    print("🍽️  MEALAWE QUALITY ASSESSMENT PIPELINE v2.0")
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

    # Ask what to check
    print("What do you want to evaluate?")
    print("  1. Food Quality")
    print("  2. Chef Hygiene")
    print("  3. Both")
    choice = input("\nEnter choice (1/2/3): ").strip()

    image_path = input("📸 Enter image path: ").strip()
    if not image_path:
        print("No path provided.")
        sys.exit(1)

    if choice in ("1", "3"):
        print("\n⏳ Running food quality checks...")
        food_pipeline = FoodQualityPipeline(vlm)
        food_results = food_pipeline.evaluate(image_path)
        food_pipeline.format_output(food_results)

    if choice in ("2", "3"):
        print("\n⏳ Running chef hygiene checks...")
        chef_pipeline = ChefHygienePipeline(vlm)
        chef_results = chef_pipeline.evaluate(image_path)
        chef_pipeline.format_output(chef_results)
