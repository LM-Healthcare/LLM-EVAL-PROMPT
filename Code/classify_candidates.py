"""
Step 2: Classify sampled candidates into 4 macro-categories using an LLM.

Usage:
    python classify_candidates.py --model model_a
    python classify_candidates.py --model model_b

Reads from: output/sampled_candidates.jsonl
Writes to:  output/classified_<model_key>.jsonl

Each question gets a 'classification' object with:
  - category: one of the 4 macro-categories
  - confidence: high / medium / low
  - reasoning: brief LLM explanation
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from config import (
    CANDIDATES_FILE,
    CATEGORIES,
    CATEGORY_DESCRIPTIONS,
    MAX_CONCURRENT_REQUESTS,
    MAX_RETRIES,
    MODEL_CONFIGS,
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a medical education expert. Your task is to classify medical multiple-choice questions into exactly one of 4 macro-categories.

## Categories

{categories}

## Instructions

- Read the question text and all answer options carefully.
- Consider the clinical context, not just the subject label.
- Choose the single BEST matching category.
- Provide a confidence level: "high", "medium", or "low".
- Provide a brief reasoning (1-2 sentences max).

## Output Format

Respond with ONLY a JSON object (no markdown, no extra text):
{{"category": "<category_name>", "confidence": "<high|medium|low>", "reasoning": "<brief explanation>"}}
"""

USER_PROMPT_TEMPLATE = """Classify the following medical question:

**Subject:** {subject_name}
**Question:** {question}

**A.** {opa}
**B.** {opb}
**C.** {opc}
**D.** {opd}
"""


def build_system_prompt():
    cat_text = ""
    for i, cat in enumerate(CATEGORIES, 1):
        cat_text += f"{i}. **{cat}**: {CATEGORY_DESCRIPTIONS[cat]}\n\n"
    return SYSTEM_PROMPT.format(categories=cat_text.strip())


def build_user_prompt(question):
    return USER_PROMPT_TEMPLATE.format(
        subject_name=question.get("subject_name", "N/A"),
        question=question.get("question", ""),
        opa=question.get("opa", ""),
        opb=question.get("opb", ""),
        opc=question.get("opc", ""),
        opd=question.get("opd", ""),
    )


# ---------------------------------------------------------------------------
# LLM Client abstraction
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified async client for Anthropic and OpenAI-compatible APIs."""

    def __init__(self, model_config):
        self.provider = model_config["provider"]
        self.model_name = model_config["model_name"]
        self.api_key = os.environ.get(model_config["api_key_env"], "")
        self.base_url = model_config.get("base_url")

        if not self.api_key:
            print(f"ERROR: Environment variable '{model_config['api_key_env']}' is not set.")
            print(f"Set it with:  $env:{model_config['api_key_env']}='your-key-here'")
            sys.exit(1)

        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                print("ERROR: 'anthropic' package not installed. Run: pip install anthropic")
                sys.exit(1)
        elif self.provider == "openai_compatible":
            try:
                import openai
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = openai.AsyncOpenAI(**kwargs)
            except ImportError:
                print("ERROR: 'openai' package not installed. Run: pip install openai")
                sys.exit(1)
        else:
            print(f"ERROR: Unknown provider '{self.provider}'")
            sys.exit(1)

    async def classify(self, system_prompt, user_prompt):
        """Send a classification request and return the parsed JSON response."""
        if self.provider == "anthropic":
            return await self._classify_anthropic(system_prompt, user_prompt)
        else:
            return await self._classify_openai(system_prompt, user_prompt)

    async def _classify_anthropic(self, system_prompt, user_prompt):
        response = await self._client.messages.create(
            model=self.model_name,
            max_tokens=256,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    async def _classify_openai(self, system_prompt, user_prompt):
        response = await self._client.chat.completions.create(
            model=self.model_name,
            max_completion_tokens=256,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    async def close(self):
        if self.provider == "anthropic":
            await self._client.close()
        else:
            await self._client.close()


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def parse_llm_response(raw_text):
    """Parse the LLM JSON response, handling common formatting issues."""
    text = raw_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            cat = obj.get("category", "")
            # Validate category
            if cat not in CATEGORIES:
                # Try fuzzy match
                for valid_cat in CATEGORIES:
                    if valid_cat.lower() in cat.lower() or cat.lower() in valid_cat.lower():
                        obj["category"] = valid_cat
                        break
                else:
                    obj["category"] = "Mixed"
                    obj["confidence"] = "low"
                    obj["reasoning"] = f"[PARSE WARNING: unrecognized category '{cat}'] " + obj.get("reasoning", "")
            return obj
        except json.JSONDecodeError:
            pass
    # Fallback
    return {
        "category": "Mixed",
        "confidence": "low",
        "reasoning": f"[PARSE ERROR] Raw response: {raw_text[:200]}",
    }


async def classify_single(client, system_prompt, question, semaphore, index, total):
    """Classify a single question with retries."""
    user_prompt = build_user_prompt(question)
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                raw = await asyncio.wait_for(
                    client.classify(system_prompt, user_prompt),
                    timeout=REQUEST_TIMEOUT,
                )
                result = parse_llm_response(raw)
                if (index + 1) % 50 == 0 or index == 0:
                    print(f"  [{index+1}/{total}] {question.get('subject_name', '?')} → {result['category']} ({result['confidence']})")
                return result
            except Exception as e:
                delay = RETRY_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    print(f"  [{index+1}/{total}] Retry {attempt+1}/{MAX_RETRIES} after error: {e}")
                    await asyncio.sleep(delay)
                else:
                    print(f"  [{index+1}/{total}] FAILED after {MAX_RETRIES} attempts: {e}")
                    return {
                        "category": "Mixed",
                        "confidence": "low",
                        "reasoning": f"[API ERROR] {str(e)[:200]}",
                    }


BATCH_SIZE = 50  # Save progress every N questions


async def run_classification(model_key):
    """Run classification for all candidates using the specified model.

    Progress is saved incrementally every BATCH_SIZE questions by appending
    to the output JSONL file, so the run can be safely resumed after a crash
    or quota error.
    """
    if model_key not in MODEL_CONFIGS:
        print(f"ERROR: Unknown model key '{model_key}'. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    config = MODEL_CONFIGS[model_key]
    print(f"Model: {config['model_name']} (provider: {config['provider']})")

    # Load candidates
    if not CANDIDATES_FILE.exists():
        print(f"ERROR: Candidates file not found: {CANDIDATES_FILE}")
        print("Run sample_candidates.py first.")
        sys.exit(1)

    candidates = []
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates from {CANDIDATES_FILE}")

    # Check for existing progress (resume support)
    output_file = OUTPUT_DIR / f"classified_{model_key}.jsonl"
    done_count = 0
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_count += 1
        if done_count >= len(candidates):
            print(f"Already fully classified ({done_count}/{len(candidates)}). Nothing to do.")
            return
        print(f"Resuming from question {done_count + 1}/{len(candidates)} ({done_count} already done)")

    # Initialize client
    client = LLMClient(config)
    system_prompt = build_system_prompt()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    remaining = candidates[done_count:]
    total = len(candidates)

    print(f"\nClassifying {len(remaining)} questions in batches of {BATCH_SIZE}...")
    t0 = time.time()
    classified_total = done_count

    # Process in batches so we can save after each batch
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        global_offset = done_count + batch_start

        tasks = [
            classify_single(client, system_prompt, q, semaphore, global_offset + i, total)
            for i, q in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks)

        # Append batch results to file immediately
        with open(output_file, "a", encoding="utf-8") as f:
            for q, classification in zip(batch, results):
                q_out = dict(q)
                q_out["classification"] = classification
                f.write(json.dumps(q_out, ensure_ascii=False) + "\n")

        classified_total += len(batch)
        elapsed = time.time() - t0
        rate = classified_total - done_count
        print(f"  [checkpoint] {classified_total}/{total} saved ({rate} done in {elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nClassification complete in {elapsed:.1f}s")
    print(f"Saved → {output_file}")

    # Quick stats (read back all results)
    from collections import Counter
    all_classified = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_classified.append(json.loads(line))
    cat_counts = Counter(q["classification"]["category"] for q in all_classified)
    conf_counts = Counter(q["classification"]["confidence"] for q in all_classified)
    print(f"\nCategory distribution:")
    for cat in CATEGORIES:
        print(f"  {cat}: {cat_counts.get(cat, 0)}")
    print(f"\nConfidence distribution:")
    for conf in ["high", "medium", "low"]:
        print(f"  {conf}: {conf_counts.get(conf, 0)}")

    await client.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classify MedMCQA candidates with an LLM")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Which model configuration to use (defined in config.py)",
    )
    args = parser.parse_args()
    asyncio.run(run_classification(args.model))


if __name__ == "__main__":
    main()
