from pathlib import Path


path = Path("src/transformers/generation/candidate_generator.py")
source = path.read_text()
old = "                next_token_scores = LogitsProcessorList(self.logits_processor)(candidate_ids, next_token_scores.float())\n"
new = "                next_token_scores = LogitsProcessorList(self.logits_processor)(\n                    candidate_ids, next_token_scores.float()\n                )\n"
if source.count(old) != 1:
    raise RuntimeError("Expected exactly one unwrapped logits processor call")
path.write_text(source.replace(old, new, 1))
