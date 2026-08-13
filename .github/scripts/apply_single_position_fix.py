from pathlib import Path


path = Path("src/transformers/generation/candidate_generator.py")
source = path.read_text()

old = '''        # Drafter autoregressive loop
        drafted_logits = []
        drafted_tokens = []

        for _ in range(max_new_tokens):
            last_token_embedding = self.target_model_input_embeddings(last_token_id)
            inputs_embeds = torch.cat([last_token_embedding, last_hidden_state], dim=-1)

            with torch.no_grad():
                outputs = self.assistant_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=model_kwargs.get("attention_mask"),
                    position_ids=position_ids,
                    shared_kv_states=shared_kv_states,
                    use_cache=False,
                )

            last_token_id = outputs.logits.argmax(dim=-1)
            last_hidden_state = outputs.last_hidden_state

            # For stopped sequences, replace drafted tokens with pad and logits with zeros.
            if sequence_stopped.any():
                stopped = sequence_stopped.unsqueeze(1)  # (batch, 1) for broadcasting
                last_token_id = torch.where(stopped, self.generation_config.pad_token_id, last_token_id)
                drafted_logits.append(
                    torch.where(stopped.unsqueeze(-1), torch.zeros_like(outputs.logits), outputs.logits)
                )
            else:
                drafted_logits.append(outputs.logits)

            drafted_tokens.append(last_token_id)

            # Update stop status: mark sequences whose latest token is an EOS token.
            if self.eos_token_id is not None:
                sequence_stopped = torch.logical_or(
                    sequence_stopped,
                    torch.isin(last_token_id.squeeze(1), self.eos_token_id.to(last_token_id.device)),
                )
                if sequence_stopped.all():
                    break

        # --- Assemble output ---
        candidate_ids = torch.cat([input_ids, torch.cat(drafted_tokens, dim=1)], dim=1)
        candidate_logits = torch.cat(drafted_logits, dim=1)
        return candidate_ids, candidate_logits
'''

new = '''        # Drafter autoregressive loop
        drafted_logits = []
        candidate_ids = input_ids

        for _ in range(max_new_tokens):
            last_token_embedding = self.target_model_input_embeddings(last_token_id)
            inputs_embeds = torch.cat([last_token_embedding, last_hidden_state], dim=-1)

            with torch.no_grad():
                outputs = self.assistant_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=model_kwargs.get("attention_mask"),
                    position_ids=position_ids,
                    shared_kv_states=shared_kv_states,
                    use_cache=False,
                )

            next_token_scores = outputs.logits[:, -1, :]
            if self.logits_processor:
                next_token_scores = LogitsProcessorList(self.logits_processor)(candidate_ids, next_token_scores.float())
            if self.generation_config.do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1, dtype=torch.float32)
                last_token_id = torch.multinomial(probs, num_samples=1)
            else:
                last_token_id = torch.argmax(next_token_scores, dim=-1, keepdim=True)
            last_hidden_state = outputs.last_hidden_state

            # For stopped sequences, replace drafted tokens with pad and logits with zeros.
            if sequence_stopped.any():
                stopped = sequence_stopped.unsqueeze(1)  # (batch, 1) for broadcasting
                last_token_id = torch.where(stopped, self.generation_config.pad_token_id, last_token_id)
                drafted_logits.append(
                    torch.where(
                        stopped.unsqueeze(-1),
                        torch.zeros_like(next_token_scores).unsqueeze(1),
                        next_token_scores.unsqueeze(1),
                    )
                )
            else:
                drafted_logits.append(next_token_scores.unsqueeze(1))

            candidate_ids = torch.cat([candidate_ids, last_token_id], dim=1)

            # Update stop status: mark sequences whose latest token is an EOS token.
            if self.eos_token_id is not None:
                sequence_stopped = torch.logical_or(
                    sequence_stopped,
                    torch.isin(last_token_id.squeeze(1), self.eos_token_id.to(last_token_id.device)),
                )
                if sequence_stopped.all():
                    break

        candidate_logits = torch.cat(drafted_logits, dim=1)
        return candidate_ids, candidate_logits
'''

count = source.count(old)
if count != 1:
    raise RuntimeError(f"Expected exactly one SinglePosition drafting block, found {count}")

path.write_text(source.replace(old, new, 1))
