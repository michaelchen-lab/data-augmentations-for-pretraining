import os, types, torch, math
import torch.distributed as dist
from transformers import LlamaConfig, LlamaForCausalLM
from sentence_transformers import SentenceTransformer

from typing import Optional, List, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from dataset import add_prediction_mode


def to_directional_lm_model(
    model, tokenizer, l2r_percent: float = 100.0,
    max_next_i: int = 1, next_i_weighting: str = "uniform",
    next_i_temperature: float = 1.0, fixed_i: Optional[int] = None,
    include_direction: Optional[bool] = None,
):
    """
    Next-token CE with the same L2R/R2L tensor layout as self-concept (direction token + flip for R2L).

    When ``max_next_i == 1`` and ``fixed_i is None``, keeps the original single-direction-token
    behaviour: ``[<|direction|>, orig[0..L-2]]`` with next-token labels.

    When ``max_next_i > 1`` or ``fixed_i`` is set, prepends tokens per sample and predicts
    the i-th-next token at every content position. ``i`` is either sampled per sample from
    the configured weighting (train) or held fixed to ``fixed_i`` (eval).

    ``include_direction`` controls whether ``<|direction|>`` is prepended in the next-i path:

    - ``None`` (default): derived from ``l2r_percent`` — ``True`` when ``l2r_percent < 100``,
      ``False`` when ``l2r_percent == 100``. This matches training-time behaviour when the
      eval ``l2r_percent`` equals the training ``l2r_percent``.
    - ``True``: always prepend ``<|direction|>``, even when ``l2r_percent == 100``. Use this
      when evaluating at 100% L2R a model that was trained with ``l2r_percent < 100``
      (i.e. the direction token was part of the training format).
    - ``False``: never prepend ``<|direction|>``.

    Resulting sequence layouts for the next-i path:

    - ``include_direction=True``:  ``[<|direction|>, <|next_i_pred|>, orig[0..L-3]]``
    - ``include_direction=False``: ``[<|next_i_pred|>, orig[0..L-2]]``

    ``fixed_i`` is only used for evaluation.
    """
    l2r_pct = min(max(float(l2r_percent), 0.0), 100.0)
    modes = ['<|l2r_pred|>', '<|r2l_pred|>']
    pred_modes = dict(zip(modes, tokenizer.convert_tokens_to_ids(modes)))
    orig_forward = model.forward

    max_next_i = int(max_next_i)
    if max_next_i < 1:
        raise ValueError(f"max_next_i must be >= 1, got {max_next_i}")
    if fixed_i is not None and (int(fixed_i) < 1 or int(fixed_i) > max_next_i):
        raise ValueError(
            f"fixed_i must satisfy 1 <= fixed_i <= max_next_i; got fixed_i={fixed_i}, "
            f"max_next_i={max_next_i}"
        )

    use_next_i_path = (max_next_i > 1) or (fixed_i is not None)

    # Default (single-direction-token) branch; no extra setup needed.
    if not use_next_i_path:
        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            if input_ids is None:
                return orig_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs,
                )

            batch_size = input_ids.shape[0]
            l2r_batch_size = min(batch_size, max(0, int(batch_size * l2r_pct / 100.0 + 0.5)))
            parts_i, parts_l = [], []
            if l2r_batch_size > 0:
                orig = input_ids[:l2r_batch_size]
                ti = add_prediction_mode(orig, pred_modes, chosen_mode='<|l2r_pred|>')
                tl = torch.full_like(orig, -100)
                tl[:, 1:] = orig[:, :-1]
                parts_i.append(ti)
                parts_l.append(tl)
            if l2r_batch_size < batch_size:
                orig = input_ids[l2r_batch_size:]
                ti = add_prediction_mode(orig, pred_modes, chosen_mode='<|r2l_pred|>')
                tl = torch.full_like(orig, -100)
                tl[:, 1:] = torch.flip(orig, dims=[1])[:, :-1]
                parts_i.append(ti)
                parts_l.append(tl)
            input_ids = parts_i[0] if len(parts_i) == 1 else torch.cat(parts_i, dim=0)
            labels = parts_l[0] if len(parts_l) == 1 else torch.cat(parts_l, dim=0)

            return orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        model.forward = types.MethodType(forward, model)
        return

    # next-i-token-prediction path (max_next_i > 1 or fixed_i set).
    next_i_token_strs = [f'<|next_{i}_pred|>' for i in range(1, max_next_i + 1)]
    unk_id = getattr(tokenizer, 'unk_token_id', None)
    next_i_ids_list = tokenizer.convert_tokens_to_ids(next_i_token_strs)
    for tok, tid in zip(next_i_token_strs, next_i_ids_list):
        if tid is None or (unk_id is not None and tid == unk_id):
            raise ValueError(
                f"Tokenizer is missing the special token {tok!r}. "
                f"Ensure build_model_and_tokenizer added the <|next_i_pred|> tokens "
                f"(set --max-next-i appropriately)."
            )

    if next_i_weighting == "uniform":
        w = torch.ones(max_next_i, dtype=torch.float32)
        if abs(float(next_i_temperature) - 1.0) > 1e-9:
            print(
                f"[to_directional_lm_model] --next-i-weighting=uniform ignores "
                f"--next-i-temperature={next_i_temperature}."
            )
    elif next_i_weighting == "exp":
        T = float(next_i_temperature)
        if T <= 0:
            raise ValueError(f"next_i_temperature must be > 0 when weighting='exp'; got {T}")
        w = torch.exp(-torch.arange(max_next_i, dtype=torch.float32) / T)
    else:
        raise ValueError(
            f"Unknown next_i_weighting={next_i_weighting!r}; expected 'uniform' or 'exp'."
        )
    probs = w / w.sum()

    model_device = next(model.parameters()).device
    model.next_i_ids = torch.tensor(next_i_ids_list, dtype=torch.long, device=model_device)
    model.next_i_probs = probs.to(model_device)
    model.max_next_i = max_next_i
    model.fixed_i = None if fixed_i is None else int(fixed_i)
    model.l2r_percent_attr = l2r_pct
    model.pred_modes = pred_modes
    if include_direction is None:
        model.include_direction = l2r_pct < 100.0 - 1e-6
    else:
        model.include_direction = bool(include_direction)

    def _build_ti_labels(orig: torch.Tensor, i_values: torch.Tensor, direction_id: int, is_r2l: bool):
        """
        orig: (B, L) raw tokens. i_values: (B,) long. Returns (ti, labels) both (B, L).

        When include_direction is True:
            ti = [direction, next_i, source[0..L-3]]
            Labels (HF causal shift): labels[k] = source[k - 3 + i] for k >= 2.
        When include_direction is False:
            ti = [next_i, source[0..L-2]]
            Labels: labels[k] = source[k - 2 + i] for k >= 1.

        source = flip(orig) if is_r2l else orig.
        """
        B, L = orig.shape
        source = torch.flip(orig, dims=[1]) if is_r2l else orig

        ti = torch.empty_like(orig)
        if model.include_direction:
            ti[:, 0] = direction_id
            ti[:, 1] = model.next_i_ids[i_values - 1]
            ti[:, 2:] = source[:, : L - 2]
            prefix = 2
        else:
            ti[:, 0] = model.next_i_ids[i_values - 1]
            ti[:, 1:] = source[:, : L - 1]
            prefix = 1

        pos = torch.arange(L, device=orig.device)
        src = pos.unsqueeze(0) - (prefix + 1) + i_values.unsqueeze(1)  # (B, L)
        valid = (src >= 0) & (src <= (L - 1)) & (pos.unsqueeze(0) >= prefix)
        gathered = source.gather(1, src.clamp(0, L - 1))
        labels = torch.where(valid, gathered, torch.full_like(orig, -100))
        return ti, labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if input_ids is None:
            return orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        batch_size, _ = input_ids.shape
        device = input_ids.device

        # next_i_ids / next_i_probs are plain attributes (not registered buffers),
        # so Trainer's .to(device) does not move them. Lazily migrate on first
        # forward where the model lives on a different device than at install time.
        if self.next_i_ids.device != device:
            self.next_i_ids = self.next_i_ids.to(device)
            self.next_i_probs = self.next_i_probs.to(device)

        l2r_batch_size = min(
            batch_size, max(0, int(batch_size * self.l2r_percent_attr / 100.0 + 0.5))
        )

        if self.fixed_i is not None:
            i_values_full = torch.full(
                (batch_size,), int(self.fixed_i), dtype=torch.long, device=device
            )
        else:
            i_values_full = torch.multinomial(self.next_i_probs, batch_size, replacement=True) + 1

        parts_i, parts_l = [], []
        if l2r_batch_size > 0:
            orig = input_ids[:l2r_batch_size]
            i_values = i_values_full[:l2r_batch_size]
            ti, tl = _build_ti_labels(
                orig, i_values, self.pred_modes['<|l2r_pred|>'], is_r2l=False
            )
            parts_i.append(ti)
            parts_l.append(tl)
        if l2r_batch_size < batch_size:
            orig = input_ids[l2r_batch_size:]
            i_values = i_values_full[l2r_batch_size:]
            ti, tl = _build_ti_labels(
                orig, i_values, self.pred_modes['<|r2l_pred|>'], is_r2l=True
            )
            parts_i.append(ti)
            parts_l.append(tl)

        input_ids = parts_i[0] if len(parts_i) == 1 else torch.cat(parts_i, dim=0)
        labels = parts_l[0] if len(parts_l) == 1 else torch.cat(parts_l, dim=0)

        return orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    model.forward = types.MethodType(forward, model)


def to_concept_model(self, output_embedding_size: int = None, loss_func: str = ''):

    self.embedding_head = torch.nn.Linear(self.config.hidden_size, output_embedding_size)
    self.loss_func = loss_func
    if loss_func == 'cosine':
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')
    elif loss_func == 'mse':
        self.mse_loss = torch.nn.MSELoss()

    def forward(
        self,
        embedding_forward: Optional[bool] = True,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Original: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        if embedding_forward:
            embedding_output = self.embedding_head(outputs.last_hidden_state) # [batch size, max length, output embed size]

            loss = None
            if labels is not None: # [batch size, max length]
                if self.loss_func == 'mse':
                    loss = self.mse_loss(embedding_output, labels)
                elif self.loss_func == 'cosine':
                    embedding_output_flat = embedding_output.view(-1, embedding_output.size(-1))
                    labels_flat = labels.view(-1, labels.size(-1))
                    target = torch.ones(labels_flat.size(0), device=labels_flat.device)
                    loss = self.cosine_loss(embedding_output_flat, labels_flat, target)
                    if dist.is_initialized(): # account for multiple GPUs
                        world_size = dist.get_world_size()
                        loss = loss / world_size

            logits = None # this should be fine

        else:
            hidden_states = outputs.last_hidden_state # [batch size, max length, hidden size]
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :]) # [batch size, max length, vocab size]]

            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    self.forward = types.MethodType(forward, self)

def to_self_concept_model(
    self, output_embedding_size: int = None, loss_func: str = '', 
    shift: int = 1, tokenizer = None, l2r_percent: float = 50.0,
    add_l2r_token: bool = True,
):

    self.loss_func = loss_func
    if loss_func == 'cosine':
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')
    elif loss_func == 'mse':
        self.mse_loss = torch.nn.MSELoss()
    self.shift = shift
    self.l2r_percent = min(max(float(l2r_percent), 0.0), 100.0)
    modes = ['<|l2r_pred|>', '<|r2l_pred|>']
    self.pred_modes = dict(zip(modes, tokenizer.convert_tokens_to_ids(modes)))

    def forward(
        self,
        embedding_forward: Optional[bool] = True,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Original: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        ```"""
        if add_l2r_token:
            batch_size = input_ids.shape[0]
            l2r_batch_size = min(batch_size, max(0, int(batch_size * self.l2r_percent / 100.0 + 0.5)))
            transformed_inputs = []
            if l2r_batch_size > 0:
                transformed_inputs.append(
                    add_prediction_mode(
                        input_ids[:l2r_batch_size, :],
                        self.pred_modes,
                        chosen_mode='<|l2r_pred|>',
                    )
                )
            if l2r_batch_size < batch_size:
                transformed_inputs.append(
                    add_prediction_mode(
                        input_ids[l2r_batch_size:, :],
                        self.pred_modes,
                        chosen_mode='<|r2l_pred|>',
                    )
                )
            input_ids = transformed_inputs[0] if len(transformed_inputs) == 1 else torch.cat(transformed_inputs, dim=0)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_hidden_states=True,
            **kwargs,
        )

        if embedding_forward:
            input_embed = outputs.hidden_states[0].detach() # [batch size, max length, hidden size]
            target_embed = input_embed[:, 1:-(self.shift-1) or None, :]
            for i in range(1, self.shift):
                target_embed = target_embed + math.e**(-i) * input_embed[:, i+1:-(self.shift-i-1) or None, :]
            target_embed = target_embed / sum([math.e**(-i) for i in range(self.shift)])
            
            target_embed = torch.nn.functional.normalize(target_embed, p=2, dim=-1)
            pred_embed = outputs.last_hidden_state[:, :-self.shift, :] # shift 1
            pred_embed = torch.nn.functional.normalize(pred_embed, p=2, dim=-1)

            if self.loss_func == 'mse':
                loss = self.mse_loss(pred_embed, target_embed)
            elif self.loss_func == 'cosine':
                pred_embed_flat = pred_embed.view(-1, pred_embed.size(-1))
                target_embed_flat = target_embed.view(-1, target_embed.size(-1))
                target = torch.ones(target_embed_flat.size(0), device=target_embed_flat.device)
                loss = self.cosine_loss(pred_embed_flat, target_embed_flat, target)
                if dist.is_initialized(): # account for multiple GPUs
                    world_size = dist.get_world_size()
                    loss = loss / world_size
            
            logits = None # this should be fine

        else:
            hidden_states = outputs.last_hidden_state # [batch size, max length, hidden size]
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :]) # [batch size, max length, vocab size]]

            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    self.forward = types.MethodType(forward, self)

def apply_token_masking(
    model, tokenizer, mask_percent: float, random_token_percent: float,
    protected_tokens: list,
):
    """
    Wrap model.forward to apply per-token input masking during training.

    Two independent replacement modes can be active simultaneously, controlled by
    separate percentage parameters (both in 0..100).  A single uniform random draw
    ``u ~ Uniform(0, 1)`` per token assigns each position to at most one mode:

    * ``u < mask_rate``                            → replace with ``<|mask|>``
    * ``mask_rate <= u < mask_rate + random_rate`` → replace with a random non-special vocab token
    * otherwise                                    → keep original

    This guarantees the two modes never overlap and each operates at its configured
    rate.  ``mask_percent + random_token_percent`` must not exceed 100.

    ``<|mask|>`` is only added to the vocabulary when ``mask_percent > 0``.  If only
    ``random_token_percent > 0``, no ``<|mask|>`` token is needed or registered.

    Labels are never modified, so the AR objective always targets the original tokens.

    Protection registry contract
    ----------------------------
    ``protected_tokens`` must be the same ``additional_specials`` list passed to
    ``tokenizer.add_special_tokens(...)`` in ``build_model_and_tokenizer``.  Every
    augmentation control token (``<|l2r_pred|>``, ``<|r2l_pred|>``, ``<|next_*_pred|>``,
    ``<|mask|>`` itself) is automatically excluded from both replacement modes.

    The wrapper is active only when ``model.training is True``; eval paths are unaffected.
    """
    mask_rate = float(mask_percent) / 100.0
    random_rate = float(random_token_percent) / 100.0
    if mask_rate + random_rate > 1.0 + 1e-9:
        raise ValueError(
            f"mask_percent ({mask_percent}) + random_token_percent ({random_token_percent}) "
            f"must not exceed 100."
        )

    protected_ids = sorted(set(tokenizer.convert_tokens_to_ids(protected_tokens)))
    device = next(model.parameters()).device
    model.mask_rate = mask_rate
    model.random_rate = random_rate
    model.protected_token_ids = torch.tensor(protected_ids, dtype=torch.long, device=device)

    # <|mask|> token — only needed when mask_percent > 0
    if mask_rate > 0.0:
        mask_id = tokenizer.convert_tokens_to_ids('<|mask|>')
        unk_id = getattr(tokenizer, 'unk_token_id', None)
        if mask_id is None or (unk_id is not None and mask_id == unk_id):
            raise ValueError(
                "Tokenizer is missing <|mask|>. Ensure build_model_and_tokenizer registered it "
                "(mask_percent > 0 triggers automatic registration)."
            )
        model.mask_token_id = int(mask_id)
    else:
        model.mask_token_id = None

    # Random vocab pool — needed when random_token_percent > 0
    if random_rate > 0.0:
        core_special_ids: set[int] = set()
        for attr in ('pad_token_id', 'bos_token_id', 'eos_token_id', 'unk_token_id',
                     'sep_token_id', 'cls_token_id', 'mask_token_id'):
            val = getattr(tokenizer, attr, None)
            if val is not None:
                core_special_ids.add(int(val))
        all_special_ids = core_special_ids | set(protected_ids)
        vocab_size = len(tokenizer)
        pool = torch.tensor(
            [i for i in range(vocab_size) if i not in all_special_ids],
            dtype=torch.long, device=device,
        )
        if pool.numel() == 0:
            raise ValueError(
                "Random token pool is empty: all vocab IDs are classified as special. "
                "Check tokenizer special-token setup."
            )
        model._random_vocab_pool = pool
    else:
        model._random_vocab_pool = None

    inner_forward = model.forward

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if self.training and input_ids is not None and (self.mask_rate > 0.0 or self.random_rate > 0.0):
            if self.protected_token_ids.device != input_ids.device:
                self.protected_token_ids = self.protected_token_ids.to(input_ids.device)

            protected = torch.isin(input_ids, self.protected_token_ids)
            u = torch.rand(input_ids.shape, device=input_ids.device)
            result = input_ids.clone()

            # Random token replacement (threshold band above mask band)
            if self.random_rate > 0.0:
                if self._random_vocab_pool.device != input_ids.device:
                    self._random_vocab_pool = self._random_vocab_pool.to(input_ids.device)
                do_random = (u >= self.mask_rate) & (u < self.mask_rate + self.random_rate) & ~protected
                n_random = int(do_random.sum().item())
                if n_random > 0:
                    idx = torch.randint(
                        len(self._random_vocab_pool), (n_random,), device=input_ids.device
                    )
                    result[do_random] = self._random_vocab_pool[idx]

            # <|mask|> replacement (lowest threshold band, highest priority in the draw)
            if self.mask_rate > 0.0:
                do_mask = (u < self.mask_rate) & ~protected
                result[do_mask] = self.mask_token_id

            input_ids = result
        return inner_forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    model.forward = types.MethodType(forward, model)


def apply_fim_augmentation(
    model, tokenizer, psm_percent: float, spm_percent: float,
    protected_tokens: list,
):
    """
    Wrap model.forward to apply Fill-in-the-Middle (FIM) data augmentation during training.

    Each training sample is independently routed to one of three modes based on a single
    uniform draw ``u ~ Uniform(0, 1)`` per sample:

    * ``u < psm_rate``                      → PSM (Prefix-Suffix-Middle)
    * ``psm_rate <= u < psm_rate + spm_rate`` → SPM (Suffix-Prefix-Middle)
    * otherwise                             → unchanged

    Two pivots ``a < b`` are sampled uniformly at random (without replacement) from
    ``{0, ..., L_content-1}`` where ``L_content = L - 3`` reserves 3 positions for the
    FIM control tokens.  This guarantees the output is always exactly length ``L`` without
    truncation.  The content used is ``tokens[:L_content]`` (the last 3 tokens of the
    original sequence are always dropped for FIM samples).

    The three segments are:

        prefix = content[:a],  middle = content[a:b],  suffix = content[b:]

    Resulting sequences:

        PSM: <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle
        SPM: <fim_prefix> suffix <fim_suffix> prefix <fim_middle> middle

    ``labels`` are set to the FIM-transformed sequence so the AR objective targets every
    token position.  When ``to_directional_lm_model`` is installed as an inner wrapper, it
    will override ``labels`` from the transformed ``input_ids``; when it is absent the
    ``labels`` are used directly by Llama's HF loss.

    ``psm_percent + spm_percent`` must not exceed 100.

    FIM control tokens (``<|fim_prefix|>``, ``<|fim_suffix|>``, ``<|fim_middle|>``) are
    always included in ``protected_tokens`` so that ``apply_token_masking`` (if also
    installed) never corrupts them.

    The wrapper is active only when ``model.training is True``; eval paths are unaffected.
    Sequences shorter than 5 tokens (``L_content < 2``) are silently skipped.
    """
    psm_rate = float(psm_percent) / 100.0
    spm_rate = float(spm_percent) / 100.0

    if psm_rate < 0 or spm_rate < 0:
        raise ValueError(
            f"psm_percent ({psm_percent}) and spm_percent ({spm_percent}) must both be >= 0."
        )
    if psm_rate + spm_rate > 1.0 + 1e-9:
        raise ValueError(
            f"psm_percent ({psm_percent}) + spm_percent ({spm_percent}) must not exceed 100."
        )
    if psm_rate + spm_rate < 1e-9:
        return  # no-op; nothing to install

    fim_prefix_id = tokenizer.convert_tokens_to_ids('<|fim_prefix|>')
    fim_suffix_id = tokenizer.convert_tokens_to_ids('<|fim_suffix|>')
    fim_middle_id = tokenizer.convert_tokens_to_ids('<|fim_middle|>')
    unk_id = getattr(tokenizer, 'unk_token_id', None)
    for tok, tid in [
        ('<|fim_prefix|>', fim_prefix_id),
        ('<|fim_suffix|>', fim_suffix_id),
        ('<|fim_middle|>', fim_middle_id),
    ]:
        if tid is None or (unk_id is not None and tid == unk_id):
            raise ValueError(
                f"Tokenizer is missing the special token {tok!r}. "
                "Ensure build_model_and_tokenizer registered FIM tokens "
                "(psm_percent > 0 or spm_percent > 0 triggers automatic registration)."
            )

    model.fim_psm_rate = psm_rate
    model.fim_spm_rate = spm_rate
    model.fim_prefix_id = int(fim_prefix_id)
    model.fim_suffix_id = int(fim_suffix_id)
    model.fim_middle_id = int(fim_middle_id)

    inner_forward = model.forward

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if self.training and input_ids is not None:
            B, L = input_ids.shape
            L_content = L - 3  # reserve 3 positions for the three FIM control tokens
            if L_content >= 2:  # need at least 2 content tokens for distinct pivots
                result = input_ids.clone()
                u = torch.rand(B, device=input_ids.device)

                for b in range(B):
                    if u[b] < self.fim_psm_rate:
                        mode = 'psm'
                    elif u[b] < self.fim_psm_rate + self.fim_spm_rate:
                        mode = 'spm'
                    else:
                        continue  # leave this sample unchanged

                    content = input_ids[b, :L_content]

                    # Sample two distinct pivot indices uniformly from [0, L_content)
                    perm = torch.randperm(L_content, device=input_ids.device)
                    a = int(perm[0].item())
                    bi = int(perm[1].item())
                    if a > bi:
                        a, bi = bi, a
                    # a < bi; prefix=content[:a], middle=content[a:bi], suffix=content[bi:]

                    prefix = content[:a]
                    middle = content[a:bi]
                    suffix = content[bi:]

                    fp = input_ids.new_tensor([self.fim_prefix_id])
                    fs = input_ids.new_tensor([self.fim_suffix_id])
                    fm = input_ids.new_tensor([self.fim_middle_id])

                    if mode == 'psm':
                        fim_seq = torch.cat([fp, prefix, fs, suffix, fm, middle])
                    else:  # spm
                        fim_seq = torch.cat([fp, suffix, fs, prefix, fm, middle])

                    # fim_seq is always exactly L tokens (L_content + 3 control tokens)
                    result[b] = fim_seq

                input_ids = result
                labels = result.clone()

        return inner_forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    model.forward = types.MethodType(forward, model)


def build_model_and_tokenizer(args):
    if torch.cuda.is_available():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", model_max_length=args.model_max_length)
    # tokenizer.pad_token = tokenizer.eos_token

    # load_in_8bit makes it 2x slower for some reason
    # Non-concept runs only need the tokenizer; load on CPU so every rank does not put Qwen on cuda:0.
    if args.model_type == 'concept':
        embedding_model = SentenceTransformer(
            args.embedding_model,
            model_kwargs={'dtype': 'float16'},
            device=str(device),
        )
    else:
        embedding_model = SentenceTransformer(args.embedding_model, device='cpu')
    print(f'Embedding model {args.embedding_model} loaded.')

    tokenizer = embedding_model.tokenizer
    additional_specials = ['<|l2r_pred|>', '<|r2l_pred|>']
    max_next_i = int(getattr(args, 'max_next_i', 1))
    if max_next_i > 1:
        additional_specials += [f'<|next_{i}_pred|>' for i in range(1, max_next_i + 1)]
    mask_percent = float(getattr(args, 'mask_percent', 0.0))
    random_token_percent = float(getattr(args, 'random_token_percent', 0.0))
    if mask_percent > 0.0:
        additional_specials += ['<|mask|>']
    psm_percent = float(getattr(args, 'psm_percent', 0.0))
    spm_percent = float(getattr(args, 'spm_percent', 0.0))
    if psm_percent > 0.0 or spm_percent > 0.0:
        additional_specials += ['<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>']
    tokenizer.add_special_tokens({'additional_special_tokens': additional_specials})
    if args.model_type != 'concept':
        embedding_model = None
    tokenizer.pad_token = tokenizer.eos_token
    print('EOS token:', tokenizer.eos_token)

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.model_hidden_size,
        intermediate_size=args.model_intermediate_size,
        num_hidden_layers=args.model_num_layers,
        num_attention_heads=args.model_num_attention_heads,
        max_position_embeddings=args.model_max_length,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True, # Qwen ties but Llama does not. Tied due to Qwen tokenizer's large vocab size
    )

    model = LlamaForCausalLM(config).to(device)
    print('total params:', sum(p.numel() for p in model.parameters()))

    if args.model_type == 'concept':
        to_concept_model(model, output_embedding_size=args.output_embedding_size, loss_func=args.embed_loss_func)
    elif args.model_type == 'self-concept':
        to_self_concept_model(
            model, output_embedding_size=args.output_embedding_size, 
            loss_func=args.embed_loss_func, shift=args.embed_window_size,
            tokenizer=tokenizer, l2r_percent=args.l2r_percent,
        )
    elif args.model_type == 'default' and (
        abs(args.l2r_percent - 100.0) > 1e-6 or max_next_i > 1
        or mask_percent > 0.0 or random_token_percent > 0.0
        or psm_percent > 0.0 or spm_percent > 0.0
    ):
        if abs(args.l2r_percent - 100.0) > 1e-6 or max_next_i > 1:
            to_directional_lm_model(
                model,
                tokenizer,
                l2r_percent=args.l2r_percent,
                max_next_i=max_next_i,
                next_i_weighting=getattr(args, 'next_i_weighting', 'uniform'),
                next_i_temperature=float(getattr(args, 'next_i_temperature', 1.0)),
            )
        if psm_percent > 0.0 or spm_percent > 0.0:
            apply_fim_augmentation(
                model, tokenizer,
                psm_percent=psm_percent,
                spm_percent=spm_percent,
                protected_tokens=additional_specials,
            )
        if mask_percent > 0.0 or random_token_percent > 0.0:
            apply_token_masking(
                model, tokenizer,
                mask_percent=mask_percent,
                random_token_percent=random_token_percent,
                protected_tokens=additional_specials,
            )

    return model, tokenizer

