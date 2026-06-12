from huggingface_hub import snapshot_download
from datasets import Dataset
from pathlib import Path
import torch, json, copy, os, time, random
from itertools import chain
import numpy as np
from tqdm import tqdm

def tokenize_raw(examples, tokenizer=None):
    return tokenizer(examples["text"])

def group_texts(examples, args=None):
    '''
    Pack multiple sequences into batches to fill the entire context
    `examples`: {'input_ids': [...], 'attention_mask': [...]}
    '''
    block_size = args.model_max_length
    # Concatenate all texts in this batch.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples['input_ids'])
    if total_length >= block_size: # drop the remainder
        total_length = (total_length // block_size) * block_size

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()

    return result

def choose_prediction_mode(pred_modes: list, l2r_percent: float = 50.0, chosen_mode: str = None):
    if chosen_mode is not None:
        return chosen_mode

    l2r_percent = min(max(float(l2r_percent), 0.0), 100.0)
    if random.random() * 100 < l2r_percent:
        return '<|l2r_pred|>'
    return '<|r2l_pred|>'


def add_prediction_mode(input_ids, pred_modes:list=None, chosen_mode:str=None, l2r_percent: float = 50.0):
    '''
    Randomly select L2R or R2L mode, then change input_id accordingly.
    input_ids: [batch_size, seq_len]
    '''
    mode = choose_prediction_mode(pred_modes, l2r_percent=l2r_percent, chosen_mode=chosen_mode)

    if mode == '<|l2r_pred|>':
        input_ids = torch.nn.functional.pad(input_ids, (1, 0), "constant", pred_modes[mode])[:, :-1]
    elif mode == '<|r2l_pred|>':
        input_ids = torch.nn.functional.pad(torch.flip(input_ids, dims=[1]), (1, 0), "constant", pred_modes[mode])[:, :-1]
    return input_ids

class ConceptDataset(Dataset):
    '''Dynamic embedding targets and attention masks for concept learning'''
    def __init__(
        self, texts, tokenizer, embed_model, args, 
        max_length=512, embeddings_dir=None, save_embed_filename='sample_embed.pt'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embed_model_name = args.embedding_model
        self.embed_model = None
        self.no_embed_sliding_window_attn = args.no_embed_sliding_window_attn
        self.batch_size = args.batch_size_per_device*args.embedding_batch_size_multiplier # for embedding inferencer

        text_dataset = Dataset.from_dict({'text': texts})
        text_dataset = text_dataset.map(tokenize_raw, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer})
        text_dataset = text_dataset.map(group_texts, batched=True, batch_size=1000, fn_kwargs={"args": args})
        self.input_ids = list(text_dataset['input_ids'])
        self.attention_mask = list(text_dataset['attention_mask'])
        del text_dataset

        self.window_size = args.embed_window_size
        self.l2r_percent = args.l2r_percent
        modes = ['<|l2r_pred|>', '<|r2l_pred|>']
        self.pred_modes = dict(zip(modes, tokenizer.convert_tokens_to_ids(modes)))
        # if args.model_type == 'concept':
        #     self.embeddings = self.exec_batch_embeddings_inference(embeddings_dir=args.embed_dir, save_embed_filename=args.save_embed_filename)
        self.embed_model_cuda = 0
        self.cuda_device_ids = list(range(torch.cuda.device_count()))

    def _init_embed_model(self):
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Use the local GPU assigned to this specific rank/process
            # This prevents both ranks from piling onto cuda:1
            if torch.cuda.is_available():
                # device = f"cuda:{torch.cuda.current_device()}"
                # Evenly distribute embed_model across GPUs.
                local_rank = int(os.environ.get("LOCAL_RANK", 0)) # Get the local rank assigned to this process by the launcher
                device_id = local_rank % torch.cuda.device_count()
                device = f"cuda:{device_id}"
            else:
                device = "cpu"
                
            print(f"Worker process loading embedding model on {device}")
            self.embed_model = SentenceTransformer(
                self.embed_model_name, 
                device=device,
                model_kwargs={"torch_dtype": torch.float16} # Ensure half-precision to save VRAM
            )
            self.embed_model.eval()

    def exec_batch_embeddings_inference(self, input_ids=None, embeddings_dir=None, save_embed_filename=None):
        self._init_embed_model()

        if embeddings_dir and os.path.isfile(embeddings_dir):
            return torch.load(embeddings_dir, weights_only=True)

        if input_ids == None: input_ids = self.input_ids

        embeddings = []
        for i in range(0, len(input_ids), self.batch_size): # ADJUST BATCH SIZE
            batch_ids = input_ids.detach().clone()[i : i + self.batch_size].to(self.embed_model.device) # [batch_size, seq_len]
            # Add pseudo special token at the front and remove last token [batch_size, seq_len]
            padded_input_ids = torch.nn.functional.pad(batch_ids, (1, 0), "constant", self.embed_model.tokenizer.pad_token_id)[:, :-1]
            # Add padding on both sides [batch_size, seq_len+2*(window_size-1)]
            padded_input_ids = torch.nn.functional.pad(padded_input_ids, (self.window_size-1, self.window_size-1), "constant", self.embed_model.tokenizer.pad_token_id)
            
            sliding_mask = None
            if not self.no_embed_sliding_window_attn:
                sliding_mask = self.create_sliding_window_mask(padded_input_ids.shape[1], self.window_size) # [seq_len+2*(window_size-1), seq_len+2*(window_size-1)]
                sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0).expand(padded_input_ids.shape[0], -1, -1, -1) # [batch_size, 1, seq_len+2*(window_size-1), seq_len+2*(window_size-1)]
                sliding_mask = sliding_mask.to(self.embed_model.device, dtype=self.embed_model[0].auto_model.dtype)
            with torch.no_grad():
                outputs = self.embed_model[0].auto_model(padded_input_ids, attention_mask=sliding_mask)
                token_embeddings = outputs.last_hidden_state.to('cpu') # Shape: [batch_size, seq_len+2*(window_size-1), hidden_dim]
            embeddings.append(token_embeddings)

        embeddings = torch.cat(embeddings, dim=0)
        if save_embed_filename:
            torch.save(embeddings, save_embed_filename)

        return embeddings

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.input_ids)

    def create_sliding_window_mask(self, seq_len, window_size):
        """
        Creates a causal mask that limits attention to a fixed window size.
        See https://gemini.google.com/share/152ae435e544
        """
        attn_shape = (seq_len, seq_len)

        # 2. Standard Causal Mask (Lower Triangular)
        # Keeps j <= i
        causal_mask = torch.tril(torch.ones(attn_shape, dtype=torch.bool))

        # 3. Window Mask (Upper Triangular from a lower diagonal)
        # Keeps j >= i - (window_size - 1)
        # diagonal=-(window_size - 1) shifts the diagonal down by window_size - 1
        lookback_mask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=-(window_size - 1))

        # 4. Combine them (Intersection)
        # We want positions that satisfy BOTH causal AND lookback constraints
        final_mask = causal_mask & lookback_mask

        # Convert to float mask for attention (0.0 for allow, -inf for mask)
        # Many models expect a float mask where masked positions are -inf
        float_mask = torch.zeros(attn_shape)
        float_mask.masked_fill_(~final_mask, float('-inf'))

        return float_mask

    def fast_sliding_window_embedding(self, input_ids, window_size, stride, mode):
        '''
        R2L requires some extra work. The forward pass must still be L2R.
        Steps for R2L:
        1. Reverse input_ids back to L2R.
        2. Add padding at the start instead of the back
        3. At the end, reverse the order of the embeddings.
        '''
        # input_id Shape: [1, seq_len]
        input_ids_wo_special = input_ids.clone()
        input_ids_wo_special[0, 0] = self.embed_model.tokenizer.pad_token_id # replace the special token
        if mode == '<|r2l_pred|>':
            input_ids_wo_special = torch.flip(input_ids_wo_special, dims=[1])
        padding_for_window = {'<|l2r_pred|>': (0, window_size-1), '<|r2l_pred|>': (window_size-1, 0)}[mode]
        input_ids_wo_special = torch.nn.functional.pad(input_ids_wo_special, padding_for_window, "constant", self.embed_model.tokenizer.pad_token_id)
        # Shape: [seq_len+window_size-1, hidden_dim]

        # Run the model once
        sliding_mask = self.create_sliding_window_mask(input_ids_wo_special.shape[1], window_size)
        sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0).to(self.embed_model.device, dtype=self.embed_model[0].auto_model.dtype)
        with torch.no_grad():
            outputs = self.embed_model[0].auto_model(input_ids_wo_special.to(self.embed_model.device))
            token_embeddings = outputs.last_hidden_state[0] # Shape: [seq_len+window_size-1, hidden_dim]

        window_embeddings = token_embeddings[window_size-1:] # Shape: [seq_len, hidden_dim]
        if mode == '<|r2l_pred|>':
            window_embeddings = torch.flip(window_embeddings, dims=[0])
        return window_embeddings

    def get_embeddings(self, _id, mode, raw_embed=None):
        if raw_embed == None:
            raw_embed = self.embeddings[_id]
        if mode == '<|l2r_pred|>':
            return raw_embed[(self.window_size-1)*2:]
        elif mode == '<|r2l_pred|>':
            return torch.flip(raw_embed[self.window_size-1: -(self.window_size-1)], dims=[0])

    def add_prediction_mode(self, input_id, chosen_mode=None):
        mode = choose_prediction_mode(
            self.pred_modes,
            l2r_percent=self.l2r_percent,
            chosen_mode=chosen_mode,
        )
        if mode == '<|l2r_pred|>':
            input_id = torch.cat((torch.tensor([self.pred_modes[mode]]), input_id))
        elif mode == '<|r2l_pred|>':
            input_id = torch.cat((torch.tensor([self.pred_modes[mode]]), torch.flip(input_id, dims=[0])))
        return input_id[:-1], mode

    def __getitem__(self, idx):
        is_single_item = isinstance(idx, int)
        indices = [idx] if is_single_item else list(idx)
        inputs = {
            'input_ids': torch.tensor([self.input_ids[i] for i in indices]),
            'attention_mask': torch.tensor([self.attention_mask[i] for i in indices])
        }

        # end_time2 = time.perf_counter()

        # This for loop is of acceptable speed; it isn't the main speed bottleneck.
        raw_embeddings = self.exec_batch_embeddings_inference(inputs['input_ids'])
        embeddings = []
        for _id, input_id, raw_embed in zip(indices, inputs['input_ids'], raw_embeddings):
            input_id, mode = self.add_prediction_mode(input_id)
            # new_embeds = self.fast_sliding_window_embedding(input_id.unsqueeze(0), self.window_size, 1, mode).to(self.embed_model.device)
            new_embeds = self.get_embeddings(_id, mode, raw_embed=raw_embed)
            embeddings.append(new_embeds)
        embeddings = torch.stack(embeddings, dim=0).to(self.embed_model.device)

        # end_time3 = time.perf_counter()
        # print(f"Execution time 3: {(end_time3 - end_time2):.6f} seconds")

        outputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": embeddings.to('cpu')
        }
        if is_single_item:
            return {key: value[0] for key, value in outputs.items()}
        return outputs

def tokenize(examples):
    '''Tokenization for default AR language modeling'''
    mapping = tokenizer(examples["text"], padding="max_length", max_length=args.model_max_length, truncation=True)
    mapping['labels'] = [[_id if _id != tokenizer.eos_token_id else -100 for _id in _ids] for _ids in mapping['input_ids']]
    # TO DO: Gemini said the above line is wrong (model won't learn the EOS token). Check in the future.
    return mapping

def get_dataset(args, tokenizer=None, embedding_model=None):

    token_folder = f"{args.pretraining_tokens}M"
    data_dir = Path(f"./pretraining_data/{token_folder}")
    val_path = Path("./pretraining_data/val_shard_00000000_processed.jsonl")

    missing_train = not data_dir.is_dir()
    missing_val = not val_path.exists()
    if missing_train or missing_val:
        patterns = []
        if missing_train:
            patterns.append(f"{token_folder}/*")
        if missing_val:
            patterns.append("val_shard_*")
        snapshot_download(
            repo_id="michaelchenkj/DCLM-pretraining-dataset",
            repo_type="dataset",
            local_dir="./pretraining_data",
            allow_patterns=patterns,
        )

    train_json, eval_json = [], []
    for i in range(args.training_files_no):
        num_str = f"{i:08}"
        with open(data_dir / f"shard_{num_str}_processed.jsonl", "r") as file:
            for line in file:
                train_json.append(json.loads(line.strip()))
    with open(val_path, "r") as file:
        for line in file:
            eval_json.append(json.loads(line.strip()))
    train_json = train_json[:args.train_max_samples]
    eval_json = eval_json[:args.eval_max_samples]

    if args.model_type == 'default' or args.model_type == 'self-concept':
        # The <|endoftext|> token (151643) is auto added in the mapping
        train_dataset = Dataset.from_dict({'text': [s['text'] for s in train_json]})
        # 1. Tokenize (remove columns to ensure we don't carry over raw text)
        train_dataset = train_dataset.map(tokenize_raw, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer})
        # 2. Pack (Group)
        train_dataset = train_dataset.map(group_texts, batched=True, batch_size=1000, fn_kwargs={"args": args})
        # Batch size here determines how many examples are concatenated before splitting.
        # 1000 is usually a good balance for memory vs efficiency.
        eval_dataset = Dataset.from_dict({'text': [s['text'] for s in eval_json]})
        eval_dataset = eval_dataset.map(tokenize_raw, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer})
        eval_dataset = eval_dataset.map(group_texts, batched=True, batch_size=1000, fn_kwargs={"args": args})
    elif args.model_type == 'concept':
        train_dataset = ConceptDataset(
            texts=[s['text'] for s in train_json], tokenizer=tokenizer,
            embed_model=embedding_model, args=args
        )
        eval_dataset = ConceptDataset(
            texts=[s['text'] for s in eval_json], tokenizer=tokenizer,
            embed_model=embedding_model, args=args
        )
        # eval_dataset = copy.copy(train_dataset)
        # eval_dataset.input_ids = eval_dataset.input_ids[:int(len(train_dataset) / 20)]
    
    return train_dataset, eval_dataset

def calculate_no_of_tokens(train_json, tokenizer):
    all_texts = '\n'.join([sample['text'] for sample in train_json])
    return f"No. of tokens in the dataset: {len(tokenizer(all_texts)['input_ids'])}"
# calculate_no_of_tokens(train_json, tokenizer)

if __name__ == "__main__":
    pass