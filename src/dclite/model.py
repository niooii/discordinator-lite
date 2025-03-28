from enum import Enum
import gc
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from huggingface_hub import login
from tqdm import tqdm
from typing import Optional, Dict, Any
import os
import json

logged_in = False

class ModelType(Enum):
    GPT2 = ("gpt2", "causal")
    GPT2_MEDIUM = ("gpt2-medium", "causal")
    GPT2_LARGE = ("gpt2-large", "causal")
    GPT_NEO = ("EleutherAI/gpt-neo-1.3B", "causal")
    OPT = ("facebook/opt-1.3b", "causal")
    BLOOM = ("bigscience/bloom-1b1", "causal")
    LLAMA2 = ("meta-llama/Llama-2-7b-chat-hf", "causal")
    LLAMA3 = ("meta-llama/Llama-3.2-1B", "causal")

class ChatModel:
    def __init__(
        self, 
        model_type: ModelType,
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        # global logged_in
        # if not logged_in:
        #     login()
        #     logged_in = True

        self.model_type = model_type
        self.model_name, self.architecture_type = model_type.value
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_kwargs = model_kwargs or {}
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Loading model {self.model_name} on {self.device}...")
        
        quant_config = BitsAndBytesConfig(
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_use_double_quant=True,
            # llm_int8_enable_fp32_cpu_offload=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto", 
            **self.model_kwargs
        ).to(self.device)
        
        # for loading from disk later
        self.config = {
            "model_type": model_type.name,
            "architecture_type": self.architecture_type,
            "model_name": self.model_name
        }

    def cleanup(self):
        try:
            if hasattr(self, 'model'):
                self.model.cpu()
                del self.model
                self.__dict__.pop('model', None)
            
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                self.__dict__.pop('tokenizer', None)
            
            gc.collect()
            
            if self.device == "cuda" or self.device.startswith("cuda:"):
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            
            print(f"Cleaned up resources for {self.model_name}")
        except Exception as e:
            print(f"You're cooked {e}")
        
    def gen_text(
        self,
        input: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> list[str]:
        inputs = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts
        
    def finetune(
        self,
        dataset,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        warmup_steps: int = 0,
        save_steps: int = 0,
        output_dir: Optional[str] = None,
        **kwargs
    ):
        print("hehehaw")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True  # no incomplete batches
        )
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            **kwargs.get("optimizer_kwargs", {})
        )
        
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            try:
                for batch in tqdm(dataloader, desc=f"Finetuning [epoch {epoch+1}/{epochs}]"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    epoch_loss += loss.item()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                if save_steps > 0 and epoch % save_steps == 0 and output_dir:
                    self.save(os.path.join(output_dir, f"checkpoint-{epoch}"))
                    print(f"Saved checkpoint {epoch}")
                
                print(epoch_loss)
                print(len(dataloader))
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
            except KeyboardInterrupt:
                print("Training interrupted..")
                break
            finally:
                if save_steps == 0:
                    self.save(output_dir=output_dir)
                else:
                    self.save(os.path.join(output_dir, f"final"))
            
        self.model.eval()
        
    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        # save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # save class info
        custom_config = {
            "custom_model_type": self.model_type.name,
            "architecture_type": self.architecture_type,
            "model_name": self.model_name
        }
        with open(os.path.join(output_dir, "class_cfg.json"), "w") as f:
            json.dump(custom_config, f)
            
        # the model"s own config
        self.model.config.save_pretrained(output_dir)
            
    @classmethod
    def from_pretrained(cls, model_dir: str, device: Optional[str] = None) -> "ChatModel":
        with open(os.path.join(model_dir, "class_cfg.json"), "r") as f:
            class_cfg = json.load(f)
            
        model_type = ModelType[class_cfg["custom_model_type"]]

        instance = cls(model_type, device=device)
        instance.model = AutoModelForCausalLM.from_pretrained(model_dir).to(instance.device)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        return instance
