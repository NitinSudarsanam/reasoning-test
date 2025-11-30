"""Generator LLM for producing multi-stage reasoning and code."""

import torch
import re
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMGenerator:
    """Generator model that produces multi-stage reasoning outputs and code."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize generator from HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading generator model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device
        )
        self.model.eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_stage_output(
        self,
        problem: str,
        previous_stages: List[str],
        stage_id: int,
        prompt_template: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate output for a specific reasoning stage.
        
        Args:
            problem: Problem description
            previous_stages: Outputs from previous stages
            stage_id: Current stage ID (1-5)
            prompt_template: Template for this stage
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated output for this stage
        """
        # Format previous stages
        previous_text = "\n\n".join([
            f"Stage {i+1}:\n{stage}" 
            for i, stage in enumerate(previous_stages)
        ])
        
        # Format prompt
        prompt = prompt_template.format(
            problem=problem,
            previous_stages=previous_text if previous_stages else "None"
        )
        
        # Generate
        output = self._generate(prompt, max_new_tokens, temperature, top_p)
        
        # Sanitize output
        output = self._sanitize_output(output)
        
        return output
    
    def generate_code(
        self,
        problem: str,
        reasoning_chain: List[str],
        prompt_template: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate final executable code (stage 5).
        
        Args:
            problem: Problem description
            reasoning_chain: All previous reasoning stages
            prompt_template: Template for code generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated Python code
        """
        output = self.generate_stage_output(
            problem=problem,
            previous_stages=reasoning_chain,
            stage_id=5,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Extract code from markdown if present
        output = self._extract_code_from_markdown(output)
        
        return output
    
    def get_log_probs(self, prompt: str, output: str) -> torch.Tensor:
        """Get log probabilities for RL training.
        
        Args:
            prompt: Input prompt
            output: Generated output
            
        Returns:
            Log probabilities tensor
        """
        # Tokenize
        full_text = prompt + output
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get log probs for generated tokens only
        prompt_len = prompt_inputs.input_ids.shape[1]
        generated_logits = logits[0, prompt_len-1:-1, :]
        generated_tokens = inputs.input_ids[0, prompt_len:]
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(generated_logits, dim=-1)
        token_log_probs = log_probs.gather(1, generated_tokens.unsqueeze(1)).squeeze(1)
        
        return token_log_probs
    
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        """Internal generation method.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text
    
    def _sanitize_output(self, output: str) -> str:
        """Sanitize generated output.
        
        Args:
            output: Raw generated output
            
        Returns:
            Sanitized output
        """
        # Remove excessive whitespace
        output = output.strip()
        
        # Remove incomplete sentences at the end
        if output and not output[-1] in '.!?\n':
            # Find last complete sentence
            last_period = max(
                output.rfind('.'),
                output.rfind('!'),
                output.rfind('?'),
                output.rfind('\n')
            )
            if last_period > len(output) // 2:  # Only if we have substantial content
                output = output[:last_period + 1]
        
        return output
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks.
        
        Args:
            text: Text potentially containing markdown code blocks
            
        Returns:
            Extracted code or original text
        """
        # Look for ```python ... ``` blocks
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Look for ``` ... ``` blocks
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return text.strip()
