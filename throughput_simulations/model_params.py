from dataclasses import dataclass

@dataclass
class ModelParams:
    """Parameters for a transformer-based LLM"""
    name: str = "llama-3-3-70b"
    hidden_size: int = 8192
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    intermediate_size: int = 28672
    vocab_size: int = 128256
    dtype_bytes: int = 2  # bfloat16 = 2 bytes
    
    def __post_init__(self):
        # If using preset models, update parameters accordingly
        if self.name.lower() == "llama-3-1-8b":
            self.hidden_size = 4096
            self.num_hidden_layers = 32
            self.num_attention_heads = 32
            self.num_key_value_heads = 8
            self.intermediate_size = 14336
            self.vocab_size = 128256
        elif self.name.lower() == "llama-3-3-70b":
            self.hidden_size = 8192
            self.num_hidden_layers = 80
            self.num_attention_heads = 64
            self.num_key_value_heads = 8
            self.intermediate_size = 28672
            self.vocab_size = 128256
            
        # Calculate derived values
        self.head_size = self.hidden_size // self.num_attention_heads
        
        # Calculate total parameters
        self.total_params = self._calculate_total_params()
        
        # Calculate model size in bytes
        self.model_size_bytes = self.total_params * self.dtype_bytes
    
    def _calculate_total_params(self):
        """Calculate the total number of parameters in the model"""
        # Parameters per transformer block - simplified from the llama architecture
        block_params = (
            # Attention
            self.hidden_size * self.hidden_size +  # q projection
            self.hidden_size * self.hidden_size // 8 +  # k projection (8x smaller for LLama)
            self.hidden_size * self.hidden_size // 8 +  # v projection (8x smaller for LLama)
            self.hidden_size * self.hidden_size +  # o projection
            
            # MLP
            self.hidden_size * self.intermediate_size +  # gate projection
            self.hidden_size * self.intermediate_size +  # up projection
            self.intermediate_size * self.hidden_size +  # down projection
            
            # LayerNorms
            2 * self.hidden_size
        )
        
        # Embedding and head
        embedding_params = self.vocab_size * self.hidden_size
        head_params = self.hidden_size * self.vocab_size
        
        # Final layer norm
        final_norm_params = self.hidden_size
        
        # Total
        total = embedding_params + (self.num_hidden_layers * block_params) + final_norm_params + head_params
        
        return total