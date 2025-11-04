"""
Token Tracking Utility for Google Gemini with LangSmith

Since Google Gemini doesn't automatically report token usage to LangSmith,
this module provides utilities to estimate and track token usage manually.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text (rough approximation)
    
    Rule of thumb:
    - English: ~1 token per 4 characters
    - Or ~0.75 tokens per word
    
    Args:
        text (str): Text to estimate tokens for
        
    Returns:
        int: Estimated token count
    """
    if not text:
        return 0
    
    # Count words as a more accurate estimate
    words = len(text.split())
    
    # Average: 0.75 tokens per word for English
    # Add 20% buffer for formatting, JSON structure, etc.
    estimated_tokens = int(words * 0.75 * 1.2)
    
    return max(estimated_tokens, 1)


def log_token_usage(node_name: str, prompt: str, response: str) -> dict:
    """
    Log token usage for a Gemini API call
    
    Args:
        node_name (str): Name of the node making the call
        prompt (str): The prompt sent to Gemini
        response (str): The response from Gemini
        
    Returns:
        dict: Token usage statistics
    """
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(response)
    total_tokens = input_tokens + output_tokens
    
    token_info = {
        "node": node_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_chars": len(prompt),
        "output_chars": len(response)
    }
    
    logger.info(
        f"🎯 {node_name} - Tokens: {total_tokens} "
        f"(in: {input_tokens}, out: {output_tokens})"
    )
    
    return token_info


def estimate_cost(total_tokens: int, model: str = "gemini-2.5-flash") -> float:
    """
    Estimate cost for Gemini API usage
    
    Pricing (as of 2024):
    - Gemini 2.5 Flash: $0.075 per 1M input tokens, $0.30 per 1M output tokens
    - Gemini Pro: $0.50 per 1M input tokens, $1.50 per 1M output tokens
    
    Args:
        total_tokens (int): Total token count
        model (str): Model name
        
    Returns:
        float: Estimated cost in USD
    """
    # Simplified cost calculation (using average rate)
    if "flash" in model.lower():
        # Average rate for flash: ~$0.15 per 1M tokens
        cost_per_1m = 0.15
    else:
        # Average rate for pro: ~$1.00 per 1M tokens
        cost_per_1m = 1.00
    
    cost = (total_tokens / 1_000_000) * cost_per_1m
    
    return round(cost, 6)


class TokenTracker:
    """
    Context manager for tracking token usage across a workflow
    """
    
    def __init__(self, workflow_name: str = "ConceptMapping"):
        self.workflow_name = workflow_name
        self.node_tokens = {}
        self.total_tokens = 0
        
    def add_node(self, node_name: str, token_info: dict):
        """Add token info for a node"""
        self.node_tokens[node_name] = token_info
        self.total_tokens += token_info["total_tokens"]
        
    def get_summary(self) -> dict:
        """Get token usage summary"""
        return {
            "workflow": self.workflow_name,
            "total_tokens": self.total_tokens,
            "total_cost": estimate_cost(self.total_tokens),
            "nodes": self.node_tokens
        }
    
    def log_summary(self):
        """Log token usage summary"""
        summary = self.get_summary()
        
        logger.info("=" * 60)
        logger.info(f"📊 Token Usage Summary - {self.workflow_name}")
        logger.info("=" * 60)
        
        for node_name, info in self.node_tokens.items():
            logger.info(
                f"  {node_name}: {info['total_tokens']} tokens "
                f"(in: {info['input_tokens']}, out: {info['output_tokens']})"
            )
        
        logger.info("-" * 60)
        logger.info(f"  TOTAL: {self.total_tokens} tokens")
        logger.info(f"  ESTIMATED COST: ${summary['total_cost']:.4f}")
        logger.info("=" * 60)


# Global tracker instance
_global_tracker: Optional[TokenTracker] = None


def get_tracker() -> TokenTracker:
    """Get or create global token tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TokenTracker()
    return _global_tracker


def reset_tracker():
    """Reset global token tracker"""
    global _global_tracker
    _global_tracker = TokenTracker()
