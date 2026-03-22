"""
Artifex Assistant V5 — Code Generation Mode.
Specialized configuration for code-focused models and prompts.
"""

import os

# Known code-specialized models with their HuggingFace repo IDs
CODE_MODELS = {
    "starcoder2-7b": "bigcode/starcoder2-7b",
    "starcoder2-3b": "bigcode/starcoder2-3b",
    "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
}

# Code-specific stop tokens (model output should stop here)
CODE_STOP_TOKENS = [
    "\n```\n",
    "\n---\n",
    "\nHuman:",
    "\nUser:",
]

CODE_SYSTEM_PROMPT = """You are an expert programming assistant. Write clean, efficient, well-documented code.

RULES:
- Always use proper language-specific conventions and idioms
- Include brief docstrings/comments for non-trivial functions
- Handle edge cases and errors appropriately
- When modifying existing code, show only the changed portions
- Explain your reasoning briefly before code blocks
- If the request is ambiguous, ask for clarification
- Test your logic mentally before presenting code
- Prefer standard library solutions over third-party dependencies when reasonable

OUTPUT FORMAT:
- Use fenced code blocks with language identifiers: ```python, ```javascript, etc.
- For file modifications, specify the file path before the code block
- Keep explanations concise — the code should speak for itself
"""


def detect_code_model(model_path):
    """Check if a loaded model is a code-specialized model.

    Args:
        model_path: Path to the model directory or HF repo ID

    Returns:
        True if the model appears to be code-specialized
    """
    name = os.path.basename(model_path or "").lower()
    code_indicators = [
        "coder", "code", "starcoder", "codellama", "deepseek-coder",
        "codegen", "incoder", "replit-code", "wizardcoder",
    ]
    return any(indicator in name for indicator in code_indicators)


def code_generation_params():
    """Get optimal generation parameters for code tasks.

    Returns:
        dict of generation kwargs optimized for code output
    """
    return {
        "temperature": 0.2,      # Low temp for precise code
        "max_tokens": 4096,      # Code can be verbose
        "top_p": 0.95,
        "repetition_penalty": 1.05,
    }


def build_code_prompt(system_info="", cwd="", language_hint=""):
    """Build the system prompt for code generation mode.

    Args:
        system_info: System/environment info string
        cwd: Current working directory
        language_hint: Primary programming language (if known)

    Returns:
        Complete system prompt string
    """
    prompt = CODE_SYSTEM_PROMPT

    if language_hint:
        prompt += f"\nPrimary language: {language_hint}\n"

    if system_info:
        prompt += f"\nENVIRONMENT:\n{system_info}\n"

    if cwd:
        prompt += f"\nCWD: {cwd}\n"

    return prompt
