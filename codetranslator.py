# -*- coding: utf-8 -*-

import os
import re
import logging
import shutil
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, TypedDict, Tuple, Union, Set
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from argparse import ArgumentParser

# Third-party imports
import tiktoken
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from dotenv import load_dotenv

# LLM APIs
from openai import OpenAI
from anthropic import Anthropic

###############################################################################
# Configuration and Types
###############################################################################

@dataclass
class LLMResponse:
    """Standardized response from LLM providers"""
    content: str
    tokens_used: int
    model: str
    provider: str

class TranslationResult(TypedDict):
    """Type definition for translation results"""
    final_code: str
    success: bool
    attempts: List[Dict[str, Any]]
    message: str
    metrics: Dict[str, Any]

@dataclass
class TranslationAttempt:
    """Record of a translation attempt"""
    attempt_number: int
    timestamp: str
    source_code: str
    translated_code: str
    errors: List[str]
    success: bool
    tokens_used: int
    execution_output: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp,
            "source_code": self.source_code,
            "translated_code": self.translated_code,
            "errors": self.errors,
            "success": self.success,
            "tokens_used": self.tokens_used,
            "execution_output": self.execution_output
        }

###############################################################################
# Error Handling and Metrics
###############################################################################

class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.console = Console()
        self.error_history: List[Tuple[datetime, str, str]] = []
        
    def handle_error(self, error: Exception, context: str = "") -> str:
        """
        Process and format system errors
        
        Args:
            error: The caught exception
            context: Additional context about the error
            
        Returns:
            Formatted error message
        """
        error_msg = f"{type(error).__name__}: {str(error)}"
        if context:
            error_msg = f"{context}: {error_msg}"
            
        self.error_history.append((datetime.now(), context, str(error)))
        return error_msg
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error types and frequencies"""
        error_counts: Dict[str, int] = {}
        for _, _, error in self.error_history:
            error_type = error.split(':')[0]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts

class MetricsCollector:
    """Execution metrics collection system"""
    
    def __init__(self):
        self.total_tokens = 0
        self.successful_translations = 0
        self.failed_translations = 0
        self.total_attempts = 0
        self.errors: List[str] = []
        self.start_time = datetime.now()
        self.language_metrics: Dict[str, Dict[str, int]] = {}
        
    def update_metrics(self, 
                      tokens: int, 
                      success: bool, 
                      error: Optional[str] = None,
                      source_lang: Optional[str] = None,
                      target_lang: Optional[str] = None):
        """
        Update execution metrics
        
        Args:
            tokens: Number of tokens used
            success: Whether the operation was successful
            error: Error message if operation failed
            source_lang: Source programming language
            target_lang: Target programming language
        """
        self.total_tokens += tokens
        self.total_attempts += 1
        
        if success:
            self.successful_translations += 1
        else:
            self.failed_translations += 1
            if error:
                self.errors.append(error)
        
        if source_lang and target_lang:
            lang_pair = f"{source_lang}->{target_lang}"
            if lang_pair not in self.language_metrics:
                self.language_metrics[lang_pair] = {
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                    "tokens": 0
                }
            
            self.language_metrics[lang_pair]["attempts"] += 1
            self.language_metrics[lang_pair]["tokens"] += tokens
            if success:
                self.language_metrics[lang_pair]["successes"] += 1
            else:
                self.language_metrics[lang_pair]["failures"] += 1
    
    def get_execution_time(self) -> float:
        """Get total execution time in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get complete metrics report"""
        return {
            "total_tokens": self.total_tokens,
            "successful_translations": self.successful_translations,
            "failed_translations": self.failed_translations,
            "total_attempts": self.total_attempts,
            "error_count": len(self.errors),
            "execution_time": self.get_execution_time(),
            "language_metrics": self.language_metrics
        }
    
    def display_metrics(self, console: Console):
        """Display metrics in a formatted table"""
        table = Table(title="Translation Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        metrics = self.get_metrics()
        table.add_row("Total Tokens Used", str(metrics["total_tokens"]))
        table.add_row("Successful Translations", str(metrics["successful_translations"]))
        table.add_row("Failed Translations", str(metrics["failed_translations"]))
        table.add_row("Total Attempts", str(metrics["total_attempts"]))
        table.add_row("Total Errors", str(metrics["error_count"]))
        table.add_row("Execution Time (s)", f"{metrics['execution_time']:.2f}")
        
        console.print(table)
        
        if self.language_metrics:
            lang_table = Table(title="Language Pair Metrics")
            lang_table.add_column("Language Pair", style="cyan")
            lang_table.add_column("Attempts", style="magenta")
            lang_table.add_column("Success Rate", style="green")
            lang_table.add_column("Avg Tokens", style="yellow")
            
            for pair, stats in self.language_metrics.items():
                success_rate = (stats["successes"] / stats["attempts"]) * 100 if stats["attempts"] else 0
                avg_tokens = stats["tokens"] / stats["attempts"] if stats["attempts"] else 0
                lang_table.add_row(
                    pair,
                    str(stats["attempts"]),
                    f"{success_rate:.1f}%",
                    f"{avg_tokens:.1f}"
                )
            
            console.print(lang_table)

###############################################################################
# LLM Provider System
###############################################################################

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def initialize_client(self) -> None:
        pass
    
    @abstractmethod
    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        model: Optional[str] = None
    ) -> LLMResponse:
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self):
        self.client = None
        self.default_model = "gpt-4"
    
    def initialize_client(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)
    
    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        model: Optional[str] = None
    ) -> LLMResponse:
        if not self.client:
            self.initialize_client()
            
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=model or self.default_model,
            provider="openai"
        )

class AnthropicProvider(LLMProvider):
    """Anthropic API provider implementation"""
    
    def __init__(self):
        self.client = None
        self.default_model = "claude-3-opus-20240229"
    
    def initialize_client(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = Anthropic(api_key=api_key)
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert messages to Anthropic format (simplified placeholder example)"""
        converted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            # The real Anthropic client usage may differ significantly,
            # but this is a placeholder to illustrate the idea.
            if role == "system":
                converted.append({"role": "assistant", "content": content})
            elif role in ["user", "assistant"]:
                converted.append({"role": role, "content": content})
        return converted
    
    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        model: Optional[str] = None
    ) -> LLMResponse:
        if not self.client:
            self.initialize_client()
        
        anthropic_messages = self._convert_messages(messages)
        # Placeholder call; real usage may differ
        response = self.client.messages.create(
            model=model or self.default_model,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Simplified placeholder example to match structure
        return LLMResponse(
            content=response.content[0].text,
            tokens_used=response.usage.output_tokens + response.usage.input_tokens,
            model=model or self.default_model,
            provider="anthropic"
        )

###############################################################################
# Enhanced Translation Agent
###############################################################################

class EnhancedTranslationAgent:
    """Enhanced multi-language translation agent with advanced features"""
    
    # Language configurations
    LANGUAGE_CONFIG = {
        "Python": {
            "extension": ".py",
            "run_cmd": ["python", "-u"],
            "timeout": 300,
            "compile": None,
            "validator": ["python", "-m", "py_compile"],
            "formatter": ["black"],
            "libraries": ["numpy", "pandas", "scipy"]
        },
        "R": {
            "extension": ".r",
            "run_cmd": ["Rscript", "--vanilla"],
            "timeout": 450,
            "compile": None,
            "validator": ["R", "CMD", "check"],
            "formatter": ["styler::style_text"],
            "libraries": ["tidyverse", "data.table"]
        },
        "Julia": {
            "extension": ".jl",
            "run_cmd": ["julia"],
            "timeout": 600,
            "compile": None,
            "validator": ["julia", "--check-bounds=yes"],
            "formatter": ["using JuliaFormatter; format_text"],
            "libraries": ["DataFrames", "Statistics"]
        },
        "C++": {
            "extension": ".cpp",
            "run_cmd": ["./a.out"],
            "timeout": 200,
            "compile": ["g++", "-std=c++17", "-Wall", "-Wextra", "-O2"],
            "validator": ["cppcheck", "--enable=all"],
            "formatter": ["clang-format", "-style=google"],
            "libraries": ["iostream", "vector", "algorithm"]
        },
        "Rcpp": {
            "extension": ".cpp",
            "run_cmd": ["R", "CMD", "SHLIB"],
            "timeout": 400,
            "compile": ["g++", "-std=c++17", "-I/usr/share/R/include"],
            "validator": ["R", "CMD", "check"],
            "formatter": ["clang-format", "-style=google"],
            "libraries": ["Rcpp", "RcppArmadillo"]
        }
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        max_attempts: int = 5,
        temperature: float = 0,
        max_tokens: int = 4096,
        working_dir: str = "translation_workspace",
        trace: bool = False
    ):
        """
        Initialize the EnhancedTranslationAgent.

        Args:
            provider (str): Name of the LLM provider. Either 'openai' or 'anthropic'.
            model (Optional[str]): Model name to use for the LLM.
            max_attempts (int): Maximum number of correction/validation attempts.
            temperature (float): LLM temperature for generation.
            max_tokens (int): Maximum tokens for LLM response.
            working_dir (str): Directory where final outputs and log will be stored.
            trace (bool): Whether to show debug-level logs.
        """
        load_dotenv()
        
        # Initialize providers
        self.providers = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider()
        }
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not supported")
        
        # Setup core components
        self.current_provider = self.providers[provider]
        self.current_provider.initialize_client()
        self.error_handler = ErrorHandler()
        self.metrics_collector = MetricsCollector()
        
        # Configuration
        self.model = model
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.trace = trace
        self.seen_errors: Set[str] = set()
        
        # Keep track of all validation/correction attempts
        self.attempt_records: List[TranslationAttempt] = []

        # Prepare console
        self.console = Console()
        
        # Working directory for final artifacts
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporary directory (will be cleaned up at the end)
        self.temp_dir = self.working_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging in the output-dir (single log file)
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging system to save a single log file in working_dir."""
        log_format = "%(asctime)s [%(levelname)s] %(message)s"
        log_file = self.working_dir / "translation.log"  # single consistent log
        
        logging.basicConfig(
            level=logging.INFO if self.trace else logging.WARNING,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, mode='w'),  # overwrite each run
                RichHandler(rich_tracebacks=True)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging initialized.")
    
    def _validate_languages(self, source_lang: str, target_lang: str) -> bool:
        """Validate that both languages are supported"""
        return (source_lang in self.LANGUAGE_CONFIG and 
                target_lang in self.LANGUAGE_CONFIG)
    
    def _build_translation_messages(
        self,
        source_code: str,
        source_lang: str,
        target_lang: str
    ) -> List[Dict[str, str]]:
        """
        Build prompts for the translation request.
        We explicitly ask the LLM to return the code in a single code block,
        with no extra commentary or text after the triple backticks.
        """
        source_config = self.LANGUAGE_CONFIG[source_lang]
        target_config = self.LANGUAGE_CONFIG[target_lang]
        
        system_message = (
            f"You are a code translation expert specialized in converting {source_lang} to {target_lang}.\n"
            "Your response MUST:\n"
            "1) Return only the translated code in a single code block.\n"
            "2) No commentary or text after triple backticks.\n"
            "3) Keep the same functionality as the original.\n"
            "4) Use only standard best practices for the target language.\n"
            "5) ALWAYS Check if needed imports/packages/libraries exists in system before include or calling them. If not exists, install and use conditional imports.\n"
            "6) Preserve comments/docstrings from the original code if relevant.\n"
            "7) If there's nothing to preserve, do not add extraneous text.\n"
            "8) Return it as: ```<language>\n<translated code>\n```\n"
        )

        translation_prompt = (
            f"Translate the following {source_lang} code to {target_lang}, "
            "returning only one code block with no extra text or explanation:\n"
            f"```{source_lang}\n{source_code}\n```\n"
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": translation_prompt}
        ]
    
    def _extract_code(self, content: str, language: str) -> str:
        """
        Extract code blocks from LLM response, preferring a single code block.
        We'll look for the code block that matches the pattern:
        
            ```<language>
            ... code ...
            ```
        
        If multiple blocks exist, we pick the largest by content length.
        If no match, we fall back to the entire content.
        Finally, we remove anything after the closing triple backticks
        to avoid leftover text.
        """
        # Pattern that captures code in triple backticks, optionally specifying language
        # The language group is optional in case the model doesn't specify it
        pattern = rf"```(?:{language.lower()}|{language})?\s*(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

        if not matches:
            # Return the entire content if no matches found
            return content.strip()

        # If there are multiple code blocks, pick the largest one
        code_block = max(matches, key=len).strip()
        return code_block

    def _validate_code(self, code: str, language: str) -> Tuple[bool, str]:
        """
        Validate code using language-specific tools.
        Returns (is_valid, error_message).
        """
        config = self.LANGUAGE_CONFIG[language]
        if not config["validator"]:
            return True, ""
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=config["extension"],
            dir=self.temp_dir,
            delete=False,
            encoding='utf-8'
        ) as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name
        
        try:
            process = subprocess.run(
                config["validator"] + [temp_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            return (process.returncode == 0,
                    (process.stderr.strip() or process.stdout.strip()))
        except subprocess.TimeoutExpired:
            return False, "Validation timeout"
        except Exception as e:
            return False, str(e)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def _format_code(self, code: str, language: str) -> str:
        """
        Format code using language-specific formatters.
        If formatting fails for any reason, return the original code.
        """
        config = self.LANGUAGE_CONFIG[language]
        if not config["formatter"]:
            return code
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=config["extension"],
            dir=self.temp_dir,
            delete=False,
            encoding='utf-8'
        ) as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name
        
        try:
            process = subprocess.run(
                config["formatter"] + [temp_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if process.returncode == 0:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            return code
        except Exception as e:
            self.logger.warning(f"Formatting failed: {str(e)}")
            return code
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def _execute_code(self, code: str, language: str) -> Tuple[bool, str]:
        """
        Execute code with specific configurations for R.
        Args:
            code (str): The code to execute
            language (str): Programming language
        Returns:
            Tuple[bool, str]: (success, output_or_error)
        """
        config = self.LANGUAGE_CONFIG[language]
        
        # Special handling for R
        if language.lower() == 'r':
            # Create R initialization commands
            r_init_commands = """
            # Set default user library
            if (!dir.exists(Sys.getenv("R_LIBS_USER"))) {
                dir.create(Sys.getenv("R_LIBS_USER"), recursive=TRUE)
            }
            .libPaths(Sys.getenv("R_LIBS_USER"))
            
            # Set default CRAN mirror
            options(repos=c(CRAN="https://cloud.r-project.org"))
            
            # Original code follows
            """
            # Combine initialization commands with user code
            code = r_init_commands + "\n" + code
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=config["extension"],
            dir=self.temp_dir,
            delete=False,
            encoding='utf-8'
        ) as temp_file:
            temp_file.write(code)
            temp_path = Path(temp_file.name)
        
        try:
            if config["compile"]:
                compile_cmd = config["compile"] + [str(temp_path)]
                compile_process = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if compile_process.returncode != 0:
                    return (
                        False,
                        f"Compilation failed:\n{compile_process.stderr.strip() or compile_process.stdout.strip()}"
                    )
            
            # Execution step
            if language == "C++":
                exe_path = temp_path.with_suffix('')
                run_cmd = [str(exe_path)]
            else:
                run_cmd = config["run_cmd"] + [str(temp_path)]
            
            # Set R specific environment variables if needed
            env = os.environ.copy()
            if language.lower() == 'r':
                env['R_LIBS_USER'] = os.path.expanduser('~/R/library')
            
            process = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=config["timeout"],
                env=env
            )
            
            success = (process.returncode == 0)
            output = process.stdout.strip() if success else process.stderr.strip()
            return success, output
        except subprocess.TimeoutExpired:
            return False, f"Execution timeout after {config['timeout']}s"
        except Exception as e:
            return False, str(e)
        finally:
            temp_path.unlink(missing_ok=True)
            if language == "C++" and config["compile"]:
                exe_path = temp_path.with_suffix('')
                exe_path.unlink(missing_ok=True)

    # def _execute_code(self, code: str, language: str) -> Tuple[bool, str]:
    #     """
    #     Execute code to verify it runs.
    #     Returns (success, output_or_error).
    #     """
    #     config = self.LANGUAGE_CONFIG[language]
        
    #     with tempfile.NamedTemporaryFile(
    #         mode='w',
    #         suffix=config["extension"],
    #         dir=self.temp_dir,
    #         delete=False,
    #         encoding='utf-8'
    #     ) as temp_file:
    #         temp_file.write(code)
    #         temp_path = Path(temp_file.name)
        
    #     try:
    #         if config["compile"]:
    #             compile_cmd = config["compile"] + [str(temp_path)]
    #             compile_process = subprocess.run(
    #                 compile_cmd,
    #                 capture_output=True,
    #                 text=True,
    #                 timeout=300
    #             )
                
    #             if compile_process.returncode != 0:
    #                 return (
    #                     False,
    #                     f"Compilation failed:\n{compile_process.stderr.strip() or compile_process.stdout.strip()}"
    #                 )
            
    #         # Execution step
    #         if language == "C++":
    #             exe_path = temp_path.with_suffix('')
    #             run_cmd = [str(exe_path)]
    #         else:
    #             run_cmd = config["run_cmd"] + [str(temp_path)]
            
    #         process = subprocess.run(
    #             run_cmd,
    #             capture_output=True,
    #             text=True,
    #             timeout=config["timeout"]
    #         )
            
    #         success = (process.returncode == 0)
    #         output = process.stdout.strip() if success else process.stderr.strip()
    #         return success, output
    #     except subprocess.TimeoutExpired:
    #         return False, f"Execution timeout after {config['timeout']}s"
    #     except Exception as e:
    #         return False, str(e)
    #     finally:
    #         temp_path.unlink(missing_ok=True)
    #         if language == "C++" and config["compile"]:
    #             exe_path = temp_path.with_suffix('')
    #             exe_path.unlink(missing_ok=True)
    
    def _build_correction_messages(
        self,
        code: str,
        error: str,
        target_lang: str,
        original_code: str,
        source_lang: str
    ) -> List[Dict[str, str]]:
        """
        Build prompts for code correction (LLM-based).
        Again, we request a single code block with no extraneous text.
        """
        system_message = (
            f"You are a {target_lang} expert fixing code translation issues. "
            "Respond with ONLY the corrected code in a single code block (no extra commentary)."
        )

        correction_prompt = (
            f"Original {source_lang} code:\n"
            f"```{source_lang}\n{original_code}\n```\n\n"
            f"Translated {target_lang} code with errors:\n"
            f"```{target_lang}\n{code}\n```\n\n"
            f"Error message:\n{error}\n\n"
            f"Fix ALL errors while maintaining functionality.\n"
            "Return only one code block with the corrected code."
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": correction_prompt}
        ]
    
    def _validation_cycle(
        self,
        code: str,
        target_lang: str,
        original_code: str,
        source_lang: str
    ) -> str:
        """
        Run validation and LLM-based correction cycles up to self.max_attempts.
        """
        attempt = 0
        while attempt < self.max_attempts:
            attempt += 1
            
            valid, error_msg = self._validate_code(code, target_lang)

            self.attempt_records.append(
                TranslationAttempt(
                    attempt_number=attempt,
                    timestamp=datetime.now().isoformat(),
                    source_code=original_code,
                    translated_code=code,
                    errors=[error_msg] if not valid else [],
                    success=valid,
                    tokens_used=0,
                    execution_output=""
                )
            )

            if not valid:
                # Check for recurring error
                if hash(error_msg) in self.seen_errors:
                    self.logger.warning("Recurring error detected, breaking cycle.")
                    break
                self.seen_errors.add(hash(error_msg))
                
                # Request LLM-based correction
                correction_messages = self._build_correction_messages(
                    code,
                    error_msg,
                    target_lang,
                    original_code,
                    source_lang
                )
                
                try:
                    response = self.current_provider.generate_completion(
                        messages=correction_messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        model=self.model
                    )
                    new_code = self._extract_code(response.content, target_lang)
                    code = new_code
                    
                    self.metrics_collector.update_metrics(
                        response.tokens_used,
                        False,
                        error_msg,
                        source_lang,
                        target_lang
                    )
                    
                except Exception as e:
                    self.logger.error(f"Correction failed: {str(e)}")
                    break
            else:
                break
        
        return code
    
    def _build_result(
        self,
        code: str,
        success: bool,
        message: str,
        metrics: Dict[str, Any]
    ) -> TranslationResult:
        """Build standardized translation result."""
        return {
            "final_code": code,
            "success": success,
            "attempts": [a.to_dict() for a in self.attempt_records],
            "message": message,
            "metrics": metrics
        }

    def translate(
        self,
        source_code: str,
        source_lang: str,
        target_lang: str,
        validate: bool = True,
        format_code: bool = True
    ) -> TranslationResult:
        """
        Main translation method with optional validation and formatting.
        
        Args:
            source_code: Code to translate.
            source_lang: Source programming language (must exist in LANGUAGE_CONFIG).
            target_lang: Target programming language (must exist in LANGUAGE_CONFIG).
            validate: Whether to validate the translated code with a compile/check step.
            format_code: Whether to format the translated code using a formatter.
            
        Returns:
            TranslationResult with translation details.
        """
        try:
            # Check if languages are supported
            if not self._validate_languages(source_lang, target_lang):
                raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")
            
            self.logger.info(f"Starting translation from {source_lang} to {target_lang}")
            self.metrics_collector.start_time = datetime.now()
            
            # Build initial translation prompt
            messages = self._build_translation_messages(source_code, source_lang, target_lang)
            
            # Get initial translation from LLM
            response = self.current_provider.generate_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model=self.model
            )
            
            translated_code = self._extract_code(response.content, target_lang)
            self.metrics_collector.update_metrics(
                response.tokens_used,
                True,
                None,
                source_lang,
                target_lang
            )
            
            # Validation cycle
            if validate:
                translated_code = self._validation_cycle(
                    translated_code,
                    target_lang,
                    source_code,
                    source_lang
                )
            
            # Optional formatting
            if format_code:
                translated_code = self._format_code(translated_code, target_lang)
            
            # Final execution test
            success, output = self._execute_code(translated_code, target_lang)
            if not success:
                self.logger.warning(f"Final execution failed: {output}")
            
            final_message = "Translation successful" if success else f"Translation failed: {output}"
            self.logger.info(final_message)
            
            return self._build_result(
                translated_code,
                success,
                final_message,
                self.metrics_collector.get_metrics()
            )
            
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "Translation failed")
            self.logger.error(error_msg)
            return self._build_result(
                source_code,
                False,
                error_msg,
                self.metrics_collector.get_metrics()
            )
        finally:
            self.logger.info("Translation process completed.")

###############################################################################
# codetranslate() CORE FUNCTION
###############################################################################
def codetranslate(
    source_file: str,
    source_lang: str,
    target_lang: str,
    provider: str = "openai",
    model: Optional[str] = None,
    validate: bool = True,
    format_code: bool = True,
    trace: bool = False,
    output_dir: str = "translation_workspace"
) -> Dict[str, Any]:
    """
    Translate a code file from one language to another using the EnhancedTranslationAgent.
    
    Args:
        source_file (str): 
            The path (string) of the source code file that needs to be translated.
        source_lang (str): 
            The source programming language, e.g. 'py', 'python', 'r', 'R', 'jl', 'julia',
            'cpp', 'c++', 'rcpp', 'rc++'.
        target_lang (str): 
            The target programming language, e.g. 'py', 'python', 'r', 'R', 'jl', 'julia',
            'cpp', 'c++', 'rcpp', 'rc++'.
        provider (str, optional): 
            The LLM provider to use for translation ('openai' or 'anthropic'). Default is "openai".
        model (Optional[str], optional): 
            A specific model to be used by the LLM provider. If None, uses the provider's default model.
        validate (bool, optional): 
            Whether to validate (compile/check) the translated code. Default is True.
        format_code (bool, optional): 
            Whether to format the translated code using a formatter (if available for the language). 
            Default is True.
        trace (bool, optional): 
            If True, uses detailed logging (INFO). Otherwise logs only WARN or higher. Default is False.
        output_dir (str, optional): 
            The directory where the final translated code file and the log file will be saved.
            A temporary subdirectory (temp) will be created inside this folder for intermediate files.
            Default is "translation_workspace".
    
    Returns:
        Dict[str, Any]: 
            A dictionary containing the translation results. Keys include:
            - "final_code": The final translated code (string).
            - "success": True if the code passed execution test, False otherwise.
            - "attempts": A list of attempts (with errors, time stamps, etc.).
            - "message": A summary message of success or failure.
            - "metrics": A dictionary with metrics (tokens used, time, etc.).
    
    Raises:
        ValueError: If the source or target language is not recognized.
        FileNotFoundError: If the source_file does not exist.
        Exception: Any other unexpected errors during translation.
    """
    import shutil
    from pathlib import Path
    
    # Local dictionary to handle synonyms (same as the CLI usage)
    LANGUAGE_SYNONYMS = {
        "python": "Python",
        "py": "Python",
        "r": "R",
        "R": "R",
        "jl": "Julia",
        "julia": "Julia",
        "cpp": "C++",
        "c++": "C++",
        "rcpp": "Rcpp",
        "rc++": "Rcpp"
    }
    
    # Normalize language synonyms
    norm_source_lang = LANGUAGE_SYNONYMS.get(source_lang.lower())
    norm_target_lang = LANGUAGE_SYNONYMS.get(target_lang.lower())
    
    if not norm_source_lang:
        raise ValueError(f"Unknown source language: '{source_lang}'")
    if not norm_target_lang:
        raise ValueError(f"Unknown target language: '{target_lang}'")
    
    # Check if source file exists
    source_path = Path(source_file)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Source file '{source_file}' does not exist.")
    
    # Read source code
    with open(source_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    # Import or reference your EnhancedTranslationAgent from the same module
    # Here, we assume the EnhancedTranslationAgent is already defined above in the same script
    agent = EnhancedTranslationAgent(
        provider=provider,
        model=model,
        working_dir=output_dir,
        trace=trace
    )
    
    # Run translation
    result = agent.translate(
        source_code=source_code,
        source_lang=norm_source_lang,
        target_lang=norm_target_lang,
        validate=validate,
        format_code=format_code
    )
    
    # Choose extension
    ext = agent.LANGUAGE_CONFIG[norm_target_lang]["extension"]
    final_code_name = f"final_translated_code{ext}"
    final_code_path = Path(output_dir) / final_code_name
    
    # Write final code
    with open(final_code_path, 'w', encoding='utf-8') as f:
        f.write(result["final_code"])
    
    # Clean up: remove 'temp' folder
    temp_dir_path = Path(output_dir) / "temp"
    if temp_dir_path.exists() and temp_dir_path.is_dir():
        shutil.rmtree(temp_dir_path, ignore_errors=True)
    
    # Remove any other files not named 'translation.log' or 'final_translated_code*'
    for item in Path(output_dir).iterdir():
        if item.is_file():
            if (
                item.name != "translation.log"
                and not item.name.startswith("final_translated_code")
            ):
                item.unlink(missing_ok=True)
    
    return result


###############################################################################
# CLI Interface
###############################################################################

def main():
    """
    CLI interface for the translation agent.
    Ensures that only two files remain in 'output-dir':
        (1) final_translated_code.<ext> (always created)
        (2) translation.log (containing logs)
    All temporary files (including compiled binaries, etc.)
    go into 'temp' subfolder of 'output-dir' and are deleted at the end.
    """
    parser = ArgumentParser(description="AI-powered code translation tool")

    # Map user-friendly language labels to the internal standardized ones
    LANGUAGE_SYNONYMS = {
        "python": "Python",
        "py": "Python",
        "r": "R",
        "jl": "Julia",
        "julia": "Julia",
        "cpp": "C++",
        "c++": "C++",
        "rcpp": "Rcpp",
        "rc++": "Rcpp"
    }

    parser.add_argument(
        "source_file",
        type=str,
        help="Source code file to translate"
    )
    parser.add_argument(
        "source_lang",
        type=str,
        help="Source programming language (e.g. 'py', 'python', 'r', 'R', 'jl', 'julia', 'cpp', 'c++', 'rcpp', 'rc++')"
    )
    parser.add_argument(
        "target_lang",
        type=str,
        help="Target programming language (e.g. 'py', 'python', 'r', 'R', 'jl', 'julia', 'cpp', 'c++', 'rcpp', 'rc++')"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to use"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip code validation"
    )
    parser.add_argument(
        "--no-format",
        action="store_true",
        help="Skip code formatting"
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable detailed logging"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for translated code and logs"
    )
    
    args = parser.parse_args()
    
    # Normalize source/target languages to internal names
    norm_source_lang = LANGUAGE_SYNONYMS.get(args.source_lang.lower())
    norm_target_lang = LANGUAGE_SYNONYMS.get(args.target_lang.lower())
    if not norm_source_lang:
        print(f"Error: unknown source language '{args.source_lang}'")
        exit(1)
    if not norm_target_lang:
        print(f"Error: unknown target language '{args.target_lang}'")
        exit(1)

    try:
        # Read source file
        with open(args.source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Initialize agent
        agent = EnhancedTranslationAgent(
            provider=args.provider,
            model=args.model,
            working_dir=args.output_dir,
            trace=args.trace
        )
        
        # Perform translation
        result = agent.translate(
            source_code=source_code,
            source_lang=norm_source_lang,
            target_lang=norm_target_lang,
            validate=not args.no_validate,
            format_code=not args.no_format
        )
        
        # Choose extension based on target_lang
        ext = agent.LANGUAGE_CONFIG[norm_target_lang]["extension"]
        final_code_name = f"final_translated_code{ext}"
        final_code_path = Path(args.output_dir) / final_code_name
        
        # Write final code
        with open(final_code_path, 'w', encoding='utf-8') as f:
            f.write(result["final_code"])
        
        # Show user feedback
        if result["success"]:
            print("\nTranslation successful!")
        else:
            print(f"\nTranslation failed! Reason:\n{result['message']}")
        
        # Show some minimal metrics
        metrics = result["metrics"]
        print("\n--- Translation Summary ---")
        print(f"Success: {result['success']}")
        print(f"Total tokens used: {metrics['total_tokens']}")
        print(f"Execution time (s): {metrics['execution_time']:.2f}")
        print(f"Total attempts: {metrics['total_attempts']}")

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        exit(1)
    finally:
        # Clean up the temp folder
        output_dir_path = Path(args.output_dir)
        temp_dir_path = output_dir_path / "temp"
        
        if temp_dir_path.exists() and temp_dir_path.is_dir():
            shutil.rmtree(temp_dir_path, ignore_errors=True)
        
        # Remove anything in output-dir that isn't the log or final_translated_code
        for item in output_dir_path.iterdir():
            if item.is_file():
                if (
                    item.name != "translation.log"
                    and not item.name.startswith("final_translated_code")
                ):
                    item.unlink(missing_ok=True)

        print(f"\nAll done. The '{args.output_dir}' folder contains:")
        print(f"  1) {final_code_name}")
        print("  2) translation.log\n")

        # Exit with code 0 if success, 1 if fail
        if 'result' in locals() and not result["success"]:
            exit(1)
        exit(0)

if __name__ == "__main__":
    main()
