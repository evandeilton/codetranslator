# CodeTranslator

## Overview

CodeTranslator is an advanced AI-powered code translation tool that leverages large language models to translate code between multiple programming languages, including Python, R, Julia, and C++.

## Features

- Multi-language support (Python, R, Julia, C++)
- AI-powered translation using OpenAI and Anthropic models
- Code validation and formatting
- Detailed translation metrics and logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/evandeilton/codetranslator.git
cd codetranslator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### CLI Usage

Translate between different programming languages directly from the command line:

```bash
# Basic translation
python codetranslator.py tests/teste.py python r

# Specify a different provider
python codetranslator.py tests/teste.py python julia --provider anthropic

# Disable validation or formatting
python codetranslator.py tests/teste.py python cpp --no-validate --no-format
```

### Programmatic Usage

Use the `codetranslate()` function in your Python scripts:

```python
from codetranslator import codetranslate

# Translate Python to R
result = codetranslate(
    source_file='tests/teste.py', 
    source_lang='python', 
    target_lang='r'
)

# Print translation details
print(result['final_code'])
print(f"Translation success: {result['success']}")
```

### Supported Languages

CodeTranslator supports translations between:
- Python
- R
- Julia
- C++
- Rcpp

### Example Test Files

The `tests/` directory contains example source files for different languages:
- `teste.py`: Python implementation of bubble sort
- `teste.R`: R script example
- `teste.jl`: Julia script example
- `teste.cpp`: C++ script example

## Configuration

### API Keys

Set up your API keys in a `.env` file:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Advanced Options

- `provider`: Choose between 'openai' and 'anthropic'
- `model`: Specify a particular LLM model
- `validate`: Enable/disable code validation
- `format_code`: Enable/disable code formatting
- `trace`: Enable detailed logging

## Performance Metrics

CodeTranslator provides detailed metrics after each translation:
- Total tokens used
- Execution time
- Number of translation attempts
- Success/failure status

## Limitations

- Translation quality depends on the complexity of the source code
- Some language-specific idioms may not translate perfectly
- Requires API access to OpenAI or Anthropic

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Project Link: [https://github.com/evandeilton/codetranslator](https://github.com/evandeilton/codetranslator)