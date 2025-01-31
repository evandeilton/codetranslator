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

```bash
python codetranslator.py source_file.py python r
```

### Programmatic Usage

```python
from codetranslator import codetranslate

result = codetranslate(
    source_file='example.py', 
    source_lang='python', 
    target_lang='r'
)
```

## Configuration

Set up your API keys in a `.env` file:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Project Link: [https://github.com/evandeilton/codetranslator](https://github.com/evandeilton/codetranslator)