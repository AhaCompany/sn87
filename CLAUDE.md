# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install the package
python -m pip install -e .

# Run all tests
pytest

# Run a single test
pytest tests/test_file.py::TestClass::test_method

# Verify Python compatibility
./scripts/check_compatibility.sh <python_version>
```

## Code Style

- **Format**: Use `black` for consistent formatting
- **Indentation**: 4 spaces
- **Line Length**: 79 characters max
- **Imports**: Order by standard library, third-party, local
- **Naming**: 
  - Classes: CapWords
  - Functions/Variables: lowercase_with_underscores
  - Constants: ALL_CAPS
  - Non-public: _single_leading_underscore
  - Private: __double_leading_underscore
- **Docstrings**: Include for classes and functions
- **Error Handling**: Use exceptions, not return codes
- **String Formatting**: Prefer f-strings
- **Comments**: Complete sentences, explain why not how

## Git Guidelines

- Atomic commits focusing on single tasks
- Subject line: 50 chars max, imperative mood
- Body: Wrapped at 72 chars, explains what/why
- PRs should be focused with < 50 files
- New features must include tests