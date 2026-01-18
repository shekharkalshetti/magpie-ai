# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2025-01-18

### Added
- Parameter validation for monitor decorator
- Comprehensive README with examples and API documentation
- Validation for project_id, custom metadata, and pricing configuration
- Clear error messages for invalid configurations

### Fixed
- Module import path (triton → magpie_ai)
- Type hints on decorators now properly show `Callable[[F], F]`
- Package metadata in setup.py

### Changed
- Improved docstrings with Raises section
- Better error handling for conflicting pricing parameters

## [0.2.1] - 2025-01-18

### Fixed
- Resolved import path issues (package now correctly exports as magpie_ai)
- Fixed top_level.txt in distribution metadata
- Added proper type hints with py.typed marker

### Added
- Full type support for IDE autocomplete
- Generic type annotations for monitor decorator

## [0.2.0] - 2025-01-18

### Changed
- Renamed package from triton-sdk to magpie-ai
- Updated all internal module references

### Added
- Proper package naming aligned with PyPI repository name

## [0.1.0] - 2024-12-26

### Added
- Initial release
- Core monitoring decorator (@monitor)
- PII detection and redaction
- Content moderation support
- Automatic token counting and cost calculation
- Context manager for fine-grained monitoring
- Support for OpenAI, Anthropic, and custom LLMs
- Comprehensive logging and metrics collection
