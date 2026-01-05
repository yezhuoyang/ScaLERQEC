# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.0.1   | :white_check_mark: |

## Reporting a Vulnerability

The ScaLERQEC team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **Email**: Send details to yezhuoyang@cs.ucla.edu
2. **GitHub Security Advisory**: Use the [GitHub Security Advisory](https://github.com/yezhuoyang/ScaLERQEC/security/advisories/new) feature

### What to Include in Your Report

To help us understand and resolve the issue quickly, please include:

- **Type of vulnerability** (e.g., buffer overflow, SQL injection, cross-site scripting)
- **Full paths of source file(s)** related to the vulnerability
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: We aim to acknowledge receipt within 48 hours
- **Status Update**: We will provide a detailed response within 7 days, including:
  - Confirmation of the vulnerability
  - Our planned timeline for a fix
  - Any workarounds or mitigations available
- **Fix Release**: We aim to release a fix within 30 days for critical vulnerabilities

### Disclosure Policy

- We request that you give us reasonable time to address the vulnerability before public disclosure
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We will coordinate the disclosure timeline with you

## Security Best Practices for Contributors

### For All Contributors

1. **Never commit secrets**: Do not commit API keys, passwords, or other sensitive credentials
2. **Review dependencies**: Be cautious when adding new dependencies
3. **Validate inputs**: Always validate and sanitize user inputs
4. **Follow secure coding practices**: Use parameterized queries, avoid eval(), etc.
5. **Keep dependencies updated**: Regularly update dependencies to patch vulnerabilities

### For AI-Assisted Development

When using AI agents (like Manus, GitHub Copilot, etc.) to contribute to this project:

1. **Review all AI-generated code**: Never merge AI-generated code without thorough human review
2. **Check for security issues**: AI may inadvertently introduce vulnerabilities
3. **Verify dependencies**: Ensure AI doesn't add unnecessary or insecure dependencies
4. **Test thoroughly**: AI-generated code requires comprehensive testing
5. **Use the 'ai-generated' label**: Mark PRs containing AI code for enhanced scrutiny

### Security-Sensitive Areas

The following areas require extra caution and owner review:

- Build configuration (`setup.py`, `pyproject.toml`)
- CI/CD workflows (`.github/workflows/`)
- C++ backend (`QEPG/src/`)
- Authentication or authorization code
- Input parsing and validation
- External API interactions

## Security Features

### Current Security Measures

- **Automated security scanning**: GitHub Actions run security scans on all PRs
- **Dependency vulnerability checking**: Automated checks for known vulnerabilities
- **Code review requirements**: All changes require review before merge
- **Branch protection**: Main branch is protected from direct pushes
- **AI code review**: Enhanced scrutiny for AI-generated code

### Planned Security Enhancements

- Dependabot integration for automated dependency updates
- Regular security audits
- Signed commits requirement
- Additional static analysis tools

## Known Security Considerations

### Quantum Computing Context

As a quantum error correction framework, ScaLERQEC is designed for research and simulation purposes. When using this software:

- **Not for production quantum systems**: This is a research tool, not production-ready quantum hardware control
- **Simulation only**: The framework simulates quantum circuits and error correction
- **No cryptographic guarantees**: This is not a cryptographic library

### C++ Backend

The C++ backend uses:

- **Boost**: For dynamic bitsets and data structures
- **Eigen**: For linear algebra operations
- **pybind11**: For Python bindings

These dependencies are well-established but should be kept updated.

## Security Update Process

When a security vulnerability is confirmed:

1. **Assessment**: We assess the severity and impact
2. **Fix Development**: We develop and test a fix
3. **Security Advisory**: We create a GitHub Security Advisory
4. **Release**: We release a patched version
5. **Notification**: We notify users through:
   - GitHub Security Advisory
   - Release notes
   - README update (if critical)

## Contact

For security concerns, contact:

- **Primary**: yezhuoyang@cs.ucla.edu
- **GitHub**: @yezhuoyang

---

**Last Updated**: January 4, 2026
