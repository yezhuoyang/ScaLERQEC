# GitHub Configuration

This directory contains GitHub-specific configuration files for the ScaLERQEC repository.

## Contents

### Workflows (`workflows/`)

Automated CI/CD workflows using GitHub Actions:

- **`pr-validation.yml`**: Validates all pull requests with linting, building, testing, and security scanning
- **`ai-code-review.yml`**: Enhanced review process for AI-generated code (triggered by `ai-generated` label)

### Code Ownership (`CODEOWNERS`)

Defines code owners for different parts of the repository. Code owners are automatically requested for review when files they own are modified.

### Pull Request Template (`PULL_REQUEST_TEMPLATE.md`)

Standard template for pull requests to ensure consistent information and proper review process.

### Security Policy (`SECURITY.md`)

Defines how to report security vulnerabilities and our security practices.

## Workflow Overview

### Pull Request Validation Workflow

**Triggers**: Pull requests to main branch, pushes to feature branches

**Jobs**:
1. **Python Linting**: Checks code style with ruff
2. **C++ Code Quality**: Validates C++ formatting
3. **Build & Test (Linux)**: Tests on Ubuntu with Python 3.8-3.11
4. **Build & Test (macOS)**: Tests on macOS with Python 3.11
5. **Build & Test (Windows)**: Tests on Windows with Python 3.11
6. **Security Scanning**: Runs bandit and safety checks
7. **Test Coverage Check**: Ensures ≥60% test coverage
8. **Integration Tests**: Validates end-to-end workflows

**Status**: All jobs must pass for PR to be mergeable (when branch protection is enabled)

### AI Code Review Workflow

**Triggers**: Pull requests labeled with `ai-generated`

**Enhanced Checks**:
1. **Change Scope Analysis**: Verifies changes are within expected scope
2. **Dependency Audit**: Detects and flags new dependencies
3. **Enhanced Coverage**: Requires ≥80% test coverage (vs 60% for regular PRs)
4. **Breaking Change Detection**: Identifies potential API changes
5. **Security-Sensitive File Protection**: Flags changes to critical files
6. **Enhanced Security Scan**: Stricter security analysis

## Using the Workflows

### For Regular Pull Requests

1. Create a feature branch
2. Make your changes
3. Push to GitHub
4. Create a pull request to `main`
5. Wait for automated checks to complete
6. Address any failures
7. Request review from code owners

### For AI-Generated Pull Requests

1. Create a feature branch (e.g., `ai/feature-name`)
2. Let AI make changes
3. Review all AI-generated code carefully
4. Push to GitHub
5. Create a pull request to `main`
6. **Add the `ai-generated` label** to trigger enhanced review
7. Wait for both standard and AI-specific checks to complete
8. Address any failures
9. Request review from repository owner
10. Provide justification for any flagged changes (dependencies, security-sensitive files, etc.)

## Branch Protection (To Be Enabled)

Once branch protection is enabled on the `main` branch, the following rules will be enforced:

- ✅ Require pull request reviews (1 approval, 2 for AI-generated)
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require conversation resolution before merging
- ✅ Restrict direct pushes to main (even from admins)
- ✅ Require linear history

## Maintenance

### Updating Workflows

When modifying workflow files:

1. Test changes in a fork or feature branch first
2. Ensure all jobs still pass
3. Update this README if adding new workflows
4. Get review from repository owner

### Adding New Checks

To add new validation checks:

1. Add a new job to `pr-validation.yml` or create a new workflow file
2. Ensure the job fails appropriately when issues are found
3. Add the job to required status checks in branch protection settings
4. Document the new check in this README

## Troubleshooting

### Workflow Failures

**Build failures**: Check that all system dependencies are correctly installed in the workflow

**Test failures**: Ensure tests pass locally before pushing

**Coverage failures**: Add tests to increase coverage above the threshold

**Security scan failures**: Review and fix flagged security issues

### AI Code Review Failures

**Scope exceeded**: Review the changed files and ensure AI stayed within bounds

**Dependency changes flagged**: Provide justification for new dependencies in PR description

**Coverage too low**: Add more tests for AI-generated code (target ≥80%)

**Security-sensitive files**: Get explicit approval from repository owner

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Code Owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)

---

**Maintained by**: @yezhuoyang  
**Last Updated**: January 4, 2026
