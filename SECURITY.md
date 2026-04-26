# Security Policy

EZLLM is a local runtime manager and API proxy. Treat it as local developer
software unless you have reviewed and hardened your deployment.

## Supported Versions

Security fixes target the latest version on the default branch until formal
releases are published.

## Reporting a Vulnerability

Please report security issues privately to the project maintainer before
opening a public issue. Include:

- A clear description of the issue.
- Steps to reproduce.
- Impact and affected versions, if known.
- Any relevant logs with secrets removed.

## Local Runtime Notes

- Do not expose EZLLM or llama.cpp ports to an untrusted network without adding
  your own authentication and network controls.
- Do not commit model files, llama.cpp binaries, runtime logs, or local config
  files containing private paths or credentials.
- Review request and response logs before sharing them because prompts may
  contain sensitive data.
