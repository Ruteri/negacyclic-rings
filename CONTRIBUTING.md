# Contributing

Enable the repository's pre-commit checks once after cloning:

```sh
git config core.hooksPath .githooks
```

The hook verifies formatting and runs Clippy with warnings denied. Full tests run in GitHub CI on x86_64 and AArch64.
