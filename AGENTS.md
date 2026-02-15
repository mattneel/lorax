# Repository Guidelines

## Project Structure & Module Organization
`lib/lorax.ex` is the main public API and defines `Lorax.Config`. Supporting modules live in `lib/lorax/` (`lcm.ex`, `params.ex`, `shape.ex`) and should stay focused on one responsibility each. Tests live in `test/` (`ExUnit` + doctests). Use `examples/` for runnable scripts, `guides/` for Livebook documentation (`.livemd`), `params/` for sample LoRA parameter files, and `data/` for sample text corpora.

## Build, Test, and Development Commands
- `mix deps.get`: install/update dependencies.
- `mix compile`: compile code and surface warnings early.
- `mix test`: run unit tests and doctests.
- `mix format`: apply formatting rules from `.formatter.exs`.
- `mix docs`: build local HexDocs from `README.md` and `guides/*.livemd`.
- `mix run examples/running_gpt_with_lora.exs`: run a reference example script.

## Coding Style & Naming Conventions
Use standard Elixir style with 2-space indentation and always run `mix format` before submitting changes. Module names use `CamelCase` under the `Lorax` namespace; file names and function names use `snake_case`. Keep public APIs documented with `@moduledoc`/`@doc`, and use bang functions (`read!`, `load!`) only when raising on failure is intentional.

## Testing Guidelines
Testing uses ExUnit (`test/test_helper.exs`). Name test files `*_test.exs` and mirror the module under test when practical. Add doctests for public modules and direct unit tests for tensor/shape logic and parameter transforms. There is no enforced coverage threshold, so each bug fix or feature should include at least one regression test. Run `mix test` locally before opening a PR.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects, often with prefixes like `fix:` and `chore:`. Prefer `<type>: <summary>` (for example, `fix: preserve dropout seed behavior`) and keep the subject concise.

For PRs, include:
- a clear summary of what changed and why
- linked issues (if applicable)
- verification steps/commands run (`mix format`, `mix test`, and `mix docs` when docs change)
- updates to `README.md` or `guides/` when public behavior or defaults change
