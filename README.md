# Reflex SDK skill

Agent skills for building robotics applications on top of the [Reflex Labs](https://tryreflex.ai) fine-tuning + inference API.

[![skills.sh](https://skills.sh/b/reflex-inc/reflex-sdk-skill)](https://skills.sh/reflex-inc/reflex-sdk-skill)

## Install

```bash
npx skills add reflex-inc/reflex-sdk-skill
```

Works with Claude Code, Cursor, Codex, OpenCode, GitHub Copilot, Windsurf, Gemini, and [50+ other agents](https://skills.sh).

## Available skills

### `reflex-sdk-builder`

Teaches an agent to use the public `reflex-sdk` Python package to:

- authenticate with an `rfx_…` API key (env var or key file)
- submit a LoRA fine-tune of `pi0.5` on a public LeRobot HuggingFace dataset and poll the run
- call the Reflex inference API and parse the 50-step × 14-DOF action chunk
- drive an SO-101 arm at 25 Hz with raw-step ↔ radians conversion and per-tick safety clipping
- handle cold-start retries, dimension mapping for single-arm robots, and adapter selection

**Triggers** on phrases like "reflex sdk", "reflex api", "tryreflex", "fine-tune pi0.5", "lora finetune", "robot inference", "VLA inference", "lerobot training", or "SO-101 control".

The canonical, end-to-end working example lives at https://github.com/reflex-inc/quickstart.

## License

Apache-2.0 — see [LICENSE](LICENSE).
