"""CLI является тонкой оболочкой над тем же публичным Python API."""
import argparse
from . import Engine, get_config, analyze_config, read_config, STANDARD_CONFIGS


def main():
    parser = argparse.ArgumentParser(description='Starling ML config analyzer and runner')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', help='Единый YAML или каталог четырёх YAML')
    source.add_argument('--recipe', choices=STANDARD_CONFIGS, default='segmentation')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    config = read_config(args.config) if args.config else get_config(args.recipe, max_steps=args.steps)
    report = analyze_config(config)
    print(report)
    report.raise_for_errors()
    if not args.analyze_only:
        engine = Engine(config).run()
        print(f"Завершено optimizer steps: {engine.context.data.get('run.step', 'n/a')}")


if __name__ == '__main__':
    main()
