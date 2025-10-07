from pathlib import Path
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parent
    pkg_dir = project_root / "gp_crypto_next"
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))

    from gp_crypto_next.main_gp_new import GPAnalyzer

    yaml_file_path = project_root / "gp_crypto_next" / "parameters.yaml"

    analyzer = GPAnalyzer(str(yaml_file_path))
    analyzer.run()


if __name__ == "__main__":
    main()


