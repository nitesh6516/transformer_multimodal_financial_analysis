from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
PUBLIC_STATIC_ROOT = ROOT / "public" / "static"


def main() -> None:
    if PUBLIC_STATIC_ROOT.exists():
        shutil.rmtree(PUBLIC_STATIC_ROOT)
    PUBLIC_STATIC_ROOT.mkdir(parents=True, exist_ok=True)

    for folder in ("assets", "css", "js"):
        source = STATIC_ROOT / folder
        if source.exists():
            shutil.copytree(source, PUBLIC_STATIC_ROOT / folder)


if __name__ == "__main__":
    main()
