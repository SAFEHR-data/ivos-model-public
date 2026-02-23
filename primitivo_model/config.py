from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.markdown import Markdown
from rich.panel import Panel

__all__ = ["settings"]


def _project_root() -> Path:
    try:
        return Path(__file__).parents[1]
    except NameError:
        # running interactively & no `__file__`
        return Path.cwd()


def _find_env() -> Path:
    try:
        env_path = _project_root() / ".env"
    except NameError:
        # running interactively & no `__file__`
        env_path = Path(".env")
    return env_path


class Settings(BaseSettings):
    # Data split ratios
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Path configuration
    PROJECT_ROOT: Path = _project_root()
    DATA_ROOT: Path = None

    model_config = SettingsConfigDict(
        case_sensitive=True,
        validate_default=False,
        env_file=_find_env(),
        extra="ignore",
    )

    def as_md(self) -> Panel:
        md = Markdown("   \n".join([f"`{s[0]}`: {s[1]}" for s in self]))
        return Panel(md, title="[bold white]primitivo_model settings[/]")


def _mk_settings() -> Settings:
    """
    Lazily build a Settings object and customize

    :return: Settings instance
    """
    s = Settings()

    s.DATA_ROOT = s.PROJECT_ROOT / "data"

    return s


settings = _mk_settings()  # Singleton for settings
