from dataclasses import dataclass, field


@dataclass
class StepResult:
    output_video_path: str
    extra_paths: dict[str, str] = field(default_factory=dict)
    skipped: bool = False
