from pathlib import Path

import numpy as np
import OpenEXR


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "calibration"
OUTPUT_DIR = INPUT_DIR / "samples"

SAMPLES = range(1, 5)


def read_exr(path: Path) -> np.ndarray:
    exr = OpenEXR.File(str(path))
    rgba = np.asarray(exr.channels()["RGBA"].pixels, dtype=np.float32)

    # RGBA HWC -> RGB CHW
    rgb = rgba[..., :3]

    return np.transpose(rgb, (2, 0, 1))


def make_sample(index: int) -> np.ndarray:
    radiance = read_exr(INPUT_DIR / f"radiance_{index:03d}.exr")
    albedo = read_exr(INPUT_DIR / f"albedo_{index:03d}.exr")
    normal = read_exr(INPUT_DIR / f"normal_{index:03d}.exr")

    if not (radiance.shape[1:] == albedo.shape[1:] == normal.shape[1:]):
        raise ValueError(
            f"Sample {index:03d} has mismatched dimensions: "
            f"radiance={radiance.shape}, "
            f"albedo={albedo.shape}, "
            f"normal={normal.shape}"
        )

    sample = np.concatenate(
        [radiance, albedo, normal],
        axis=0,
    )

    # Add batch dimension: [9,H,W] -> [1,9,H,W]
    sample = sample[np.newaxis, ...]

    if sample.shape != (1, 9, 1088, 1920):
        raise ValueError(
            f"Sample {index:03d} has unexpected shape: {sample.shape}"
        )

    if not np.isfinite(sample).all():
        raise ValueError(f"Sample {index:03d} contains NaN or Inf values")

    return sample.astype(np.float32, copy=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index in SAMPLES:
        sample = make_sample(index)

        output = OUTPUT_DIR / f"sample_{index:03d}.npy"
        np.save(output, sample)

        print(
            f"{output.name}: "
            f"shape={sample.shape}, "
            f"dtype={sample.dtype}, "
            f"min={sample.min():.6g}, "
            f"max={sample.max():.6g}"
        )

        for start, name in zip(
            (0, 3, 6),
            ("radiance", "albedo", "normal"),
        ):
            part = sample[0, start:start + 3]

            print(
                f"  {name:8s}: "
                f"min={part.min():.6g}, "
                f"max={part.max():.6g}, "
                f"mean={part.mean():.6g}"
            )


if __name__ == "__main__":
    main()