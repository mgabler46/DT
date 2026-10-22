#!/usr/bin/env python3
r"""
GTX 1060 - ca. 10 STUNDEN GESAMT - NEUE SEEDS - SI04 / SI05 / SI09 - ALLE 7 KANALSETS

Eigenstaendige Datei; V36/V37 werden zum Ausfuehren NICHT benoetigt.
Start unter Windows / PowerShell (Datenpfad bei Bedarf anpassen):
& ".\.venv\Scripts\python.exe" ".\hpo_gtx1060_10h_newseeds.py" "E:\DiplomDaten\CSV_Export"
Fortsetzen: gleicher Befehl mit --resume.
Nach dem frueheren Pilot-Abbruch: Datei ersetzen und denselben Befehl mit --resume starten.
Abgeschlossene V38.1.0-Pilotmessungen werden bei identischen Einstellungen uebernommen.
Nur Laufzeitpilot: --pilot-only; danach --resume ohne --pilot-only.
Logiktest OHNE Torch/Download: --logic-test
Technischer Torch-Test OHNE Download: --self-test
Echte ImageNet-Gewichte technisch pruefen: --check-model
Kurzer echter Nested-Lauf: --smoke-test --output-dir results_gtx1060_10h_smoke

Standard: sieben nichtleere Sets aus SI04/SI05/SI09; nur Winkelregression.
Feste SI15-Quellindices 3/4/8 (nullbasiert); SI05 wird wirklich neu geladen.
Immer personenweise Nested-CV: 3 neue Split-Seeds x 3 Outer-Folds, je 3 Inner-Folds.
Split-Seeds: 101 / 307 / 911. Trainings-Seeds: 1709 / 2903.
Neuer Suchseed: 93871. Neuer Augmentationsseed: 4243.
Alle sieben Kanalsets bleiben im Vergleich. Neue Seeds verwenden dieselbe
Kohorte und pruefen Split-/Trainingsstabilitaet. Es sind keine neuen Personen.
Alle Messungen/Kontakte einer Person bleiben zusammen. Kein Holdout-Modus.
CNN oder CNN+ein kleiner Transformer, eingefrorenes vortrainiertes ConvNeXt.
Primaer inneres ungewichtetes Cohen's Kappa. Smooth-L1-Regressionsloss.
Kein Klassifikationskopf und keine optimierte Entscheidungsschwelle.
Klassen unveraendert: TH>25 Grad; L/THL>20 Grad, strikt groesser.
Checkpoint, Konfiguration, Kanalset und Epochenzahl werden NUR innen gewaehlt.
Alle sieben Kanalsets erhalten aeussere Tests. hpo_inner_selected ist primaer.
Die zusaetzliche primaere Zeile verwendet vorhandene Vorhersagen, keine Fits.

Laufzeitziel: etwa 10 h zur Abschaetzung der Operationszahl, KEINE Zeitgrenze.
Der Pilot waehlt dieselbe Startzahl aus 8, 12, ..., 64 pro Kanalset.
Wenn schon 8 je Set laenger geschaetzt werden, laufen diese trotzdem vollstaendig.
Mit --search-configs N wird N direkt festgelegt und unabhaengig von der ETA ausgefuehrt.
Pilot, Featureextraktion, innere Suche und aeussere Tests duerfen laenger dauern.
Es gibt weder eine automatische Zeitsperre noch einen Abbruch wegen hoher ETA.
Broad bis 4 Epochen/1 Seed -> 2 Kandidaten je Set und Modell; Stable bis 12/1 Seed
-> 1 je Set und Modell; Final bis 25/2 Seeds -> 1 Sieger je Set fuer aeussere Tests.
Modell/Optimierer/RNG werden je Epoche gesichert. Manueller Abbruch: --resume.
Der feste Arbeitsplan steht in operation_plan.json. Bei N=8 und Standard-CV:
2016 verschiedene trainierbare Modellinitialisierungen, maximal 29610 Kopf-/
Transformer-Epochen sowie Featureextraktion und Pilot. 126 aeussere Modellfits.
Early Stopping kann die tatsaechliche Epochensumme reduzieren.

GTX-1060-Profil: FP32, kein BF16/TF32, ein GPU-Prozess, 96x160-Bilder.
CNN-Microbatch 4 (bei 6 GB optional --contact-microbatch 8), OOM-Halbierung.
Kopf-Microbatch 256; Personenbatch 8 bleibt bei OOM gleich gewichtet.
Eingaben auf CPU; Featuremaps als CPU-Memmap auf Festplatte, standardmaessig
max. 8 GiB pro Outer-Kontext (--feature-cache-gb). Freien Plattenplatz vorsehen.
Feste drei Bildbaender. Abwesende Kanaele bleiben konstant und ohne Rauschen.
Keine SI-Z-Normierung; Imputation/Regressionszielskalierung nur aus Training.
Die sieben Sets verwenden dieselbe Kohorte, Splits und Hyperparameter-Vorlagen.

Pakete: numpy, pandas sowie dein funktionierendes CUDA-Torch/Torchvision-Paar.
Keine automatische Installation. Vorhandene CUDA-Umgebung nicht blind upgraden.
Ehemalige Umgebung: torch 2.7.1 / torchvision 0.22.1. Offizielle Basisgewichte
werden beim ersten Start geladen, falls nicht bereits im Gewichtecache.
CSVs sind semikolongetrennt, UTF-8-BOM. Aggregierte CSVs ohne Personenkennungen;
_private_predictions und outer_models bleiben auf deinem Trainingsrechner.
Wichtigste Dateien: run_status.json, method_summary.csv, channel_summary.csv,
outer_fold_metrics.csv, seed_summary.csv, paired_channel_sets.csv und
inner_channel_selection.csv.
Fertig erst bei complete_nested_cv=true; Standard 63 Ensembles/126 Mitglieder,
72 Ergebniszeilen (7 Sets plus hpo_inner_selected in 9 Outer-Kontexten).
Entwicklung an derselben Kohorte bleibt ergebnisgeleitet; neue unabhaengige
Daten waeren zur Bestaetigung des gesamten Entwicklungsprozesses erforderlich.

API-Referenzen:
https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.convnext_tiny.html
https://docs.pytorch.org/docs/2.7/generated/torch.from_numpy.html
https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html
"""



from __future__ import annotations


import argparse


import copy


import gc


import hashlib


import itertools


import json


import math


import os


import random


import re


import socket


import sys


import tempfile


import time


import unicodedata


import uuid


import warnings


from collections import Counter, OrderedDict


from contextlib import contextmanager, nullcontext


from datetime import datetime, timezone


from dataclasses import asdict, dataclass, replace


from pathlib import Path


from typing import Any, Sequence


import numpy as np


import pandas as pd


try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


KINEMATIC_NAMES = (
    "avgV_x", "peakV_x", "avgV_y", "peakV_y", "avgV_2d", "peakV_2d",
    "avgA_x", "peakA_x", "avgA_y", "peakA_y", "avgA_2d", "peakA_2d",
)


GLOBAL_COP_FEATURES = (
    *KINEMATIC_NAMES, "pathLengthAsymmetry", "areaAsymmetry",
    "distanceFirstPeak_X", "distanceFirstPeak_Y",
)


FOOT_COP_FEATURES = (
    *(f"foot_{name}" for name in KINEMATIC_NAMES),
    "foot_pathLengthAsymmetry", "foot_areaAsymmetry",
    "foot_distanceFirstPeak_X", "foot_distanceFirstPeak_Y",
)


LEGACY_MODEL_FEATURES = (
    "gender", "siFirstMax", "numExtrema", "numSattel",
    *GLOBAL_COP_FEATURES, *FOOT_COP_FEATURES,
)


CONTACT_METRIC_BASES = (
    "contactDuration", "vgrfFirstMax", "vgrfNumExtrema", "vgrfNumSaddle",
    "vgrfMinimumAbs", "vgrfLoadingSlope", "vgrfUnloadingSlope",
    *(f"global_{name}" for name in KINEMATIC_NAMES),
    "global_pathLength", "global_area", "global_xAtFirstPeak",
    "global_yAtFirstPeak",
    *(f"foot_{name}" for name in KINEMATIC_NAMES),
    "foot_pathLength", "foot_area", "foot_xAtFirstPeak",
    "foot_yAtFirstPeak",
)


EXTENDED_BILATERAL_FEATURES = tuple(
    feature
    for base in CONTACT_METRIC_BASES
    for feature in (
        f"{base}_raw_avg", f"{base}_raw_peak",
        f"{base}_si_avg", f"{base}_si_peak",
    )
)


MODEL_FEATURES = (*LEGACY_MODEL_FEATURES, *EXTENDED_BILATERAL_FEATURES)


SOURCE_CHANNELS = 15


EXPECTED_CHANNELS = 3


POINTS_PER_PAIR = 101


ANGLE_BIN_WIDTH_DEG = 5.0


ANGLE_HALF_WIDTH_DEG = 5.0


CONTACT_DURATION_TOLERANCE_FRACTION = 0.25


CONTACT_ZERO_TOLERANCE = 0.0


MIN_SI_PAIRS_PER_SUBJECT = 3


SAMPLING_RATE_HZ = 100.0


CLIP_Z = 8.0


ANGLE_CLASS_THRESHOLDS_DEG = {"TH": 25.0, "L": 20.0, "THL": 20.0}


TIME_COLUMN_CANDIDATES = (
    "time_s", "time", "timestamp", "t", "zeit_s", "zeit",
    "time_ms", "zeit_ms", "timestamp_ms",
    "time_us", "zeit_us", "timestamp_us",
)


SOURCE_SI15_CHANNELS = (
    "foot_cop_x_si_dom_minus_other",
    "foot_cop_y_si_dom_minus_other",
    "foot_cop_v_si_dom_minus_other",
    "foot_cop_a_si_dom_minus_other",
    "total_cop_x_si_dom_minus_other",
    "total_cop_y_si_dom_minus_other",
    "total_cop_v_si_dom_minus_other",
    "total_cop_a_si_dom_minus_other",
    "force_si_max_dom_minus_other",
    "foot_cop_x_raw_all_measurements_maxnorm_dominant",
    "foot_cop_x_raw_all_measurements_maxnorm_other",
    "total_cop_v_raw_all_measurements_maxnorm_dominant",
    "total_cop_v_raw_all_measurements_maxnorm_other",
    "total_cop_a_raw_all_measurements_maxnorm_dominant",
    "total_cop_a_raw_all_measurements_maxnorm_other",
)


SELECTED_SOURCE_CHANNEL_INDICES = (3, 4, 8)


SELECTED_INPUT_CHANNELS = tuple(
    SOURCE_SI15_CHANNELS[index] for index in SELECTED_SOURCE_CHANNEL_INDICES
)


RAW_COP_INDICES = ()


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


@dataclass
class Measurement:
    measurement_id: str
    source_file: str
    subject: str
    pairs: np.ndarray


@dataclass
class SubjectRecord:
    subject: str
    angle_deg: float
    curve_type: str
    boundary_deg: float
    soft_target: float
    hard_reference: int
    measurements: list[Measurement]

    @property
    def pairs(self) -> np.ndarray:
        return np.concatenate([item.pairs for item in self.measurements], axis=0)

    @property
    def n_pairs(self) -> int:
        return int(sum(len(item.pairs) for item in self.measurements))


@dataclass(frozen=True)
class CVSplit:
    repeat: int
    fold: int
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]


def probability_above_boundary(
    angle_deg: np.ndarray | float,
    boundary_deg: np.ndarray | float,
) -> np.ndarray:
    """P(Klasse 1) fuer einen um +/-5 Grad gleichverteilten Winkel."""
    return np.clip(
        (
            np.asarray(angle_deg, np.float64)
            + ANGLE_HALF_WIDTH_DEG
            - np.asarray(boundary_deg, np.float64)
        )
        / (2.0 * ANGLE_HALF_WIDTH_DEG),
        0.0,
        1.0,
    )


def parse_measurement_filename(stem: str) -> tuple[str, str, float, str]:
    """Liest <Subjekt>-<Kontrollklasse>-<Winkel>-<TH|L|THL>-<Rest>."""
    parts = str(stem).split("-", 4)
    if len(parts) < 5:
        raise ValueError(
            "Dateiname muss '<Subjekt>-<Kontrollklasse>-<Winkel>-"
            "<TH|L|THL>-<Messung>.csv' folgen."
        )
    subject = parts[0].strip()
    control = parts[1].strip()
    angle_text = parts[2].strip().replace("°", "").replace(",", ".")
    curve_type = parts[3].strip().upper()
    if not subject or not control or not angle_text:
        raise ValueError("Subjekt, Kontrollklasse oder Winkel ist leer.")
    try:
        angle = float(angle_text)
    except ValueError as error:
        raise ValueError(f"Winkel '{parts[2]}' ist keine Zahl.") from error
    if not np.isfinite(angle):
        raise ValueError("Winkel ist nicht endlich.")
    if curve_type not in ANGLE_CLASS_THRESHOLDS_DEG:
        raise ValueError(
            f"Typ '{parts[3]}' ist ungueltig; erwartet: TH, L oder THL."
        )
    return subject, control, angle, curve_type


def _read_measurement_csv(path: Path) -> pd.DataFrame:
    """Robuster Import fuer Semikolon-/Komma-/Tabulator-CSV."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        for separator, decimal in ((";", ","), (",", "."), ("\t", ".")):
            try:
                frame = pd.read_csv(
                    path,
                    sep=separator,
                    decimal=decimal,
                    encoding=encoding,
                    low_memory=False,
                )
                frame.columns = [
                    re.sub(r"\s+", "_", str(column).strip().lower())
                    for column in frame.columns
                ]
                if {"force_left", "force_right"}.issubset(frame.columns):
                    return frame
            except (OSError, UnicodeError, pd.errors.ParserError) as error:
                last_error = error
    raise ValueError(f"CSV nicht lesbar oder Kraftspalten fehlen: {last_error}")


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan, np.float64)
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
    text_values = series.astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    both = text_values.str.contains(",", regex=False) & text_values.str.contains(
        ".", regex=False
    )
    european = both & (text_values.str.rfind(",") > text_values.str.rfind("."))
    american = both & ~european
    text_values.loc[european] = text_values.loc[european].str.replace(
        ".", "", regex=False
    )
    text_values.loc[american] = text_values.loc[american].str.replace(
        ",", "", regex=False
    )
    text_values = text_values.str.replace(",", ".", regex=False)
    return pd.to_numeric(text_values, errors="coerce").to_numpy(np.float64)


def _find_time_column(frame: pd.DataFrame) -> str | None:
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    for column in frame.columns:
        if "time" in str(column) or "zeit" in str(column):
            return str(column)
    return None


def _time_to_seconds(values: np.ndarray, column_name: str) -> np.ndarray:
    name = str(column_name).lower()
    if "ns" in name:
        scale = 1e-9
    elif "µs" in name or "μs" in name or "us" in name:
        scale = 1e-6
    elif "ms" in name:
        scale = 1e-3
    else:
        scale = 1.0
    return np.asarray(values, np.float64) * scale


def resolve_time_seconds(
    frame: pd.DataFrame, source_file: str
) -> tuple[np.ndarray, float, str]:
    column = _find_time_column(frame)
    if column is None:
        time_values = np.arange(len(frame), dtype=np.float64) / SAMPLING_RATE_HZ
        return time_values, SAMPLING_RATE_HZ, "configured_sampling_rate"
    series = pd.to_numeric(frame[column], errors="coerce").interpolate(
        limit_direction="both"
    )
    time_values = _time_to_seconds(series.to_numpy(np.float64), column)
    if len(time_values) < 2 or not np.all(np.isfinite(time_values)):
        raise ValueError(f"Ungueltige Zeitspalte '{column}' in {source_file}.")
    differences = np.diff(time_values)
    if np.any(differences <= 0.0):
        raise ValueError(
            f"Zeitspalte '{column}' ist in {source_file} nicht streng monoton."
        )
    return time_values, float(1.0 / np.median(differences)), str(column)


def _contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, bool)
    if not len(values):
        return []
    changes = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _contact_segments(
    frame: pd.DataFrame,
    side: str,
    force: np.ndarray,
    force_threshold: float,
) -> tuple[list[tuple[int, int, str]], str]:
    id_column = f"contact_id_{side}"
    mask_column = f"contact_{side}"
    if id_column in frame:
        raw = frame[id_column].astype(str).str.strip()
        inactive = raw.str.lower().isin(("", "0", "0.0", "nan", "none", "false"))
        segments: list[tuple[int, int, str]] = []
        start: int | None = None
        current: str | None = None
        for index, value in enumerate(raw):
            active = not bool(inactive.iloc[index])
            if active and (start is None or str(value) != current):
                if start is not None:
                    segments.append((start, index, str(current)))
                start, current = index, str(value)
            elif not active and start is not None:
                segments.append((start, index, str(current)))
                start, current = None, None
        if start is not None:
            segments.append((start, len(raw), str(current)))
        return segments, id_column
    if mask_column in frame:
        numeric = _numeric(frame, mask_column)
        words = frame[mask_column].astype(str).str.strip().str.lower()
        mask = (np.nan_to_num(numeric) > 0.0) | words.isin(
            ("true", "yes", "ja")
        ).to_numpy()
        return [(start, stop, "") for start, stop in _contiguous_true_segments(mask)], mask_column
    mask = np.isfinite(force) & (force > float(force_threshold))
    return [
        (start, stop, "") for start, stop in _contiguous_true_segments(mask)
    ], f"force_{side} > {float(force_threshold):g}"


def _dynamics(
    x: np.ndarray,
    y: np.ndarray,
    time_values: np.ndarray,
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values, times = [
        np.asarray(values, np.float64) for values in (x, y, time_values)
    ]
    valid = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(times)
    if allowed is not None:
        valid &= np.asarray(allowed, bool)
    speed = np.full(len(times), np.nan, np.float64)
    acceleration = np.full(len(times), np.nan, np.float64)
    for start, stop in _contiguous_true_segments(valid):
        count = stop - start
        if count < 2 or np.any(np.diff(times[start:stop]) <= 0.0):
            continue
        edge_order = 2 if count >= 3 else 1
        velocity_x = np.gradient(
            x_values[start:stop], times[start:stop], edge_order=edge_order
        )
        velocity_y = np.gradient(
            y_values[start:stop], times[start:stop], edge_order=edge_order
        )
        speed[start:stop] = np.hypot(velocity_x, velocity_y)
        if count >= 3:
            acceleration_x = np.gradient(
                velocity_x, times[start:stop], edge_order=2
            )
            acceleration_y = np.gradient(
                velocity_y, times[start:stop], edge_order=2
            )
            acceleration[start:stop] = np.hypot(acceleration_x, acceleration_y)
    return speed, acceleration


def _resample_piecewise(
    values: np.ndarray,
    native_time: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    """Interpoliert nur innerhalb zusammenhaengender endlicher Abschnitte."""
    values = np.asarray(values, np.float64)
    native_time = np.asarray(native_time, np.float64)
    target_time = np.asarray(target_time, np.float64)
    result = np.full(len(target_time), np.nan, np.float64)
    valid = np.isfinite(values) & np.isfinite(native_time)
    for start, stop in _contiguous_true_segments(valid):
        if stop - start == 1:
            nearest = int(np.argmin(np.abs(target_time - native_time[start])))
            if np.isclose(target_time[nearest], native_time[start], atol=1e-10):
                result[nearest] = values[start]
            continue
        selected = (
            (target_time >= native_time[start] - 1e-10)
            & (target_time <= native_time[stop - 1] + 1e-10)
        )
        result[selected] = np.interp(
            target_time[selected], native_time[start:stop], values[start:stop]
        )
    return result


def _phase_score(
    values: np.ndarray, native_time: np.ndarray
) -> float:
    target = np.linspace(native_time[0], native_time[-1], POINTS_PER_PAIR)
    phase_values = _resample_piecewise(values, native_time, target)
    valid = np.isfinite(phase_values[:-1]) & np.isfinite(phase_values[1:])
    if not np.any(valid):
        return 0.0
    phase = np.linspace(0.0, 100.0, POINTS_PER_PAIR)
    return float(
        np.sum(
            0.5
            * (phase_values[:-1][valid] + phase_values[1:][valid])
            * np.diff(phase)[valid]
        )
    )


def _phase_resample(values: np.ndarray, native_time: np.ndarray) -> np.ndarray:
    """Exakte 0--100-%-Interpolation der bisherigen V12/V21-Aufbereitung."""
    values = np.asarray(values, np.float64)
    native_time = np.asarray(native_time, np.float64)
    result = np.full(POINTS_PER_PAIR, np.nan, np.float64)
    if not len(native_time):
        return result
    if len(native_time) == 1:
        if np.isfinite(values[0]):
            result[0] = values[0]
        return result
    duration = float(native_time[-1] - native_time[0])
    if not np.isfinite(duration) or duration <= 0.0:
        return result
    target_phase = np.linspace(0.0, 100.0, POINTS_PER_PAIR)
    native_phase = 100.0 * (native_time - native_time[0]) / duration
    for start, stop in _contiguous_true_segments(np.isfinite(values)):
        if stop - start == 1:
            nearest = int(np.argmin(np.abs(target_phase - native_phase[start])))
            if np.isclose(
                target_phase[nearest], native_phase[start], atol=1e-9
            ):
                result[nearest] = values[start]
            continue
        selected = (
            (target_phase >= native_phase[start] - 1e-10)
            & (target_phase <= native_phase[stop - 1] + 1e-10)
        )
        result[selected] = np.interp(
            target_phase[selected], native_phase[start:stop], values[start:stop]
        )
    return result


def _phase_integral(values: np.ndarray) -> float:
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    if not np.any(valid):
        return np.nan
    phase = np.linspace(0.0, 100.0, POINTS_PER_PAIR)
    return float(
        np.sum(
            0.5
            * (values[:-1][valid] + values[1:][valid])
            * np.diff(phase)[valid]
        )
    )


def _nanmean_curves(curves: Sequence[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if not curves:
        return np.full(shape, np.nan, np.float64)
    array = np.asarray(curves, np.float64)
    valid = np.isfinite(array)
    count = valid.sum(axis=0)
    return np.divide(
        np.where(valid, array, 0.0).sum(axis=0),
        count,
        out=np.full(shape, np.nan, np.float64),
        where=count > 0,
    )


def _symmetry_index(
    dominant: np.ndarray, other: np.ndarray, denominator_mode: str = "amplitude"
) -> np.ndarray:
    """Dimensionsloser SI: 2*(dominant-other)/Amplitudensumme."""
    dominant = np.asarray(dominant, np.float64)
    other = np.asarray(other, np.float64)
    if denominator_mode != "amplitude":
        raise ValueError("V29.11 verwendet verbindlich den Amplituden-SI.")
    dominant_max = np.max(
        np.where(np.isfinite(dominant), np.abs(dominant), -np.inf),
        axis=-1,
        keepdims=True,
    )
    other_max = np.max(
        np.where(np.isfinite(other), np.abs(other), -np.inf),
        axis=-1,
        keepdims=True,
    )
    dominant_max[~np.isfinite(dominant_max)] = np.nan
    other_max[~np.isfinite(other_max)] = np.nan
    denominator = dominant_max + other_max
    valid = (
        np.isfinite(dominant)
        & np.isfinite(other)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-12)
    )
    return np.divide(
        2.0 * (dominant - other),
        denominator,
        out=np.full_like(dominant, np.nan),
        where=valid,
    )


def _apply_contact_duration_qc(
    contacts: list[dict[str, Any]], tolerance_fraction: float
) -> dict[str, Any]:
    tolerance = float(tolerance_fraction)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("Kontaktdauer-Toleranz muss in (0,1) liegen.")
    durations = np.asarray(
        [contact["duration_seconds"] for contact in contacts], np.float64
    )
    zero_complete = np.asarray(
        [contact["zero_boundary_valid"] for contact in contacts], bool
    )
    eligible = np.isfinite(durations) & (durations > 0.0) & zero_complete
    mean_duration = float(np.mean(durations[eligible])) if np.any(eligible) else np.nan
    lower = mean_duration * (1.0 - tolerance) if np.isfinite(mean_duration) else np.nan
    upper = mean_duration * (1.0 + tolerance) if np.isfinite(mean_duration) else np.nan
    for contact, duration in zip(contacts, durations):
        if np.isfinite(mean_duration) and mean_duration > 0.0 and np.isfinite(duration):
            deviation = abs(float(duration) - mean_duration) / mean_duration
            duration_valid = bool(
                duration > 0.0 and deviation <= tolerance + 1e-12
            )
        else:
            deviation = np.nan
            duration_valid = False
        contact["duration_deviation_fraction"] = float(deviation)
        contact["duration_within_tolerance"] = duration_valid
        contact["contact_valid"] = bool(
            contact["zero_boundary_valid"] and duration_valid
        )
        contact["duration_valid"] = contact["contact_valid"]
        if not contact["zero_start_valid"] and not contact["zero_end_valid"]:
            reason = "missing_zero_at_start_and_end"
        elif not contact["zero_start_valid"]:
            reason = "missing_zero_at_start"
        elif not contact["zero_end_valid"]:
            reason = "missing_zero_at_end"
        elif not np.isfinite(duration) or duration <= 0.0:
            reason = "invalid_duration"
        elif duration_valid:
            reason = "within_tolerance"
        elif duration < lower:
            reason = "too_short"
        else:
            reason = "too_long"
        contact["duration_qc_reason"] = reason
    return {
        "mean_seconds": mean_duration,
        "lower_seconds": lower,
        "upper_seconds": upper,
        "tolerance_fraction": tolerance,
        "n_contacts": len(contacts),
        "n_valid": int(sum(contact["contact_valid"] for contact in contacts)),
        "n_rejected": int(sum(not contact["contact_valid"] for contact in contacts)),
        "n_zero_boundary_valid": int(sum(zero_complete)),
        "n_zero_boundary_rejected": int(sum(~zero_complete)),
        "n_duration_within_tolerance": int(
            sum(contact["duration_within_tolerance"] for contact in contacts)
        ),
        "n_too_short": int(
            sum(contact["duration_qc_reason"] == "too_short" for contact in contacts)
        ),
        "n_too_long": int(
            sum(contact["duration_qc_reason"] == "too_long" for contact in contacts)
        ),
    }


def _prepare_raw_measurement(
    path: Path,
    force_threshold: float,
    duration_tolerance: float,
    zero_tolerance: float,
) -> dict[str, Any]:
    frame = _read_measurement_csv(path)
    subject, control, angle, curve_type = parse_measurement_filename(path.stem)
    time_values, sampling_rate, time_source = resolve_time_seconds(frame, path.name)
    force = {
        side: _numeric(frame, f"force_{side}") for side in ("left", "right")
    }
    foot_x = {side: _numeric(frame, f"cop_{side}_x") for side in ("left", "right")}
    foot_y = {side: _numeric(frame, f"cop_{side}_y") for side in ("left", "right")}
    total_x = _numeric(frame, "cop_total_x")
    total_y = _numeric(frame, "cop_total_y")
    intervals: dict[str, list[tuple[int, int, str]]] = {}
    sources: dict[str, str] = {}
    contacts: list[dict[str, Any]] = []
    for side in ("left", "right"):
        intervals[side], sources[side] = _contact_segments(
            frame, side, force[side], force_threshold
        )
        for number, (start, stop, source_id) in enumerate(intervals[side], 1):
            before = start - 1
            after = stop
            before_force = (
                float(force[side][before])
                if before >= 0 and np.isfinite(force[side][before])
                else np.nan
            )
            after_force = (
                float(force[side][after])
                if after < len(force[side]) and np.isfinite(force[side][after])
                else np.nan
            )
            start_valid = bool(
                np.isfinite(before_force) and abs(before_force) <= zero_tolerance
            )
            end_valid = bool(
                np.isfinite(after_force) and abs(after_force) <= zero_tolerance
            )
            duration = (
                float(time_values[stop - 1] - time_values[start])
                if stop - start >= 2
                else 0.0
            )
            contact_time = time_values[start:stop]
            local_x = foot_x[side][start:stop].copy()
            local_y = foot_y[side][start:stop].copy()
            finite_x = np.flatnonzero(np.isfinite(local_x))
            finite_y = np.flatnonzero(np.isfinite(local_y))
            if len(finite_x):
                local_x -= local_x[finite_x[0]]
            if len(finite_y):
                local_y -= local_y[finite_y[0]]
            local_total_x = total_x[start:stop].copy()
            local_total_y = total_y[start:stop].copy()
            finite_total_x = np.flatnonzero(np.isfinite(local_total_x))
            finite_total_y = np.flatnonzero(np.isfinite(local_total_y))
            if len(finite_total_x):
                local_total_x -= local_total_x[finite_total_x[0]]
            if len(finite_total_y):
                local_total_y -= local_total_y[finite_total_y[0]]
            # Exakt wie V12: nur Total-COP-x linker Kontakte wird gespiegelt.
            if side == "left":
                local_total_x *= -1.0
            speed, acceleration = _dynamics(local_x, local_y, contact_time)
            total_speed, total_acceleration = _dynamics(
                local_total_x, local_total_y, contact_time
            )
            phase = np.stack(
                [
                    _phase_resample(signal, contact_time)
                    for signal in (
                        local_x,
                        local_y,
                        speed,
                        acceleration,
                        local_total_x,
                        local_total_y,
                        total_speed,
                        total_acceleration,
                    )
                ]
            )
            # Die sechs spaeter fold-intern max-normalisierten Rohkurven werden
            # ohne Kontakt-Baselinekorrektur rekonstruiert, wie in V21.
            raw_total_speed, raw_total_acceleration = _dynamics(
                total_x[start:stop], total_y[start:stop], contact_time
            )
            contacts.append(
                {
                    "side": side,
                    "number": number,
                    "start": int(start),
                    "stop": int(stop),
                    "source_id": str(source_id),
                    "n_samples": int(stop - start),
                    "duration_seconds": duration,
                    "zero_start_boundary_index": int(before),
                    "zero_end_boundary_index": int(after),
                    "zero_start_force": before_force,
                    "zero_end_force": after_force,
                    "zero_start_valid": start_valid,
                    "zero_end_valid": end_valid,
                    "zero_boundary_valid": bool(start_valid and end_valid),
                    "phase": phase,
                    "force_phase": _phase_resample(
                        force[side][start:stop], contact_time
                    ),
                    "foot_x_raw_phase": _phase_resample(
                        foot_x[side][start:stop], contact_time
                    ),
                    "total_speed_raw_phase": _phase_resample(
                        raw_total_speed, contact_time
                    ),
                    "total_acceleration_raw_phase": _phase_resample(
                        raw_total_acceleration, contact_time
                    ),
                }
            )
    contacts.sort(
        key=lambda item: (
            item["start"],
            0 if item["side"] == "left" else 1,
            item["stop"],
        )
    )
    duration_qc = _apply_contact_duration_qc(contacts, duration_tolerance)

    measurement_score: dict[str, float] = {}
    for side in ("left", "right"):
        mean_phase = _nanmean_curves(
            [
                contact["phase"]
                for contact in contacts
                if contact["side"] == side and contact["contact_valid"]
            ],
            (8, POINTS_PER_PAIR),
        )
        components = [_phase_integral(mean_phase[row]) for row in (2, 3)]
        measurement_score[side] = float(
            sum(value for value in components if np.isfinite(value))
        )
    return {
        "subject": subject,
        "control_label": control,
        "angle": float(angle),
        "curve_type": curve_type,
        "source_file": path.name,
        "source_path": str(path),
        "time": time_values,
        "sampling_rate": sampling_rate,
        "time_source": time_source,
        "force": force,
        "foot_x": foot_x,
        "foot_y": foot_y,
        "total_x": total_x,
        "total_y": total_y,
        "contacts": contacts,
        "contact_sources": sources,
        "contact_duration_qc": duration_qc,
        "measurement_score": measurement_score,
    }


def angle_bin_index(angle_deg: float) -> int:
    return int(math.floor(float(angle_deg) / ANGLE_BIN_WIDTH_DEG))


def angle_bin_bounds(bin_index: int) -> tuple[float, float]:
    lower = float(bin_index) * ANGLE_BIN_WIDTH_DEG
    return lower, lower + ANGLE_BIN_WIDTH_DEG


def angle_bin_label(bin_index: int) -> str:
    lower, upper = angle_bin_bounds(bin_index)
    return f"{lower:g}–<{upper:g}°"


def balanced_angle_fold_indices(
    records: Sequence[SubjectRecord],
    requested_splits: int,
    seed: int,
    candidates: int = 512,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Subjektweise 5-Grad-balancierte Foldzuordnung."""
    n_subjects = len(records)
    folds = min(int(requested_splits), n_subjects)
    if folds < 2:
        raise ValueError(f"Zu wenige Patienten fuer CV: n={n_subjects}")
    angles = np.asarray([item.angle_deg for item in records], np.float64)
    bins = np.asarray([angle_bin_index(value) for value in angles], np.int64)
    hard = np.asarray([item.hard_reference for item in records], np.int64)
    type_names = sorted({item.curve_type for item in records})
    type_lookup = {name: index for index, name in enumerate(type_names)}
    type_codes = np.asarray([type_lookup[item.curve_type] for item in records])
    unique_bins = np.unique(bins)
    positions_by_bin = {
        int(value): np.flatnonzero(bins == value) for value in unique_bins
    }
    best_score: tuple[Any, ...] | None = None
    best_assignment: np.ndarray | None = None
    for candidate in range(max(int(candidates), 1)):
        generator = np.random.default_rng(
            stable_seed(seed, "angle_bin_candidate", candidate)
        )
        assignment = np.full(n_subjects, -1, np.int64)
        fold_total = np.zeros(folds, np.int64)
        fold_hard = np.zeros((folds, 2), np.int64)
        fold_type = np.zeros((folds, len(type_names)), np.int64)
        fold_bin = {
            int(value): np.zeros(folds, np.int64) for value in unique_bins
        }
        bin_ties = {int(value): float(generator.random()) for value in unique_bins}
        ordered_bins = sorted(
            (int(value) for value in unique_bins),
            key=lambda value: (-len(positions_by_bin[value]), bin_ties[value]),
        )
        for bin_value in ordered_bins:
            local_positions = positions_by_bin[bin_value].copy()
            generator.shuffle(local_positions)
            for position in local_positions:
                counts = fold_bin[bin_value]
                eligible = np.flatnonzero(counts == counts.min())
                random_ties = {
                    int(fold): float(generator.random()) for fold in eligible
                }
                label = int(hard[position])
                type_code = int(type_codes[position])
                chosen = min(
                    (int(fold) for fold in eligible),
                    key=lambda fold: (
                        int(fold_total[fold]),
                        int(fold_hard[fold, label]),
                        int(fold_type[fold, type_code]),
                        random_ties[fold],
                    ),
                )
                assignment[position] = chosen
                fold_total[chosen] += 1
                fold_hard[chosen, label] += 1
                fold_type[chosen, type_code] += 1
                fold_bin[bin_value][chosen] += 1
        global_hard = np.bincount(hard, minlength=2)
        empty_class_penalty = int(
            sum(
                np.count_nonzero(fold_hard[:, label] == 0)
                for label in (0, 1)
                if global_hard[label] >= folds
            )
        )
        fold_means = np.asarray(
            [angles[assignment == fold].mean() for fold in range(folds)]
        )
        score = (
            empty_class_penalty,
            int(np.ptp(fold_total)),
            int(sum(np.ptp(fold_hard[:, label]) for label in (0, 1))),
            int(
                sum(
                    np.ptp(fold_type[:, code]) for code in range(len(type_names))
                )
            ),
            float(np.std(fold_means)),
            candidate,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment.copy()
    if best_assignment is None:
        raise RuntimeError("Keine gueltige Foldzuordnung gefunden.")
    indices = np.arange(n_subjects, dtype=np.int64)
    result = []
    for fold in range(folds):
        test = indices[best_assignment == fold]
        train = indices[best_assignment != fold]
        if not len(train) or not len(test):
            raise RuntimeError("Leerer Trainings- oder Testfold.")
        result.append((train, test))
    for bin_value in unique_bins:
        counts = [int(np.sum(bins[test] == bin_value)) for _, test in result]
        if max(counts) - min(counts) > 1:
            raise RuntimeError(
                f"5-Grad-Balance verletzt fuer {angle_bin_label(int(bin_value))}."
            )
    return result


def make_repeated_splits(
    records: Sequence[SubjectRecord], repeats: int, cv_folds: int, seed: int
) -> list[CVSplit]:
    subjects = np.asarray([item.subject for item in records], dtype=str)
    result: list[CVSplit] = []
    for repeat in range(1, int(repeats) + 1):
        local = balanced_angle_fold_indices(
            records,
            cv_folds,
            stable_seed(seed, "repeated_cv", repeat),
        )
        for fold, (train, test) in enumerate(local, start=1):
            result.append(
                CVSplit(
                    repeat,
                    fold,
                    tuple(subjects[train]),
                    tuple(subjects[test]),
                )
            )
    return result


class FoldInputNormalizer:
    """Train-only Imputation; V38 verwendet ausschliesslich SI, raw_indices=()."""

    def __init__(
        self,
        raw_indices: Sequence[int] = RAW_COP_INDICES,
        clip_z: float = CLIP_Z,
    ):
        self.raw_indices = tuple(int(index) for index in raw_indices)
        self.clip_z = float(clip_z)
        self.raw_denominators_: np.ndarray | None = None
        self.impute_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @staticmethod
    def _all_pairs(record: SubjectRecord) -> np.ndarray:
        return np.concatenate(
            [np.asarray(item.pairs, np.float64) for item in record.measurements],
            axis=0,
        )

    def _max_normalize(self, pairs: np.ndarray) -> np.ndarray:
        if self.raw_denominators_ is None:
            raise RuntimeError("Normalizer ist noch nicht angepasst.")
        values = np.asarray(pairs, np.float64).copy()
        values[:, self.raw_indices, :] /= self.raw_denominators_[
            None, self.raw_indices, None
        ]
        return values

    def fit(self, records: Sequence[SubjectRecord]) -> "FoldInputNormalizer":
        if not records:
            raise ValueError("Kein Trainingspatient fuer die Normierung.")
        n_channels = self._all_pairs(records[0]).shape[1]
        self.raw_denominators_ = np.ones(n_channels, np.float64)
        for index in self.raw_indices:
            maxima = []
            for record in records:
                values = self._all_pairs(record)[:, index, :]
                finite = np.abs(values[np.isfinite(values)])
                if len(finite):
                    maxima.append(float(np.max(finite)))
            denominator = max(maxima) if maxima else 1.0
            if not np.isfinite(denominator) or denominator <= 1e-12:
                denominator = 1.0
            self.raw_denominators_[index] = denominator
        patient_medians = []
        for record in records:
            values = self._max_normalize(self._all_pairs(record))
            flat = values.transpose(0, 2, 1).reshape(-1, n_channels)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                patient_medians.append(np.nanmedian(flat, axis=0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.impute_ = np.nanmedian(np.stack(patient_medians), axis=0)
        self.impute_ = np.where(np.isfinite(self.impute_), self.impute_, 0.0)
        # Kompatibilitaetsattribute bleiben erhalten, bilden aber bewusst die
        # Identitaet ab. Es wird weder zentriert noch durch eine Streuung
        # geteilt und es findet kein Z-Clipping statt.
        self.mean_ = np.zeros(n_channels, dtype=np.float64)
        self.scale_ = np.ones(n_channels, dtype=np.float64)
        return self

    def transform_pairs(self, pairs: np.ndarray) -> np.ndarray:
        if self.impute_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Normalizer ist noch nicht angepasst.")
        values = self._max_normalize(pairs)
        values = np.where(np.isfinite(values), values, self.impute_[None, :, None])
        return values.astype(np.float32)


def load_standalone_si15_records(
    data_dir: str | Path,
    force_threshold: float = 0.0,
    contact_duration_tolerance: float = CONTACT_DURATION_TOLERANCE_FRACTION,
    contact_zero_tolerance: float = CONTACT_ZERO_TOLERANCE,
    min_si_pairs_per_subject: int = MIN_SI_PAIRS_PER_SUBJECT,
    budget=None,
) -> tuple[list[SubjectRecord], dict[str, Any], pd.DataFrame, str]:
    """Rekonstruiert SI15 wie V29.10 und gibt die feste SI04/SI05/SI09-Teilmenge aus."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Pfad '{root}' existiert nicht.")
    files = (
        sorted(
            [path for path in root.rglob("*.csv") if not path.name.startswith("~$")],
            key=lambda path: str(path).lower(),
        )
        if root.is_dir()
        else [root]
    )
    print(
        f"📂 {len(files)} CSV-Dateien gefunden. Eigenstaendiger SI15→SI04/SI05/SI09-Import mit "
        f"Kontaktdauer-QC (±{contact_duration_tolerance * 100:g}% vom "
        f"Messungsmittel), Nullrand-Toleranz={contact_zero_tolerance:g} ..."
    )
    raw_measurements: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in files:
        if budget is not None:
            budget.check()
        try:
            raw_measurements.append(
                _prepare_raw_measurement(
                    path,
                    float(force_threshold),
                    float(contact_duration_tolerance),
                    float(contact_zero_tolerance),
                )
            )
        except Exception as error:
            failures.append({"source_file": path.name, "error": str(error)})
            print(f"❌ Datei nicht lesbar: {path.name}: {error}")
    if not raw_measurements:
        raise ValueError("Keine Messung konnte gelesen werden.")

    patient_scores: dict[str, dict[str, float]] = {}
    for measurement in raw_measurements:
        scores = patient_scores.setdefault(
            measurement["subject"], {"left": 0.0, "right": 0.0}
        )
        for side in ("left", "right"):
            scores[side] += float(measurement["measurement_score"][side])
    dominant_side = {
        subject: ("right" if score["right"] >= score["left"] else "left")
        for subject, score in patient_scores.items()
    }
    right_count = sum(side == "right" for side in dominant_side.values())
    ties = sum(
        score["right"] == score["left"] for score in patient_scores.values()
    )
    print(
        f"🦶 Dominanter Fuss pro Patient: rechts={right_count} | "
        f"links={len(dominant_side) - right_count} | Patienten={len(dominant_side)}"
    )
    if ties:
        print(f"   Gleichstand: {ties} Patient(en) rechts zugeordnet.")

    subject_targets: dict[str, dict[str, Any]] = {}
    for subject in sorted(patient_scores):
        local = [item for item in raw_measurements if item["subject"] == subject]
        types = sorted({str(item["curve_type"]).upper() for item in local})
        if len(types) != 1:
            raise ValueError(f"Patient '{subject}' besitzt mehrere Typen: {types}")
        angle = float(np.mean([item["angle"] for item in local]))
        boundary = float(ANGLE_CLASS_THRESHOLDS_DEG[types[0]])
        subject_targets[subject] = {
            "angle": angle,
            "curve_type": types[0],
            "boundary": boundary,
            "soft_target": float(probability_above_boundary(angle, boundary)),
            "hard_reference": int(angle > boundary),
        }

    contact_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    measurement_objects: dict[str, list[Measurement]] = {
        subject: [] for subject in subject_targets
    }
    candidate_total = 0
    rejected_total = 0
    for measurement in raw_measurements:
        subject = measurement["subject"]
        patient_dominant = dominant_side[subject]
        duration_qc = measurement["contact_duration_qc"]
        contacts = measurement["contacts"]
        for chronological, contact in enumerate(contacts, 1):
            contact_rows.append(
                {
                    "source_file": measurement["source_file"],
                    "subject": subject,
                    "chronological_contact_number": chronological,
                    "side": contact["side"],
                    "side_contact_number": contact["number"],
                    "source_contact_id": contact["source_id"],
                    "start_index": contact["start"],
                    "stop_index_exclusive": contact["stop"],
                    "n_samples": contact["n_samples"],
                    "duration_seconds": contact["duration_seconds"],
                    "zero_start_boundary_index": contact[
                        "zero_start_boundary_index"
                    ],
                    "zero_end_boundary_index": contact["zero_end_boundary_index"],
                    "zero_start_force": contact["zero_start_force"],
                    "zero_end_force": contact["zero_end_force"],
                    "zero_start_valid": contact["zero_start_valid"],
                    "zero_end_valid": contact["zero_end_valid"],
                    "zero_boundary_valid": contact["zero_boundary_valid"],
                    "measurement_mean_duration_seconds": duration_qc["mean_seconds"],
                    "allowed_lower_duration_seconds": duration_qc["lower_seconds"],
                    "allowed_upper_duration_seconds": duration_qc["upper_seconds"],
                    "duration_deviation_fraction": contact[
                        "duration_deviation_fraction"
                    ],
                    "duration_within_tolerance": contact[
                        "duration_within_tolerance"
                    ],
                    "contact_valid": contact["contact_valid"],
                    "duration_qc_reason": contact["duration_qc_reason"],
                }
            )

        pairs: list[np.ndarray] = []
        local_candidates = 0
        local_rejected = 0
        for index in range(1, len(contacts)):
            previous, current = contacts[index - 1], contacts[index]
            if (
                previous["side"] == current["side"]
                or previous["start"] == current["start"]
            ):
                continue
            local_candidates += 1
            candidate_total += 1
            accepted = bool(previous["contact_valid"] and current["contact_valid"])
            if not accepted:
                local_rejected += 1
                rejected_total += 1
                pair_rows.append(
                    {
                        "source_file": measurement["source_file"],
                        "subject": subject,
                        "candidate_number": local_candidates,
                        "previous_side": previous["side"],
                        "current_side": current["side"],
                        "accepted": False,
                        "rejection_reason": "contact_qc",
                    }
                )
                continue
            if current["side"] == patient_dominant:
                dominant_contact, other_contact = current, previous
            else:
                dominant_contact, other_contact = previous, current
            base_si = _symmetry_index(
                dominant_contact["phase"], other_contact["phase"]
            )
            force_si = _symmetry_index(
                dominant_contact["force_phase"][None, :],
                other_contact["force_phase"][None, :],
            )[0]
            source_pair = np.concatenate(
                [
                    base_si,
                    np.stack(
                        [
                            force_si,
                            dominant_contact["foot_x_raw_phase"],
                            other_contact["foot_x_raw_phase"],
                            dominant_contact["total_speed_raw_phase"],
                            other_contact["total_speed_raw_phase"],
                            dominant_contact["total_acceleration_raw_phase"],
                            other_contact["total_acceleration_raw_phase"],
                        ]
                    ),
                ],
                axis=0,
            ).astype(np.float32)
            if source_pair.shape != (SOURCE_CHANNELS, POINTS_PER_PAIR):
                raise RuntimeError(
                    f"Interne SI15-Quellform ist {source_pair.shape}."
                )
            pair = source_pair[
                np.asarray(SELECTED_SOURCE_CHANNEL_INDICES, dtype=np.int64)
            ].copy()
            if pair.shape != (EXPECTED_CHANNELS, POINTS_PER_PAIR):
                raise RuntimeError(f"Interne SI04/SI05/SI09-Paarform ist {pair.shape}.")
            pairs.append(pair)
            pair_rows.append(
                {
                    "source_file": measurement["source_file"],
                    "subject": subject,
                    "candidate_number": local_candidates,
                    "pair_number": len(pairs),
                    "previous_side": previous["side"],
                    "current_side": current["side"],
                    "dominant_side": patient_dominant,
                    "dominant_contact_number": dominant_contact["number"],
                    "other_contact_number": other_contact["number"],
                    "accepted": True,
                    "rejection_reason": "accepted",
                }
            )
        if pairs:
            measurement_objects[subject].append(
                Measurement(
                    measurement_id=measurement["source_file"],
                    source_file=measurement["source_file"],
                    subject=subject,
                    pairs=np.stack(pairs),
                )
            )
        measurement_rows.append(
            {
                "source_file": measurement["source_file"],
                "subject": subject,
                "angle_deg": measurement["angle"],
                "curve_type": measurement["curve_type"],
                "dominant_side": patient_dominant,
                "sampling_rate_hz": measurement["sampling_rate"],
                "time_source": measurement["time_source"],
                "contacts_total": duration_qc["n_contacts"],
                "contacts_valid_after_all_qc": duration_qc["n_valid"],
                "contacts_rejected": duration_qc["n_rejected"],
                "candidate_opposite_consecutive_pairs": local_candidates,
                "pairs_rejected_by_contact_qc": local_rejected,
                "opposite_consecutive_pairs": len(pairs),
                "status": (
                    "used"
                    if pairs
                    else (
                        "all_candidate_pairs_rejected_by_contact_qc"
                        if local_candidates and local_rejected == local_candidates
                        else "no_consecutive_opposite_pair"
                    )
                ),
            }
        )

    pair_counts = {
        subject: int(sum(len(item.pairs) for item in local))
        for subject, local in measurement_objects.items()
    }
    retained = {
        subject
        for subject, count in pair_counts.items()
        if count >= int(min_si_pairs_per_subject)
    }
    subject_qc = pd.DataFrame(
        [
            {
                "subject": subject,
                "valid_si_pairs_after_contact_qc": pair_counts[subject],
                "minimum_required_si_pairs": int(min_si_pairs_per_subject),
                "subject_retained": subject in retained,
                "exclusion_reason": (
                    "retained"
                    if subject in retained
                    else "fewer_than_minimum_valid_si_pairs"
                ),
                "dominant_side": dominant_side[subject],
                "patient_target_angle_deg": subject_targets[subject]["angle"],
                "curve_type": subject_targets[subject]["curve_type"],
                "derived_hard_label": subject_targets[subject]["hard_reference"],
                "target_probability_label1": subject_targets[subject]["soft_target"],
            }
            for subject in sorted(subject_targets)
        ]
    )
    records = [
        SubjectRecord(
            subject=subject,
            angle_deg=float(subject_targets[subject]["angle"]),
            curve_type=str(subject_targets[subject]["curve_type"]),
            boundary_deg=float(subject_targets[subject]["boundary"]),
            soft_target=float(subject_targets[subject]["soft_target"]),
            hard_reference=int(subject_targets[subject]["hard_reference"]),
            measurements=measurement_objects[subject],
        )
        for subject in sorted(retained)
    ]
    if not records:
        raise ValueError(
            "Keine Person besitzt mindestens "
            f"{int(min_si_pairs_per_subject)} gueltige SI-Paare."
        )
    hard = np.asarray([record.hard_reference for record in records], np.int64)
    if len(np.unique(hard)) != 2:
        raise ValueError("Nach der Personen-QC sind nicht beide Klassen vorhanden.")

    dataset = {
        "contact_qc": pd.DataFrame(contact_rows),
        "pair_qc": pd.DataFrame(pair_rows),
        "measurement_qc": pd.DataFrame(measurement_rows),
        "subject_qc": subject_qc,
        "import_failures": pd.DataFrame(failures),
        "dominant_side": dominant_side,
    }
    raw_audit = pd.DataFrame(
        [
            {
                "input_mode": "standalone_original_si15_contact_pairs",
                "external_python_dependencies_v12_v21_v22": False,
                "files_found": len(files),
                "files_loaded": len(raw_measurements),
                "files_failed": len(failures),
                "subjects_before_pair_qc": len(subject_targets),
                "subjects_retained": len(records),
                "measurements_retained": sum(
                    len(record.measurements) for record in records
                ),
                "si_pairs_retained": sum(record.n_pairs for record in records),
                "candidate_opposite_consecutive_pairs": candidate_total,
                "pairs_rejected_by_contact_qc": rejected_total,
                "contact_resampling": "each contact separately to 101 points",
                "pair_definition": "chronologically consecutive opposite-side contacts",
                "si_denominator": "amplitude",
                "si_formula": (
                    "2*(dominant-other)/(dominant_amplitude+other_amplitude)"
                ),
                "si_percent_scale": False,
                "total_cop_x_mirrored_side": "left",
                "source_channels_json": json.dumps(SOURCE_SI15_CHANNELS),
                "selected_source_channel_indices_json": json.dumps(
                    SELECTED_SOURCE_CHANNEL_INDICES
                ),
                "channels_json": json.dumps(SELECTED_INPUT_CHANNELS),
            }
        ]
    )
    per_measurement = [
        row["opposite_consecutive_pairs"] for row in measurement_rows
    ]
    print(
        f"✅ Messungen geladen: {len(raw_measurements)} / {len(files)} | "
        f"SI-Paare nach Personen-QC: {sum(record.n_pairs for record in records)}"
    )
    print(
        f"   Paare je Messung: min={min(per_measurement)}, "
        f"median={np.median(per_measurement):.1f}, max={max(per_measurement)}"
    )
    print(
        f"👤 Personen-QC: mindestens {int(min_si_pairs_per_subject)} gueltige "
        f"SI-Paare | behalten={len(records)}/{len(subject_targets)}"
    )
    print(
        f"   Harte Splitreferenz: n0={int(np.sum(hard == 0))}, "
        f"n1={int(np.sum(hard == 1))}"
    )
    return records, dataset, raw_audit, "standalone_original_si15"


def section_balanced_accuracy_details_v13(
    truth: np.ndarray,
    prediction: np.ndarray,
    angles: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    truth = np.asarray(truth, np.int64)
    prediction = np.asarray(prediction, np.int64)
    angles = np.asarray(angles, np.float64)
    if not (len(truth) == len(prediction) == len(angles)) or not len(truth):
        raise ValueError("Abschnittsmetrik erhielt leere/inkonsistente Vektoren.")
    bins = np.floor(angles / ANGLE_BIN_WIDTH_DEG).astype(np.int64)
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for bin_index in sorted(np.unique(bins)):
        local = bins == bin_index
        recalls: dict[int, float] = {}
        for class_value in sorted(np.unique(truth[local])):
            mask = local & (truth == int(class_value))
            recalls[int(class_value)] = float(
                np.mean(prediction[mask] == int(class_value))
            )
        score = float(np.mean(list(recalls.values())))
        scores.append(score)
        rows.append(
            {
                "angle_bin_index": int(bin_index),
                "angle_bin_low_deg": float(
                    bin_index * ANGLE_BIN_WIDTH_DEG
                ),
                "angle_bin_high_deg_exclusive": float(
                    (bin_index + 1) * ANGLE_BIN_WIDTH_DEG
                ),
                "n_subjects": int(local.sum()),
                "n_true_class_0": int(np.sum(local & (truth == 0))),
                "n_true_class_1": int(np.sum(local & (truth == 1))),
                "recall_class_0": recalls.get(0, np.nan),
                "recall_class_1": recalls.get(1, np.nan),
                "section_balanced_accuracy": score,
            }
        )
    return float(np.mean(scores)), rows


def safe_ratio_v13(
    numerator: float,
    denominator: float,
    zero_division: float = np.nan,
) -> float:
    if float(denominator) == 0.0:
        return float(zero_division)
    return float(numerator) / float(denominator)


def binary_roc_auc_v13(truth: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(truth, np.int64)
    scores = np.asarray(scores, np.float64)
    finite = np.isfinite(scores) & np.isin(truth, (0, 1))
    truth = truth[finite]
    scores = scores[finite]
    positives = int(np.sum(truth == 1))
    negatives = int(np.sum(truth == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(np.float64)
    positive_rank_sum = float(np.sum(ranks[truth == 1]))
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def classification_metrics_v13(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        raise ValueError("Metriken koennen nicht aus einer leeren Tabelle entstehen.")
    truth = frame["true_class"].to_numpy(np.int64)
    prediction = frame["predicted_class"].to_numpy(np.int64)
    probability = frame["probability_class_1"].to_numpy(np.float64)
    angles = frame["true_angle_deg"].to_numpy(np.float64)
    section_ba, _ = section_balanced_accuracy_details_v13(
        truth, prediction, angles
    )
    true_positive = int(np.sum((truth == 1) & (prediction == 1)))
    true_negative = int(np.sum((truth == 0) & (prediction == 0)))
    false_positive = int(np.sum((truth == 0) & (prediction == 1)))
    false_negative = int(np.sum((truth == 1) & (prediction == 0)))
    sensitivity = safe_ratio_v13(
        true_positive, true_positive + false_negative
    )
    specificity = safe_ratio_v13(
        true_negative, true_negative + false_positive
    )
    balanced_accuracy = float(np.nanmean([specificity, sensitivity]))
    precision = safe_ratio_v13(
        true_positive,
        true_positive + false_positive,
        zero_division=0.0,
    )
    negative_predictive_value = safe_ratio_v13(
        true_negative,
        true_negative + false_negative,
        zero_division=0.0,
    )
    f1 = safe_ratio_v13(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
        zero_division=0.0,
    )
    mcc_denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = safe_ratio_v13(
        true_positive * true_negative - false_positive * false_negative,
        mcc_denominator,
        zero_division=0.0,
    )
    angle_mae = float("nan")
    angle_rmse = float("nan")
    if "prediction_deg" in frame.columns:
        predicted_angle = pd.to_numeric(
            frame["prediction_deg"], errors="coerce"
        ).to_numpy(np.float64)
        finite_angle = np.isfinite(predicted_angle) & np.isfinite(angles)
        if np.any(finite_angle):
            error = predicted_angle[finite_angle] - angles[finite_angle]
            angle_mae = float(np.mean(np.abs(error)))
            angle_rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "n_subjects": int(len(frame)),
        "n_true_class_0": int(np.sum(truth == 0)),
        "n_true_class_1": int(np.sum(truth == 1)),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "section_balanced_accuracy": section_ba,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": binary_roc_auc_v13(truth, probability),
        "sensitivity_class_1": sensitivity,
        "specificity_class_0": specificity,
        "accuracy": safe_ratio_v13(
            true_positive + true_negative, len(truth)
        ),
        "precision_class_1": precision,
        "negative_predictive_value_class_0": negative_predictive_value,
        "f1_class_1": f1,
        "matthews_correlation_coefficient": mcc,
        "angle_mae_deg": angle_mae,
        "angle_rmse_deg": angle_rmse,
        "hard_brier": float(np.mean((probability - truth) ** 2)),
    }


VERSION = "38.1.1"
LEGACY_V3810_SOURCE_SHA256 = "ebe4eca28157ee99d0808ebd3348d32ef005ba1a3f02979deef1fa5cdf77c576"


METRICS = ("balanced_accuracy", "section_balanced_accuracy", "accuracy", "cohen_kappa", "roc_auc", "sensitivity_class_1", "specificity_class_0", "f1_class_1", "angle_mae_deg", "angle_rmse_deg")


ModuleBase = nn.Module if nn is not None else object


def log(message: str) -> None:
    print(message, flush=True)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True,
                                    ensure_ascii=True).encode()).hexdigest()


def atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        writer(temporary)
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise PermissionError(
                        f"Kann {path} nicht ersetzen. Datei in Excel/anderen Programmen schliessen."
                    ) from None
                time.sleep(min(0.1 * 2 ** attempt, 1.0))
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, lambda p: p.write_text(
        json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8"))


def atomic_csv(path: Path, rows: Any, columns=None) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    atomic_write(path, lambda p: frame.to_csv(p, sep=";", index=False, encoding="utf-8-sig"))


def atomic_torch(path: Path, value: Any) -> None:
    def compatible(item):
        # weights_only=True soll auch teilweise fertige Epochen laden koennen.
        # NumPy-Metrikskalare daher als Pythonwerte, Optimierer-INT-Keys behalten.
        if isinstance(item, dict):
            return {k: compatible(v) for k, v in item.items()}
        if isinstance(item, tuple):
            return tuple(compatible(v) for v in item)
        if isinstance(item, list):
            return [compatible(v) for v in item]
        if isinstance(item, np.ndarray):
            return compatible(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        return item
    atomic_write(path, lambda p: torch.save(compatible(value), p))


def require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch fehlt. Die passende CUDA-/XPU-Version in diesem Python installieren.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def choose_device(requested: str) -> str:
    require_torch()
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            if torch.version.cuda is None:
                raise RuntimeError(
                    f"CUDA angefordert, aber PyTorch {torch.__version__} in {sys.executable} "
                    "enthaelt keinen CUDA-Build. CUDA-torch und dazu passendes torchvision "
                    "wiederherstellen; Befehl im Dateikopf.")
            raise RuntimeError(
                f"CUDA angefordert; PyTorch {torch.__version__}, CUDA-Build {torch.version.cuda}, "
                f"Python {sys.executable}. torch.cuda.is_available() ist False. "
                "Der CUDA-Build ist vorhanden; NVIDIA-Treiber/GPU-Sichtbarkeit pruefen.")
        log(f"NVIDIA: {torch.cuda.get_device_name(device)} | PyTorch {torch.__version__}")
    elif device.type == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU angefordert, aber torch.xpu.is_available() ist False.")
        log(f"Intel XPU: {torch.xpu.get_device_name(device)} | PyTorch {torch.__version__}")
    elif device.type != "cpu":
        raise ValueError("Unterstuetzte Geraete: xpu, cuda[:Index], cpu.")
    return str(device)


def pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                raise OSError("Prozessstatus konnte nicht geprueft werden.")
            return code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def output_lock(output: Path, resume: bool):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run.lock"
    token = uuid.uuid4().hex
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except (ValueError, OSError) as error:
            raise RuntimeError(f"Unlesbare Laufsperre: {path}") from error
        if old.get("host") != socket.gethostname() or pid_running(int(old.get("pid", 0))):
            raise RuntimeError(f"Ergebnisordner wird von einem anderen Prozess verwendet: {output}")
        if not resume:
            raise RuntimeError("Ein unterbrochener Lauf ist vorhanden; zum Fortsetzen --resume verwenden.")
        path.unlink()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(), "token": token}, handle)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text()).get("token") == token:
            path.unlink()


def dataset_fingerprint(records: Sequence[SubjectRecord]) -> str:
    hasher = hashlib.sha256()
    for record in records:
        # Nur der Hash wird geschrieben; keine Kennung, kein Einzelwinkel.
        hasher.update(record.subject.encode())
        hasher.update(np.asarray([record.angle_deg, record.boundary_deg,
                                 record.hard_reference], dtype="<f8").tobytes())
        hasher.update(record.curve_type.encode())
        hasher.update(np.asarray(record.pairs, dtype="<f4").tobytes())
    return hasher.hexdigest()


def rare_angle_copy_counts(records, cap: int, seed: int) -> np.ndarray:
    bins = np.asarray([angle_bin_index(r.angle_deg) for r in records])
    unique, counts = np.unique(bins, return_counts=True)
    copies = np.zeros(len(records), dtype=int)
    generator = np.random.default_rng(seed)
    for value, count in zip(unique, counts):
        positions = np.flatnonzero(bins == value)
        deficit = int(counts.max() - count)
        for _ in range(cap):
            for position in generator.permutation(positions):
                if deficit <= 0:
                    break
                copies[position] += 1
                deficit -= 1
    return copies


def metrics_for(records, values, task_id="angle_regression", threshold=.5):
    if task_id != "angle_regression":
        raise ValueError("V38 wertet ausschliesslich Winkelregression aus.")
    values = np.asarray(values, dtype=float)
    if values.shape != (len(records),) or not np.isfinite(values).all():
        raise ValueError("Metriken brauchen eine endliche Winkelprognose je Person.")
    truth = np.asarray([r.hard_reference for r in records], int)
    if set(truth) != {0, 1}:
        raise ValueError("Jeder Auswertungsfold muss beide Klassen enthalten.")
    angles = np.asarray([r.angle_deg for r in records], float)
    boundary = np.asarray([r.boundary_deg for r in records], float)
    prediction = (values > boundary).astype(int)
    frame = pd.DataFrame({"true_class": truth, "true_angle_deg": angles, "predicted_class": prediction,
                          "probability_class_1": probability_above_boundary(values, boundary), "prediction_deg": values})
    result = classification_metrics_v13(frame)
    expected = sum(np.mean(truth == label)*np.mean(prediction == label) for label in (0, 1))
    result["cohen_kappa"] = float((result["accuracy"]-expected)/(1-expected)) if expected < 1 else float("nan")
    if not np.isfinite(result["cohen_kappa"]):
        raise ValueError("Cohen's Kappa ist undefiniert.")
    return result


def ranking(metrics: dict, criterion: str) -> tuple:
    def finite(name):
        value = metrics.get(name)
        return float(value) if value is not None and np.isfinite(value) else -math.inf
    names = list(dict.fromkeys((criterion, "balanced_accuracy", "section_balanced_accuracy", "accuracy")))
    return tuple(finite(name) for name in names)


def split_count_rows(records, indices, seed, fold, role, inner_fold=0):
    counts = Counter((angle_bin_index(records[i].angle_deg), records[i].hard_reference,
                      records[i].curve_type) for i in indices)
    return [{"split_seed": seed, "outer_fold": fold, "inner_fold": inner_fold,
             "role": role, "angle_bin_low_deg": bin_index * 5.,
             "angle_bin_high_deg_exclusive": (bin_index + 1) * 5.,
             "true_class": label, "curve_type": curve_type, "n_persons": count}
            for (bin_index, label, curve_type), count in sorted(counts.items())]


WEIGHTS_NAME = "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"


CHANNEL_IDS = ("SI04", "SI05", "SI09")

CHANNEL_SETS = {
    "_".join(CHANNEL_IDS[i] for i in indices): tuple(indices)
    for n in (1, 2, 3) for indices in itertools.combinations(range(3), n)
}

CONDITIONS = tuple(s + "__" + a + "_angle_regression__frozen"
                   for s in CHANNEL_SETS for a in ("cnn", "cnn_transformer"))


def encoder_parameter_formula(width, ff):
    # Q/K/V und Ausgabeprojektion, zwei FF-Linears, zwei LayerNorms; mit Bias.
    return 4 * width ** 2 + 2 * width * ff + 9 * width + ff


@dataclass(frozen=True)
class TransferConfig:
    architecture: str
    task: str = "angle_regression"
    transfer_mode: str = "frozen"
    image_width: int = 160
    learning_rate: float = .001
    backbone_lr_fraction: float = 0.
    weight_decay: float = .001
    head_dropout: float = .1
    feature_dropout: float = .1
    head_hidden: int = 0
    class_weight: str = "none"
    rare_angle_copy_cap: int = 3
    warmup_epochs: int = 0
    huber_beta: float = 1.
    transformer_width: int = 0
    attention_heads: int = 0
    transformer_dropout: float = 0.
    feedforward_multiplier: int = 0
    tokens: int = 0
    stochastic_depth: float = 0.
    channel_set: str = "SI04_SI05_SI09"

    def __post_init__(self):
        if self.task != "angle_regression" or self.transfer_mode != "frozen":
            raise ValueError("Nur Winkelregression mit eingefrorenem CNN erlaubt.")
        if self.channel_set not in CHANNEL_SETS or self.architecture not in ("cnn", "cnn_transformer"):
            raise ValueError("Unbekanntes Kanalset oder Modell.")
        if self.image_width != 160 or self.huber_beta <= 0 or self.warmup_epochs:
            raise ValueError("V38 verwendet feste Bildbreite 160, beta>0 und kein CNN-Warm-up.")
        if self.architecture == "cnn_transformer" and (
                self.transformer_width not in (8, 16, 24) or self.attention_heads not in (1, 2, 4)
                or self.transformer_width % self.attention_heads or self.tokens not in (2, 4, 5)):
            raise ValueError("Ungueltiger kleiner Transformer.")

    @property
    def condition(self):
        return self.channel_set + "__" + self.architecture + "_angle_regression__frozen"

    @property
    def template_id(self):
        return "tpl_" + digest({k: v for k, v in asdict(self).items() if k != "channel_set"})[:12]

    @property
    def config_id(self):
        return "cfg_" + digest(asdict(self))[:12]

    def row(self):
        indices = CHANNEL_SETS[self.channel_set]
        d = self.transformer_width
        return {"config_id": self.config_id, "template_id": self.template_id,
                "condition_id": self.condition, "pretrained_source": WEIGHTS_NAME,
                "image_height": 96, "input_channels": len(indices), "fixed_image_slots": 3,
                "source_indices_zero_based": ",".join(str(SELECTED_SOURCE_CHANNEL_INDICES[i]) for i in indices),
                "channel_mask": ",".join(str(int(i in indices)) for i in range(3)),
                "input_points": 101, "covered_input_points": 101,
                "transformer_layers": int(self.architecture == "cnn_transformer"),
                "encoder_parameters_formula": encoder_parameter_formula(d, d*self.feedforward_multiplier) if d else 0,
                **asdict(self)}


def sample_configs(count, seed, max_warmup=0):
    """count is per channel set; identical, nested template pools across all sets."""
    if count < 8 or count % 4:
        raise ValueError("Startzahl je Kanalset muss >=8 und durch vier teilbar sein.")
    templates = []
    for architecture in ("cnn", "cnn_transformer"):
        rng = np.random.default_rng(stable_seed(seed, architecture))
        used = set()
        while len(used) < count//2:
            is_t = architecture == "cnn_transformer"
            baseline = len(used) == 0
            d = 16 if baseline else int(rng.choice([8, 16, 24]))
            cfg = TransferConfig(
                architecture=architecture, learning_rate=.001 if baseline else float(rng.choice([3e-4, 1e-3, 2e-3])),
                weight_decay=.001 if baseline else float(rng.choice([1e-4, 1e-3, 1e-2])),
                head_dropout=.1 if baseline else float(rng.choice([0., .1, .2])),
                feature_dropout=.1 if baseline else float(rng.choice([0., .1, .2])),
                head_hidden=0 if is_t or baseline else int(rng.choice([0, 32])),
                class_weight="none" if baseline else str(rng.choice(["none", "balanced"])),
                rare_angle_copy_cap=3 if baseline else int(rng.choice([3, 4])),
                huber_beta=1. if baseline else float(rng.choice([.5, 1., 2.])),
                transformer_width=d if is_t else 0,
                attention_heads=(2 if baseline else int(rng.choice([1, 2, 4]))) if is_t else 0,
                transformer_dropout=(.1 if baseline else float(rng.choice([0., .1, .2]))) if is_t else 0.,
                feedforward_multiplier=(2 if baseline else int(rng.choice([1, 2]))) if is_t else 0,
                tokens=(4 if baseline else int(rng.choice([2, 4, 5]))) if is_t else 0)
            if cfg.template_id not in used:
                used.add(cfg.template_id)
                templates.append(cfg)
    return [replace(t, channel_set=s) for s in CHANNEL_SETS for t in templates]


def parameter_breakdown(model):
    count = lambda module: sum(p.numel() for p in module.parameters())
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    total = count(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable,
            "frozen_parameters": total - trainable, "cnn_parameters": count(model.backbone),
            "cnn_trainable_parameters": backbone_trainable,
            "encoder_parameters": count(model.encoder) if hasattr(model, "encoder") else 0,
            "projection_parameters": count(model.project) if hasattr(model, "project") else 0,
            "feature_norm_parameters": count(model.feature_norm), "head_parameters": count(model.head),
            "small_component_parameters": trainable - backbone_trainable}


def image_coverage(width):
    """Tatsaechlicher linearer Resize-Operator fuer alle 101 Quellpunkte."""
    eye = torch.eye(101).reshape(101, 1, 101)
    resized = F.interpolate(eye, size=width, mode="linear", align_corners=True)
    return resized.abs().sum((1, 2)).numpy() > 0


def signal_images(pairs, width):
    if pairs.ndim != 3 or pairs.shape[1:] != (3, 101):
        raise ValueError(f"Erwartet [Kontakte,3,101], erhalten {tuple(pairs.shape)}")
    values = .5 + torch.atan(pairs) / math.pi
    values = F.interpolate(values, size=width, mode="linear", align_corners=True)
    band_index = (torch.arange(96, device=pairs.device)//32).long()
    image = values[:, band_index].unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    mean = image.new_tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
    std = image.new_tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
    return (image-mean)/std


class BackboneFactory:
    def __init__(self, args, synthetic=False):
        self.synthetic = synthetic
        if synthetic:
            # Ausschliesslich technischer Selbsttest; niemals Produktions-Fallback.
            seed_everything(918)
            self.features = nn.Sequential(
                nn.Sequential(nn.Conv2d(3, 8, 4, stride=4), nn.GELU()),
                nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.GELU()),
                nn.Conv2d(8, 12, 2, stride=2), nn.GELU(),
                nn.Conv2d(12, 16, 2, stride=2), nn.GELU(),
                nn.Conv2d(16, 16, 2, stride=2), nn.GELU(),
            )
            self.dim = 16
            self.provenance = "SYNTHETIC_TECHNICAL_TEST_NO_PRETRAINED_CLAIM"
        else:
            try:
                from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            except (ImportError, RuntimeError) as error:
                raise RuntimeError(
                    "torchvision fehlt oder passt nicht zum CUDA-Torch. "
                    "Passendes Paar im Dateikopf installieren; vorhandenen CUDA-Build erhalten."
                ) from error
            torch.hub.set_dir(str(args.weights_dir))
            log("Lade offizielle vortrainierte ConvNeXt-Tiny-Gewichte (erster Download ~109 MiB) ...")
            net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                               stochastic_depth_prob=.1)
            self.features = net.features.cpu().eval()
            self.dim = 768
            self.provenance = WEIGHTS_NAME
            del net
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.fingerprint = self._fingerprint()

    def _fingerprint(self):
        h = hashlib.sha256()
        for name, tensor in self.features.state_dict().items():
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


class TransferModel(ModuleBase):
    def __init__(self, factory, config):
        super().__init__()
        self.config = config
        self.backbone = copy.deepcopy(factory.features)
        self.feature_dim = factory.dim
        start = {"frozen": len(self.backbone), "last_stage": 6,
                 "last_two_stages": 4}[config.transfer_mode]
        self.trainable_start = start
        for i, block in enumerate(self.backbone):
            for p in block.parameters():
                p.requires_grad_(i >= start)
            for module in block.modules():
                if module.__class__.__name__ == "StochasticDepth":
                    module.p = config.stochastic_depth * module.p / .1
        self.backbone_active = config.transfer_mode != "frozen"
        self.feature_norm = nn.LayerNorm(factory.dim)
        if config.architecture == "cnn_transformer":
            self.project = nn.Linear(factory.dim, config.transformer_width - 1)
            self.encoder = nn.TransformerEncoderLayer(
                config.transformer_width, config.attention_heads,
                dim_feedforward=config.transformer_width * config.feedforward_multiplier,
                dropout=config.transformer_dropout, activation="gelu",
                batch_first=True, norm_first=False)
            self.after_encoder = nn.Dropout(config.transformer_dropout)
            head_input = config.transformer_width
        else:
            head_input = factory.dim
        if config.head_hidden:
            self.head = nn.Sequential(nn.Dropout(config.head_dropout),
                                      nn.Linear(head_input, config.head_hidden), nn.GELU(),
                                      nn.Dropout(config.head_dropout),
                                      nn.Linear(config.head_hidden, 1))
        else:
            self.head = nn.Sequential(nn.Dropout(config.head_dropout),
                                      nn.Linear(head_input, 1))
        self.set_epoch(1)

    def set_epoch(self, epoch):
        self.backbone_active = (self.config.transfer_mode != "frozen" and
                                epoch > self.config.warmup_epochs)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "backbone"):
            for i, block in enumerate(self.backbone):
                block.train(bool(mode and self.backbone_active and i >= self.trainable_start))
        return self

    def maps(self, pairs):
        inputs = signal_images(pairs, self.config.image_width)
        if self.training and self.backbone_active:
            return self.backbone(inputs)
        with torch.no_grad():
            return self.backbone(inputs)

    def from_maps(self, maps):
        values = F.dropout(maps.mean(dim=2), self.config.feature_dropout, self.training)
        if self.config.architecture == "cnn":
            embedding = self.feature_norm(values.mean(dim=-1))
        else:
            values = F.adaptive_avg_pool1d(values, self.config.tokens).transpose(1, 2)
            values = self.project(self.feature_norm(values))
            # Phase aus allen Bild-Spalten, analog zu Stride 32 des CNNs.
            phase = torch.linspace(-1, 1, self.config.image_width, device=values.device)
            phase = F.avg_pool1d(phase.reshape(1, 1, -1), 32, 32)
            phase = F.adaptive_avg_pool1d(phase, self.config.tokens).transpose(1, 2)
            tokens = torch.cat((values, phase.expand(values.shape[0], -1, -1)), dim=-1)
            embedding = self.after_encoder(self.encoder(tokens)).mean(dim=1)
        return self.head(embedding).squeeze(-1)

    def forward(self, pairs):
        return self.from_maps(self.maps(pairs))

    def learned_state(self):
        # Vortrainierte, unveraenderte Parameter nicht tausendfach sichern.
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        return {name: v.detach().cpu().clone() for name, v in self.state_dict().items()
                if name in trainable or not name.startswith("backbone.")}

    def restore_learned(self, state):
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        expected.update(name for name in self.state_dict() if not name.startswith("backbone."))
        if not expected.issubset(state):
            raise RuntimeError("Trainierbare Parameter fehlen im Transfer-Checkpoint.")
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith("backbone.") for key in missing):
            raise RuntimeError("Unvollstaendiger Transfer-Checkpoint.")


@dataclass
class PreparedData:
    train_pairs: Any
    eval_pairs: Any
    train_owners: np.ndarray
    eval_owners: np.ndarray
    train_counts: np.ndarray
    targets: Any
    class_weights: Any
    train_records: Sequence
    eval_records: Sequence
    target_mean: float
    target_std: float
    normalizer: Any
    n_views: int
    noise_seed: int


def prepare_data(train, evaluation, config, args, seed):
    train_pairs, eval_pairs, owners, eval_owners, normalizer, copies = prepare_arrays(
        train, evaluation, config, args, seed)
    angles = np.asarray([r.angle_deg for r in train], float)
    center, scale = float(angles.mean()), max(float(angles.std()), 1.)
    labels = np.asarray([r.hard_reference for r in train], int)
    weights = np.ones(len(train), np.float32)
    if config.class_weight == "balanced":
        counts = np.bincount(labels, minlength=2)
        if not counts.min():
            raise ValueError("Trainingsfold braucht beide Klassen.")
        weights = (len(train)/(2*counts[labels])).astype(np.float32)
    return PreparedData(
        torch.from_numpy(train_pairs), torch.from_numpy(eval_pairs), owners, eval_owners,
        np.bincount(owners, minlength=len(train)),
        torch.from_numpy(((angles-center)/scale).astype(np.float32)).to(args.device),
        torch.from_numpy(weights).to(args.device), train, evaluation, center, scale,
        normalizer, int(len(train)+copies.sum()), seed)


class FrozenFeatureCache:
    """CPU memory-mapped feature maps, bounded disk LRU, one GPU microbatch."""
    def __init__(self, limit=8., root=None):
        self.limit_bytes = int(limit*1024**3)
        self.root = Path(root) if root is not None else None

    def clear(self):
        if self.root and self.root.exists():
            import shutil
            shutil.rmtree(self.root)

    def _evict(self, required, protected):
        files = list(self.root.glob("*.npy"))
        used = sum(p.stat().st_size for p in files)
        for meta in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime):
            if used+required <= self.limit_bytes:
                break
            if meta.stem == protected:
                continue
            for suffix in ("_train.npy", "_eval.npy"):
                path = self.root/(meta.stem+suffix)
                if path.exists():
                    used -= path.stat().st_size
                    path.unlink()
            meta.unlink(missing_ok=True)
        if used+required > self.limit_bytes:
            raise RuntimeError("Featurecache zu klein. --feature-cache-gb erhoehen; Daten/Folds bleiben vollstaendig.")

    def get(self, prepared, config, factory, args, model):
        if self.root is None:
            raise RuntimeError("Featurecache-Verzeichnis fehlt.")
        self.root.mkdir(parents=True, exist_ok=True)
        key = digest({"train": dataset_fingerprint(prepared.train_records),
                      "eval": dataset_fingerprint(prepared.eval_records), "channel_set": config.channel_set,
                      "mask": CHANNEL_SETS[config.channel_set], "width": 160,
                      "impute": prepared.normalizer.impute_, "noise": args.noise_sigma,
                      "noise_seed": prepared.noise_seed, "cap": config.rare_angle_copy_cap,
                      "weights": factory.fingerprint, "precision": "fp32", "version": VERSION})
        meta = self.root/(key+".json")
        paths = [self.root/(key+"_"+role+".npy") for role in ("train", "eval")]
        hit = meta.exists() and all(p.exists() for p in paths)
        feature_seconds, size = 0., args.contact_microbatch
        if not hit:
            for path in paths:
                path.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            # ConvNeXt stride 32; synthetic test factory has the same stride.
            required = (len(prepared.train_pairs)+len(prepared.eval_pairs))*factory.dim*3*5*4 + 2048
            self._evict(required, key)
            sync_device(args)
            started = time.monotonic()
            model.eval()
            try:
                with torch.no_grad():
                    for source, path in zip((prepared.train_pairs, prepared.eval_pairs), paths):
                        temporary = path.with_suffix(".partial")
                        temporary.unlink(missing_ok=True)
                        array, start = None, 0
                        try:
                            while start < len(source):
                                if args.runtime_budget is not None:
                                    args.runtime_budget.check()
                                maps = None
                                try:
                                    maps = model.maps(source[start:start+size].to(args.device)).float().cpu().numpy()
                                    if not np.isfinite(maps).all():
                                        raise RuntimeError("Nicht endliche CNN-Features.")
                                    if array is None:
                                        array = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32",
                                            shape=(len(source), *maps.shape[1:]))
                                    array[start:start+len(maps)] = maps
                                    start += len(maps)
                                    maps = None
                                except torch.cuda.OutOfMemoryError:
                                    maps = None
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    if size <= 1:
                                        raise
                                    size = max(1, size//2)
                                    log(f"CNN-Microbatch nach OOM: {size}.")
                            array.flush()
                            del array
                            array = None
                            os.replace(temporary, path)
                        finally:
                            if array is not None:
                                array.flush()
                                del array
                            temporary.unlink(missing_ok=True)
                sync_device(args)
                feature_seconds = time.monotonic()-started
                atomic_json(meta, {"key": key, "feature_microbatch": size,
                                   "channel_set": config.channel_set})
            except BaseException:
                for path in paths:
                    path.unlink(missing_ok=True)
                raise
        else:
            size = int(json.loads(meta.read_text())["feature_microbatch"])
            os.utime(meta, None)
        model._feature_microbatch = size
        # Copy-on-write mappings are writable to torch, without loading the full cache.
        values = tuple(torch.from_numpy(np.load(p, mmap_mode="c", allow_pickle=False)) for p in paths)
        return values, feature_seconds


def rng_state():
    result = {"cpu": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_epoch(model, prepared, optimizer, config, args, seed, epoch,
                cached=None, microbatch=None):
    if config.task != "angle_regression":
        raise ValueError("Nur Regressionsloss erlaubt.")
    model.train()
    model.set_epoch(epoch)
    microbatch = microbatch or args.contact_microbatch
    permutation = np.random.default_rng(stable_seed(seed, epoch, "order")).permutation(len(prepared.train_records))
    owner, total = prepared.train_owners, 0.
    for step, lo in enumerate(range(0, len(permutation), args.person_batch_size)):
        persons = permutation[lo:lo + args.person_batch_size]
        indices = np.flatnonzero(np.isin(owner, persons))
        seed_everything(stable_seed(seed, epoch, step, "dropout"))
        step_rng = rng_state()
        while True:
            optimizer.zero_grad(set_to_none=True)
            prediction = losses = loss = None
            try:
                batch_loss = 0.
                for start in range(0, len(indices), microbatch):
                    local = indices[start:start + microbatch]
                    local_owner = torch.as_tensor(owner[local], device=args.device)
                    with autocast_context(args):
                        prediction = (model.from_maps(cached[0][local].to(args.device)) if cached is not None
                                      else model(prepared.train_pairs[local].to(args.device)))
                    losses = F.smooth_l1_loss(prediction.float(), prepared.targets[local_owner].float(),
                                             beta=config.huber_beta, reduction="none")
                    denominator = torch.as_tensor(prepared.train_counts[owner[local]], dtype=losses.dtype, device=args.device)
                    weights = prepared.class_weights[local_owner] / denominator / len(persons)
                    loss = (losses * weights).sum()
                    loss.backward()
                    batch_loss += float(loss.detach())
                    prediction = losses = loss = None
                break
            except torch.cuda.OutOfMemoryError:
                prediction = losses = loss = None
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if microbatch <= 1:
                    raise
                microbatch = max(1, microbatch // 2)
                restore_rng(step_rng)
                log(f"VRAM-Anpassung: Kontakt-Microbatch={microbatch}; Personenbatch bleibt gleich.")
        nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
        optimizer.step()
        total += batch_loss * len(persons)
    return total / len(permutation), microbatch


def receipt_path(output, section, job, stage=None):
    suffix = "_" + stage if stage else ""
    return output / "_state" / section / (job + suffix + ".json")


def read_receipt(output, section, job, stage=None):
    path = receipt_path(output, section, job, stage)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def all_receipts(output, section):
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output / "_state" / section).glob("*.json"))]


def mean_sd(values):
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not len(array):
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else float("nan")


def experiment_splits(records, args):
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("Immer Nested-CV: mindestens zwei aeussere und zwei innere Folds.")
    for repeat, seed in enumerate(args.split_seeds, 1):
        outer = balanced_angle_fold_indices(records, args.outer_folds, stable_seed(seed, "outer"),
                                             args.split_candidates)
        for fold, (train, test) in enumerate(outer, 1):
            inner = balanced_angle_fold_indices([records[i] for i in train],
                                                args.inner_folds,
                                                stable_seed(seed, "inner", fold),
                                                args.split_candidates)
            yield repeat, seed, fold, train, test, [
                (train[it], train[iv]) for it, iv in inner]


def synthetic_records():
    rng = np.random.default_rng(142)
    records = []
    for i in range(24):
        curve = ("TH", "L", "THL")[i % 3]
        boundary = ANGLE_CLASS_THRESHOLDS_DEG[curve]
        label = int(i % 4 in (0, 1))
        angle = boundary + (7 if label else -7) + float(rng.uniform(-1, 1))
        values = rng.normal(0, .15, (2 + i % 2, 3, 101)).astype(np.float32)
        values[:, 0] += np.sin(np.linspace(0, math.pi * 2, 101)) * (.3 + label)
        identity = f"SYNTHETIC_PERSON_{i:02d}"
        measurement = Measurement("synthetic", "synthetic", identity, values)
        records.append(SubjectRecord(identity, angle, curve, boundary,
                                     float(label), label, [measurement]))
    return records


def fold_metrics(records, values, folds, task, threshold=.5):
    parts = [metrics_for([records[i] for i in np.flatnonzero(folds == fold)],
                         values[folds == fold], task, threshold)
             for fold in sorted(np.unique(folds))]
    mean = {m: mean_sd([p[m] for p in parts])[0] for m in METRICS}
    return mean, parts


def evaluate_oof(records, predictions, folds, task, args):
    if task != "angle_regression":
        raise ValueError("OOF-Auswahl nur fuer Winkelregression.")
    predictions, folds = np.asarray(predictions, float), np.asarray(folds, int)
    if predictions.ndim != 2 or predictions.shape[1] != len(records):
        raise ValueError("OOF-Array muss [Initialisierungen, echte Personen] enthalten.")
    if len(folds) != len(records) or not np.isfinite(predictions).all() or np.any(folds < 1):
        raise ValueError("OOF-Prognosen/Folds sind unvollstaendig.")
    if len(np.unique(folds)) != args.inner_folds:
        raise ValueError("OOF-Foldzahl passt nicht zur inneren CV.")
    ensemble = predictions.mean(axis=0)
    metrics, parts = fold_metrics(records, ensemble, folds, task)
    seed_metrics = [fold_metrics(records, row, folds, task)[0] for row in predictions]
    result = {"decision_threshold": .5, "threshold_source": "fixed_curve_type_rules_not_probability_threshold",
              "n_training_initializations": len(predictions), "n_inner_folds_completed": len(parts),
              "complete_inner_cv": True, "n_oof_persons": len(records)}
    for metric in METRICS:
        result["mean_inner_" + metric] = metrics[metric]
        result["sd_inner_folds_" + metric] = mean_sd([part[metric] for part in parts])[1]
        result["sd_initializations_" + metric] = mean_sd([part[metric] for part in seed_metrics])[1]
    return clean_json(result)


def candidate_rank(row, criterion):
    metrics = {m: row.get("mean_inner_" + m) for m in METRICS}
    return ranking(metrics, criterion)


def choose_smallest(rows, args):
    """Vorab definierte Einfachheitsheuristik; kein Outer-Zugriff, kein Test."""
    if not rows:
        raise ValueError("Keine vollstaendigen Inner-Kandidaten zur Auswahl.")
    best = max(rows, key=lambda r: (candidate_rank(r, args.criterion), r["config_id"]))
    sd = best.get("sd_inner_folds_" + args.criterion)
    nominal_se = (float(sd) / math.sqrt(best["n_inner_folds_completed"])
                  if sd is not None and np.isfinite(sd) else 0.)
    tolerance = min(args.simplicity_gap, nominal_se)
    floor = best["mean_inner_" + args.criterion] - tolerance
    eligible = [r for r in rows if r["mean_inner_" + args.criterion] >= floor - 1e-12]
    chosen = min(eligible, key=lambda r: (r["trainable_parameters"],
                                        tuple(-v for v in candidate_rank(r, args.criterion)),
                                        r["config_id"]))
    return chosen, {"best_config_id": best["config_id"], "nominal_inner_standard_error": nominal_se,
                    "simplicity_tolerance_used": tolerance, "eligibility_floor": floor,
                    "n_eligible": len(eligible), "chosen_config_id": chosen["config_id"]}


def atomic_npz(path, **arrays):
    def writer(temporary):
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
    atomic_write(path, writer)


def prediction_path(output, job, stage):
    return output / "_private_predictions" / (job + "_" + stage + ".npz")


def load_predictions(output, job, stage, evaluation):
    path = prediction_path(output, job, stage)
    with np.load(path, allow_pickle=False) as data:
        if str(data["evaluation_fingerprint"].item()) != dataset_fingerprint(evaluation):
            raise RuntimeError(f"Prognose-Cache passt nicht zur Auswertung: {path}")
        values = data["predictions"].astype(float)
    if values.shape != (len(evaluation),) or not np.isfinite(values).all():
        raise RuntimeError(f"Unvollstaendige Prognose-Datei: {path}")
    return values


def predict_persons(model, prepared, config, args, cached=None, microbatch=None):
    model.eval()
    values, size, start = [], microbatch or args.contact_microbatch, 0
    with torch.no_grad():
        while start < len(prepared.eval_pairs):
            result = None
            try:
                local = slice(start, start + size)
                with autocast_context(args):
                    result = (model.from_maps(cached[1][local].to(args.device)) if cached is not None
                              else model(prepared.eval_pairs[local].to(args.device)))
                values.append(result.float().cpu().numpy())
                start += size
                result = None
            except torch.cuda.OutOfMemoryError:
                result = None
                gc.collect()
                torch.cuda.empty_cache()
                if size <= 1:
                    raise
                size = max(1, size // 2)
                log(f"VRAM-Anpassung bei Prognose: Kontakt-Microbatch={size}.")
    model._last_eval_microbatch = size
    contacts = np.concatenate(values)
    count = len(prepared.eval_records)
    result = np.bincount(prepared.eval_owners, weights=contacts, minlength=count)
    result /= np.bincount(prepared.eval_owners, minlength=count)
    return result * prepared.target_std + prepared.target_mean


def checkpoint_rank(metrics, task, args):
    if task != "angle_regression" or args.criterion != "cohen_kappa":
        raise ValueError("V38: nur Winkelregression mit Cohen's Kappa als Auswahlziel.")
    return ranking(metrics, "cohen_kappa")


def fit_trial(train, evaluation, config, args, factory, cache, output,
              job, seed, data_seed, stage, epochs, inner, metadata, patience=0):
    if config.task != "angle_regression" or args.criterion != "cohen_kappa":
        raise ValueError("Nur Winkelregression mit Kappa-Auswahl erlaubt.")
    section = "inner_fit" if inner else "outer_member"
    completed = read_receipt(output, section, job, stage)
    if completed is not None:
        load_predictions(output, job, stage, evaluation)
        return completed
    budget = getattr(args, "runtime_budget", None)
    if budget is not None:
        budget.check()
    sync_device(args)
    started = time.monotonic()
    seed_everything(seed)
    prepared = prepare_data(train, evaluation, config, args, data_seed)
    model = TransferModel(factory, config).to(args.device)
    parameters = parameter_breakdown(model)
    adapter = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    backbone = [p for name, p in model.named_parameters() if name.startswith("backbone.") and p.requires_grad]
    groups = [{"params": adapter, "lr": config.learning_rate, "kind": "adapter"}]
    if backbone:
        groups.append({"params": backbone, "lr": config.learning_rate * config.backbone_lr_fraction, "kind": "backbone"})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    checkpoint = output / "_checkpoints" / section / (job + ".pt")
    contract = digest({"version": VERSION, "config": asdict(config), "seed": seed,
                       "data_seed": data_seed, "train": dataset_fingerprint(train),
                       "evaluation": dataset_fingerprint(evaluation), "weights": factory.fingerprint,
                       "inner": inner, "noise": args.noise_sigma, "batch": args.person_batch_size,
                       "criterion": args.criterion, "precision": args.resolved_precision, "tf32": args.tf32})
    last_epoch, best_epoch, stale = 0, 0, 0
    best_score, best_state, history = None, None, []
    microbatch = args.head_microbatch
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["contract"] != contract:
            raise RuntimeError(f"Checkpoint passt nicht zum Versuch: {checkpoint}")
        old_stage = saved["stage"]
        resume_stale = resumed_stale(old_stage, stage, saved["stale"])
        model.restore_learned(saved["current_state"])
        optimizer.load_state_dict(saved["optimizer"])
        last_epoch, best_epoch = saved["epoch"], saved["best_epoch"]
        stale = resume_stale
        best_score = tuple(saved["best_score"]) if saved["best_score"] is not None else None
        best_state, history = saved["best_state"], saved["history"]
        microbatch = saved.get("microbatch", microbatch)
        restore_rng(saved["rng"])
        if old_stage != stage:
            saved.update(stage=stage, stale=0)
            atomic_torch(checkpoint, saved)
        log(f"Resume {job}/{stage}: nach Epoche {last_epoch}, best={best_epoch}, stale={stale}.")
        del saved
    cached, feature_seconds = None, 0.
    if config.transfer_mode == "frozen":
        cached, feature_seconds = cache.get(prepared, config, factory, args, model)
    sync_device(args)
    startup_seconds = time.monotonic() - started
    initial_epoch = last_epoch
    for epoch in range(last_epoch + 1, epochs + 1):
        if inner and patience and stale >= patience:
            break
        if budget is not None:
            budget.check()
        sync_device(args)
        epoch_started = time.monotonic()
        for group in optimizer.param_groups:
            if group["kind"] == "backbone":
                group["lr"] = config.learning_rate * config.backbone_lr_fraction if epoch > config.warmup_epochs else 0.
        loss, microbatch = train_epoch(model, prepared, optimizer, config, args, seed, epoch, cached, microbatch)
        if not math.isfinite(loss):
            raise RuntimeError(f"Nicht endliche Regressionsloss in {job}, Epoche {epoch}.")
        entry = {"epoch": epoch, "stage": stage, "training_loss": loss}
        if inner:
            values = predict_persons(model, prepared, config, args, cached, microbatch)
            microbatch = min(microbatch, getattr(model, "_last_eval_microbatch", microbatch))
            metrics = metrics_for(evaluation, values, config.task)
            score = checkpoint_rank(metrics, config.task, args)
            entry.update({"validation_" + m: metrics[m] for m in METRICS})
            eligible = config.transfer_mode == "frozen" or epoch > config.warmup_epochs
            if eligible and (best_score is None or score > best_score):
                best_score, best_epoch, stale = score, epoch, 0
                best_state = model.learned_state()
            elif eligible:
                stale += 1
        else:
            best_epoch = epoch
        sync_device(args)
        entry["epoch_seconds"] = time.monotonic() - epoch_started
        history.append(entry)
        last_epoch = epoch
        atomic_torch(checkpoint, {"contract": contract, "stage": stage, "epoch": epoch, "best_epoch": best_epoch,
                                  "stale": stale, "best_score": best_score, "best_state": best_state,
                                  "current_state": model.learned_state(), "optimizer": optimizer.state_dict(),
                                  "rng": rng_state(), "history": history, "microbatch": microbatch,
                                  "config": asdict(config)})
        atomic_csv(output / "epoch_history" / (job + ".csv"), history)
        sync_device(args)
        entry["epoch_wall_seconds"] = time.monotonic() - epoch_started
        if epoch == 1 or epoch % args.log_every == 0 or epoch == epochs:
            extra = f" | Val-Kappa={metrics['cohen_kappa']:.4f} | MAE={metrics['angle_mae_deg']:.2f} Grad" if inner else ""
            log(f"    {job}/{stage}: Epoche {epoch}/{epochs} | Loss={loss:.4f}{extra}")
    if inner:
        if best_state is None:
            raise RuntimeError("Inner-Fit hat keinen gueltigen Checkpoint erzeugt.")
        model.restore_learned(best_state)
    values = predict_persons(model, prepared, config, args, cached, microbatch)
    metrics = metrics_for(evaluation, values, config.task)
    atomic_npz(prediction_path(output, job, stage), predictions=values,
               evaluation_fingerprint=np.asarray(dataset_fingerprint(evaluation)))
    if not inner:
        atomic_torch(output / "outer_models" / (job + ".pt"), {
            "version": VERSION, "config": asdict(config), "learned_state": model.learned_state(),
            "base_weights": WEIGHTS_NAME, "base_fingerprint": factory.fingerprint,
            "normalizer_raw_denominators": prepared.normalizer.raw_denominators_,
            "normalizer_impute": prepared.normalizer.impute_, "target_mean": prepared.target_mean,
            "target_std": prepared.target_std, "source_channels": SELECTED_SOURCE_CHANNEL_INDICES, "channel_set": config.channel_set,
            "active_master_indices": CHANNEL_SETS[config.channel_set],
            "precision": args.resolved_precision, "tf32": args.tf32,
            "metadata": metadata, "seed": seed, "data_seed": data_seed})
    timed = [h.get("epoch_wall_seconds", h["epoch_seconds"]) for h in history
             if h["epoch"] > max(1, config.warmup_epochs)]
    if not timed:
        timed = [h.get("epoch_wall_seconds", h["epoch_seconds"]) for h in history]
    sync_device(args)
    row = {**metadata, **config.row(), "stage": stage, "best_epoch": best_epoch,
           "epochs_run": last_epoch, "epochs_new_this_call": last_epoch - initial_epoch,
           "epoch_budget": epochs, "n_train_persons": len(train), "n_eval_persons": len(evaluation),
           "n_training_views": prepared.n_views, "duration_seconds": time.monotonic() - started,
           "feature_seconds": feature_seconds, "startup_seconds": startup_seconds,
           "mean_epoch_seconds": float(np.mean(timed)), "q75_epoch_seconds": float(np.quantile(timed, .75)),
           "train_contacts": len(prepared.train_pairs), "eval_contacts": len(prepared.eval_pairs),
           "n_backbone_training_epochs": 0 if config.transfer_mode == "frozen" else max(0, last_epoch - config.warmup_epochs),
           "contact_microbatch": getattr(model, "_feature_microbatch", args.contact_microbatch),
           "head_microbatch_effective": min(microbatch, getattr(model, "_last_eval_microbatch", microbatch)),
           "precision": args.resolved_precision, **parameters, **{m: metrics[m] for m in METRICS}}
    atomic_json(receipt_path(output, section, job, stage), row)
    if not inner:
        checkpoint.unlink(missing_ok=True)
    del model, optimizer, prepared, cached, best_state
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return clean_json(row)


def evaluate_candidate(records, args, factory, cache, output, config, context,
                       train, inner_splits, stage, epochs, initialization_ids):
    repeat, split_seed, fold = context
    name = f"W{repeat}_F{fold}_" + config.config_id
    complete = read_receipt(output, "candidate", name, stage)
    if complete is not None:
        return complete
    local_records = [records[i] for i in train]
    lookup = {int(global_index): local_index for local_index, global_index in enumerate(train)}
    predictions = np.full((len(initialization_ids), len(train)), np.nan, float)
    folds = np.zeros(len(train), int)
    fit_rows = []
    fit_start = time.monotonic()
    for initialization_position, initialization in enumerate(initialization_ids):
        for inner_fold, (it, iv) in enumerate(inner_splits, 1):
            local = np.asarray([lookup[int(i)] for i in iv], int)
            if initialization_position == 0:
                if np.any(folds[local] != 0):
                    raise AssertionError("Innere OOF-Person wurde mehrfach zugeordnet.")
                folds[local] = inner_fold
            train_records, validation = [records[i] for i in it], [records[i] for i in iv]
            job = f"W{repeat}_F{fold}_I{inner_fold}_S{initialization}_" + config.config_id
            model_seed = stable_seed(initialization, split_seed, fold, inner_fold, "model")
            data_seed = stable_seed(args.seed, split_seed, fold, inner_fold, "views")
            row = fit_trial(train_records, validation, config, args, factory, cache, output,
                            job, model_seed, data_seed, stage, epochs, True,
                            {"outer_repeat": repeat, "split_seed": split_seed, "outer_fold": fold,
                             "inner_fold": inner_fold, "initialization_seed": initialization},
                            patience=0 if stage == "broad" else (args.stable_patience if stage == "stable" else args.patience))
            predictions[initialization_position, local] = load_predictions(output, job, stage, validation)
            fit_rows.append(row)
    stats = evaluate_oof(local_records, predictions, folds, config.task, args)
    row = {"outer_repeat": repeat, "split_seed": split_seed, "outer_fold": fold,
           "stage": stage, **config.row(), **stats,
           "mean_best_epoch": float(np.mean([r["best_epoch"] for r in fit_rows])),
           "median_best_epoch": float(np.median([r["best_epoch"] for r in fit_rows])),
           "min_best_epoch": min(r["best_epoch"] for r in fit_rows),
           "max_best_epoch": max(r["best_epoch"] for r in fit_rows),
           "duration_seconds": time.monotonic() - fit_start,
           "trainable_parameters": fit_rows[0]["trainable_parameters"],
           "encoder_parameters": fit_rows[0]["encoder_parameters"],
           "cnn_trainable_parameters": fit_rows[0]["cnn_trainable_parameters"],
           "small_component_parameters": fit_rows[0]["small_component_parameters"]}
    atomic_json(receipt_path(output, "candidate", name, stage), row)
    return clean_json(row)


def aggregate_reports(output, args):
    fits, candidates, outer_rows = (all_receipts(output, s) for s in ("inner_fit", "candidate", "outer"))
    if fits:
        frame = pd.DataFrame(fits).sort_values(["split_seed", "outer_fold", "channel_set", "config_id", "initialization_seed", "inner_fold", "stage"])
        atomic_csv(output/"inner_fold_results.csv", frame)
        names = ["config_id", "channel_set", "architecture", "trainable_parameters", "total_parameters",
                 "cnn_trainable_parameters", "encoder_parameters", "small_component_parameters"]
        atomic_csv(output/"parameter_breakdown.csv", frame[names].drop_duplicates())
    if candidates:
        frame = pd.DataFrame(candidates).sort_values(["split_seed", "outer_fold", "channel_set", "config_id", "stage"])
        atomic_csv(output/"inner_candidate_results.csv", frame)
        for stage in ("broad", "stable", "final"):
            part = frame[frame.stage.eq(stage)].sort_values(
                ["mean_inner_cohen_kappa", "trainable_parameters", "config_id"], ascending=[False, True, True])
            if not part.empty:
                atomic_csv(output/("inner_ranking_"+stage+".csv"), part)
                atomic_csv(output/("top20_inner_"+stage+".csv"), part.head(args.top_n))
    choices = all_receipts(output, "channel_selection")
    if choices:
        atomic_csv(output/"inner_channel_selection.csv", [r for choice in choices for r in choice["rows"]])
    if not outer_rows:
        return None
    outer = pd.DataFrame(outer_rows).sort_values(["split_seed", "outer_fold", "method_id"])
    atomic_csv(output/"outer_fold_metrics.csv", outer)
    seed_rows = []
    for (split_seed, method), part in outer.groupby(["split_seed", "method_id"]):
        r = {"split_seed": int(split_seed), "method_id": method,
             "n_outer_folds_completed": len(part),
             "expected_outer_folds": args.outer_folds,
             "complete_outer_repeat": len(part) == args.outer_folds}
        for metric in METRICS:
            r["mean_"+metric], r["sd_outer_folds_"+metric] = mean_sd(part[metric])
        seed_rows.append(r)
    atomic_csv(output/"seed_summary.csv", seed_rows)
    members = all_receipts(output, "outer_member")
    if members:
        atomic_csv(output/"outer_member_metrics.csv", members)
    expected = len(args.split_seeds)*args.outer_folds
    methods = []
    for method in [*CHANNEL_SETS, "hpo_inner_selected"]:
        part = outer[outer.method_id.eq(method)]
        if part.empty:
            continue
        contexts = set(zip(part.split_seed, part.outer_fold))
        r = {"method_id": method, "primary_procedure": method == "hpo_inner_selected",
             "n_outer_tests": len(contexts), "expected_outer_tests": expected,
             "complete_nested_cv": len(contexts) == expected,
             "n_ensemble_members": int(part.n_ensemble_members.max()),
             "mean_trainable_parameters_per_model": part.trainable_parameters.mean()}
        for metric in METRICS:
            r["mean_"+metric], r["sd_outer_folds_"+metric] = mean_sd(part[metric])
            means = part.groupby("split_seed")[metric].mean()
            r["sd_seed_means_"+metric] = mean_sd(means)[1] if len(means) > 1 else float("nan")
        methods.append(r)
    summary = pd.DataFrame(methods)
    atomic_csv(output/"method_summary.csv", summary)
    atomic_csv(output/"channel_summary.csv", summary[~summary.primary_procedure])
    paired = []
    for a, b in itertools.combinations(CHANNEL_SETS, 2):
        p = outer[outer.method_id.eq(a)].merge(outer[outer.method_id.eq(b)],
            on=["split_seed", "outer_fold"], suffixes=("_a", "_b"))
        r = {"channel_set_a": a, "channel_set_b": b, "n_paired_outer_contexts": len(p),
             "descriptive_only": True}
        for metric in METRICS:
            r["mean_A_minus_B_"+metric], r["sd_fold_difference_"+metric] = mean_sd(p[metric+"_a"]-p[metric+"_b"])
        paired.append(r)
    atomic_csv(output/"paired_channel_sets.csv", paired)
    return summary


def cleanup_checkpoints(output, context, survivors):
    keep = {c.config_id for c in survivors}
    for path in (output / "_checkpoints" / "inner_fit").glob(context + "_I*.pt"):
        if not any(path.stem.endswith("_" + key) for key in keep):
            path.unlink()


def estimated_hours(count, models, args, contexts, records, pool):
    configs = subset_configs(pool, count)
    totals = {"broad": 0., "stable": 0., "final": 0., "outer": 0.}
    for repeat, seed, fold, train, test, splits in contexts:
        outer_costs = {s: [] for s in CHANNEL_SETS}
        for condition in CONDITIONS:
            group = [c for c in configs if c.condition == condition]
            stable, final = [], []
            for c in group:
                b = st = fi = 0.
                for it, iv in splits:
                    tr, ev = [records[i] for i in it], [records[i] for i in iv]
                    b += cost_for_fit(c, tr, ev, args, models, args.broad_epochs)
                    st += cost_for_fit(c, tr, ev, args, models, args.stable_epochs-args.broad_epochs)
                    fi += cost_for_fit(c, tr, ev, args, models, args.max_epochs-args.stable_epochs)
                    fi += (len(args.train_seeds)-1)*cost_for_fit(c, tr, ev, args, models, args.max_epochs)
                totals["broad"] += b
                stable.append(st)
                final.append(fi)
                outer_costs[c.channel_set].append(len(args.train_seeds)*cost_for_fit(c,
                    [records[i] for i in train], [records[i] for i in test], args, models, args.max_epochs))
            totals["stable"] += sum(sorted(stable, reverse=True)[:2])
            totals["final"] += max(final)
        totals["outer"] += sum(max(values) for values in outer_costs.values())
    return {"estimated_hours": sum(totals.values())/3600, "seconds_by_stage": totals,
            "configurations_per_channel_set": count, "broad_candidates_per_outer": 7*count,
            "stable_candidates_per_outer": 28, "final_candidates_per_outer": 14,
            "outer_contexts": len(contexts), "outer_ensembles": 7*len(contexts),
            "cost_assumption": "cold_cache_every_fit_no_early_stopping_discount"}


def ensure_search_plan(records, args, factory, cache, contexts, pool, output):
    path = output/"search_plan.json"
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        configs = [TransferConfig(**r) for r in plan["configurations"]]
        if set(c.condition for c in configs) != set(CONDITIONS) or len(set(Counter(c.condition for c in configs).values())) != 1:
            raise RuntimeError("Gespeicherter Suchplan deckt nicht alle sieben Sets gleichmaessig ab.")
        plan.update(time_policy="planning_target_only", automatic_time_limit=False,
                    operations=operation_counts(plan["n_search_configs_per_set"], args))
        atomic_json(path, plan)
        atomic_json(output/"operation_plan.json", plan["operations"])
        return configs, plan
    models, rows, estimate = {}, [], None
    if args.smoke_test:
        count, mode = 8, "technical_2x2_nested_smoke_not_a_runtime_estimate"
    else:
        args.runtime_budget.in_pilot = True
        args.runtime_budget.check()
        it, iv = contexts[0][5][0]
        train = pilot_subset([records[i] for i in it], args.pilot_persons)
        evaluation = pilot_subset([records[i] for i in iv], args.pilot_persons)
        pilot_cache = FrozenFeatureCache(args.feature_cache_gb, output/"_feature_cache"/"pilot")
        for channel_set in CHANNEL_SETS:
            feature_startup = 0.
            for architecture in ("cnn", "cnn_transformer"):
                cfg = TransferConfig(architecture, channel_set=channel_set, rare_angle_copy_cap=4,
                    head_hidden=32 if architecture == "cnn" else 0,
                    transformer_width=24 if architecture == "cnn_transformer" else 0,
                    attention_heads=4 if architecture == "cnn_transformer" else 0,
                    transformer_dropout=.2 if architecture == "cnn_transformer" else 0.,
                    feedforward_multiplier=2 if architecture == "cnn_transformer" else 0,
                    tokens=5 if architecture == "cnn_transformer" else 0)
                key = pilot_key(cfg)
                row = fit_trial(train, evaluation, cfg, args, factory, pilot_cache, output/"_runtime_pilot",
                    key, stable_seed(args.search_seed, "runtime_probe", key),
                    stable_seed(args.seed, "runtime_probe_views"), "pilot", args.pilot_epochs, True,
                    {"purpose": "timing_only_no_hpo_selection", "pilot_key": key})
                feature_startup = max(feature_startup, float(row["feature_seconds"]))
                m = {"epoch_seconds": max(.001, float(row["q75_epoch_seconds"])),
                     "startup_seconds": max(.001, float(row["startup_seconds"])-float(row["feature_seconds"]))+feature_startup,
                     "train_contacts": row["train_contacts"], "eval_contacts": row["eval_contacts"],
                     "pilot_duration_seconds": float(row["duration_seconds"])}
                models[key] = m
                rows.append({"pilot_key": key, **m})
                atomic_csv(output/"runtime_pilot.csv", rows)
                log(f"Pilot {len(rows)}/14 {key}: Epoche {m['epoch_seconds']:.3f} s; Start {m['startup_seconds']:.2f} s.")
        pilot_cache.clear()
        args.runtime_budget.in_pilot = False
        pilot_hours = sum(m["pilot_duration_seconds"] for m in models.values())/3600
        planning_target = max(0., args.target_hours-pilot_hours)
        counts = [args.search_configs] if args.search_configs else list(range(
            args.min_search_configs, args.max_search_configs+1, 4))
        choices = [{"n_search_configs_per_set": n, **estimated_hours(n, models, args, contexts, records, pool)} for n in counts]
        atomic_json(output/"pilot_cost_models.json", {"models": models, "planning_target_hours": planning_target,
                    "target_hours_total": args.target_hours, "automatic_time_limit": False, "choices": choices,
                    "note": "Conservative timing estimates guide operation counts and never stop the run."})
        estimate, mode = choose_operation_count(choices, planning_target, args.search_configs)
        count = estimate["n_search_configs_per_set"]
        log(f"Laufzeitschaetzung: {estimate['estimated_hours']:.2f} h fuer {count} Konfigurationen je Set. "
            f"Ziel: ca. {args.target_hours:g} h; keine automatische Zeitgrenze.")
        if estimate["estimated_hours"] > planning_target:
            log("Die Prognose liegt ueber dem Ziel. Der festgelegte Lauf wird trotzdem vollstaendig ausgefuehrt.")
    configs = subset_configs(pool, count)
    plan = {"version": VERSION, "mode": mode, "n_search_configs_per_set": count,
            "n_configurations": len(configs), "target_hours": args.target_hours,
            "time_policy": "planning_target_only", "automatic_time_limit": False,
            "operations": operation_counts(count, args),
            "estimated_hours": None if estimate is None else estimate["estimated_hours"],
            "cost_detail": estimate, "cost_models": models, "configurations": [asdict(c) for c in configs]}
    atomic_json(path, plan)
    atomic_json(output/"operation_plan.json", plan["operations"])
    atomic_csv(output/"hpo_configurations.csv", [c.row() for c in configs])
    log(f"Fester Suchplan: {count} Konfigurationen je Set, {len(configs)} insgesamt pro Outer-Fold.")
    return configs, plan


def promote(configs, rows, stage, args):
    by_id, result, details = {c.config_id: c for c in configs}, [], []
    for condition in CONDITIONS:
        group = [r for r in rows if r["condition_id"] == condition and r.get("complete_inner_cv")]
        expected = sum(c.condition == condition for c in configs)
        if len(group) != expected or not group:
            raise RuntimeError("Keine Beforderung aus unvollstaendiger innerer CV: "+condition)
        ordered = sorted(group, key=lambda r: (-float(r["mean_inner_cohen_kappa"]),
                         r["trainable_parameters"], tuple(-v for v in candidate_rank(r, args.criterion)), r["config_id"]))
        chosen = ordered[:min(2 if stage == "broad" else 1, len(ordered))]
        ids = {r["config_id"] for r in chosen}
        result.extend(by_id[r["config_id"]] for r in chosen)
        details.extend({**r, "promoted": r["config_id"] in ids} for r in ordered)
    return result, details


def fit_outer_ensemble(records, args, factory, cache, output, config, context, train, test,
                       inner_splits, candidate_row, selected_for_overall):
    repeat, split_seed, fold = context
    label = f"W{repeat}_F{fold}_" + config.channel_set
    completed = read_receipt(output, "outer", label, "outer")
    if completed is not None:
        return completed
    validation = [records[i] for i in test]
    predictions, members = [], []
    for initialization in args.train_seeds:
        inner_epochs = []
        for inner_fold in range(1, len(inner_splits)+1):
            job = f"W{repeat}_F{fold}_I{inner_fold}_S{initialization}_" + config.config_id
            row = read_receipt(output, "inner_fit", job, "final")
            if row is None:
                raise RuntimeError("Finale innere Epochenzahl fehlt: " + job)
            inner_epochs.append(row["best_epoch"])
        epochs = max(config.warmup_epochs+1, int(round(np.median(inner_epochs))))
        job = label + f"_S{initialization}_" + config.config_id
        if read_receipt(output, "outer_member", job, "outer") is None and args.cost_models:
            args.runtime_budget.check(cost_for_fit(config, [records[i] for i in train], validation,
                                                  args, args.cost_models, epochs))
        # inner=False: keine aeussere Validierung waehrend Training/Epochenwahl.
        row = fit_trial([records[i] for i in train], validation, config, args, factory, cache, output,
                        job, stable_seed(initialization, split_seed, fold, "outer_model"),
                        stable_seed(args.seed, split_seed, fold, "outer_views"), "outer", epochs, False,
                        {"outer_repeat": repeat, "split_seed": split_seed, "outer_fold": fold,
                         "initialization_seed": initialization, "method_id": config.channel_set})
        predictions.append(load_predictions(output, job, "outer", validation))
        members.append(row)
    metrics = metrics_for(validation, np.mean(predictions, axis=0), config.task)
    result = {"outer_repeat": repeat, "split_seed": split_seed, "outer_fold": fold,
              "method_id": config.channel_set, **config.row(), "stage": "outer",
              "decision_threshold": .5, "threshold_source": candidate_row["threshold_source"],
              "n_ensemble_members": len(members), "n_train_persons": len(train), "n_eval_persons": len(test),
              "mean_inner_primary_metric": candidate_row["mean_inner_cohen_kappa"],
              "selected_for_overall_hpo": selected_for_overall,
              "median_outer_epochs": float(np.median([row["epochs_run"] for row in members])),
              "trainable_parameters": members[0]["trainable_parameters"],
              "ensemble_trainable_parameters": members[0]["trainable_parameters"] * len(members),
              "encoder_parameters": members[0]["encoder_parameters"],
              "cnn_trainable_parameters": members[0]["cnn_trainable_parameters"],
              "duration_seconds": sum(row["duration_seconds"] for row in members), **metrics}
    atomic_json(output / "outer_models" / (label + "_ensemble.json"), {
        "version": VERSION, "config": asdict(config), "objective": "cohen_kappa",
        "angle_class_thresholds_deg": ANGLE_CLASS_THRESHOLDS_DEG, "comparison": "strictly_greater",
        "training_initializations": args.train_seeds, "precision": args.resolved_precision,
        "member_files": [label + f"_S{seed}_" + config.config_id + ".pt" for seed in args.train_seeds],
        "aggregation": "mean_contact_angles_per_person_then_mean_models"})
    atomic_json(receipt_path(output, "outer", label, "outer"), result)
    return clean_json(result)


def run_hpo(records, args, factory):
    output = args.output_dir
    contexts = list(experiment_splits(records, args))
    validate_nested_contexts(records, contexts, args)
    pool = sample_configs(args.search_configs or args.max_search_configs, args.search_seed)
    names = ("split_seeds", "train_seeds", "outer_folds", "inner_folds", "split_candidates",
             "criterion", "search_seed", "seed", "noise_sigma", "search_configs", "min_search_configs",
             "max_search_configs", "target_hours", "reserve_hours", "pilot_max_hours", "pilot_epochs",
             "pilot_persons", "runtime_safety_factor", "broad_epochs", "stable_epochs", "max_epochs",
             "patience", "stable_patience", "person_batch_size", "contact_microbatch", "head_microbatch",
             "feature_cache_gb", "force_threshold", "contact_duration_tolerance", "contact_zero_tolerance",
             "min_si_pairs_per_subject", "smoke_test")
    contract = {"time_policy": "planning_target_only", "version": VERSION, "dataset_hash": dataset_fingerprint(records),
                "weights": factory.fingerprint, "weights_source": factory.provenance,
                "torch_version": str(torch.__version__) if torch else None,
                "cuda_build": str(torch.version.cuda) if torch else None,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "channel_sets": CHANNEL_SETS, **{n: getattr(args, n) for n in names}}
    config_path = output/"run_config.json"
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise RuntimeError("Lauf existiert. Zum Fortsetzen --resume verwenden.")
        if saved_config["contract"] != clean_json(contract):
            if not upgrade_before_hpo(output, saved_config, contract):
                raise RuntimeError("Daten, Code, Gewichte oder Einstellungen geaendert oder HPO bereits begonnen. "
                                   "Neuen --output-dir verwenden. Abgeschlossene Fits werden nicht vermischt.")
    else:
        atomic_json(config_path, {"contract": contract, "device": args.device, "python": sys.version})
    if args.runtime_budget is None:
        args.runtime_budget = RuntimeBudget(output, args)
    args.runtime_contexts = contexts
    args.runtime_budget.in_pilot = not (output/"search_plan.json").exists() and not args.smoke_test
    atomic_json(output/"precision_plan.json", {"resolved_precision": "fp32", "tf32": False, "bf16": False})
    atomic_csv(output/"channel_set_manifest.csv", [
        {"channel_set": s, "n_active_channels": len(indices),
         "source_indices_zero_based": ",".join(str(SELECTED_SOURCE_CHANNEL_INDICES[i]) for i in indices),
         "mask_SI04_SI05_SI09": ",".join(str(int(i in indices)) for i in range(3))}
        for s, indices in CHANNEL_SETS.items()])
    counts = []
    for repeat, seed, fold, train, test, splits in contexts:
        counts.extend(split_count_rows(records, train, seed, fold, "outer_train"))
        counts.extend(split_count_rows(records, test, seed, fold, "outer_test"))
        for inner_fold, (it, iv) in enumerate(splits, 1):
            counts.extend(split_count_rows(records, it, seed, fold, "inner_train", inner_fold))
            counts.extend(split_count_rows(records, iv, seed, fold, "inner_validation", inner_fold))
    atomic_csv(output/"split_counts.csv", counts)
    cache = FrozenFeatureCache(args.feature_cache_gb, output/"_feature_cache"/"current_outer")
    configs, plan = ensure_search_plan(records, args, factory, cache, contexts, pool, output)
    args.runtime_budget.in_pilot = False
    args.cost_models, args.runtime_configs = plan.get("cost_models", {}), configs
    if args.pilot_only:
        atomic_json(output/"run_status.json", {"status": "pilot_complete", "complete_nested_cv": False,
                    "configurations_per_channel_set": plan["n_search_configs_per_set"]})
        return None
    for context_index, (repeat, seed, fold, train, test, splits) in enumerate(contexts, 1):
        context, name = (repeat, seed, fold), f"W{repeat}_F{fold}"
        if read_receipt(output, "outer", name+"_hpo_inner_selected", "outer") is not None:
            if not all(read_receipt(output, "outer", name+"_"+s, "outer") is not None for s in CHANNEL_SETS):
                raise RuntimeError("Gespeicherter Outer-Kontext ist unvollstaendig.")
            log(f"Resume {name}: alle sieben aeusseren Kanalset-Tests abgeschlossen.")
            continue
        log(f"OUTER {context_index}/{len(contexts)} {name}: {len(train)} Trainings-, {len(test)} Testpersonen.")
        active, final_rows = configs, {}
        for stage, epochs in (("broad", args.broad_epochs), ("stable", args.stable_epochs), ("final", args.max_epochs)):
            initializations, current_rows = stage_initializations(args, stage), []
            ordered = sorted(active, key=lambda c: (c.channel_set, c.rare_angle_copy_cap, c.architecture, c.config_id))
            log(f"  {stage}: {len(ordered)} Kandidaten x {args.inner_folds} Inner-Folds x {len(initializations)} Initialisierungen; bis Epoche {epochs}.")
            for index, cfg in enumerate(ordered, 1):
                if read_receipt(output, "candidate", name+"_"+cfg.config_id, stage) is None:
                    check_candidate_budget(records, args, cfg, context, splits, stage, epochs)
                row = evaluate_candidate(records, args, factory, cache, output, cfg, context, train, splits,
                                         stage, epochs, initializations)
                current_rows.append(row)
                log(f"  {index}/{len(ordered)} {cfg.channel_set}/{cfg.architecture}: Inner-Kappa={row['mean_inner_cohen_kappa']:.4f}.")
                atomic_json(output/"run_status.json", {"status": "running", "context": name, "stage": stage,
                    "stage_completed": index, "stage_total": len(ordered), "complete_nested_cv": False,
                    "target_hours": args.target_hours, "automatic_time_limit": False})
            active, details = promote(active, current_rows, stage, args)
            atomic_json(receipt_path(output, "selection", name, stage), {
                "survivor_ids": [c.config_id for c in active], "rows": details, "criterion": "inner_cohen_kappa"})
            if stage == "final":
                final_rows = {r["config_id"]: r for r in current_rows}
            cleanup_checkpoints(output, name, active)
            aggregate_reports(output, args)
        winners, decisions = [], []
        for channel_set in CHANNEL_SETS:
            best, rule = choose_smallest([final_rows[c.config_id] for c in active if c.channel_set == channel_set], args)
            winners.append(next(c for c in active if c.config_id == best["config_id"]))
            decisions.append({"outer_repeat": repeat, "split_seed": seed, "outer_fold": fold,
                              "channel_set": channel_set, "config_id": best["config_id"],
                              "architecture": best["architecture"], "inner_cohen_kappa": best["mean_inner_cohen_kappa"]})
        overall, rule = choose_smallest([final_rows[c.config_id] for c in winners], args)
        for d in decisions:
            d["selected_for_overall"] = d["config_id"] == overall["config_id"]
        # Persist all seven choices and the global choice BEFORE any outer evaluation.
        atomic_json(receipt_path(output, "channel_selection", name), {"rule": rule, "rows": decisions})
        selected = None
        for cfg in winners:
            row = fit_outer_ensemble(records, args, factory, cache, output, cfg, context, train, test,
                splits, final_rows[cfg.config_id], cfg.config_id == overall["config_id"])
            if cfg.config_id == overall["config_id"]:
                selected = dict(row, method_id="hpo_inner_selected", duration_seconds=0.)
            log(f"  AEUSSERER TEST {name}/{cfg.channel_set}: Kappa={row['cohen_kappa']:.4f}; MAE={row['angle_mae_deg']:.2f} Grad.")
        if selected is None or len(winners) != 7:
            raise AssertionError("Sieben aeussere Tests und ein innen gewaehltes Gesamtverfahren erforderlich.")
        atomic_json(receipt_path(output, "outer", name+"_hpo_inner_selected", "outer"), selected)
        cleanup_checkpoints(output, name, [])
        gc.collect()
        cache.clear()
        aggregate_reports(output, args)
    result = aggregate_reports(output, args)
    expected = len(contexts)*8
    if len(all_receipts(output, "outer")) != expected or result is None or not result.complete_nested_cv.all():
        raise AssertionError("Unvollstaendige aeussere CV.")
    atomic_json(output/"run_status.json", {"status": "completed", "complete_nested_cv": True,
        "n_outer_contexts": len(contexts), "outer_result_rows": expected, "primary_method": "hpo_inner_selected"})
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", nargs="?", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results_gtx1060_10h_newseeds"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights-dir", type=Path, default=Path("pretrained_weights_v36"))
    p.add_argument("--target-hours", type=float, default=10., help="Laufzeitziel zur Wahl des Suchumfangs; keine Abbruchfrist.")
    p.add_argument("--reserve-hours", type=float, default=1., help=argparse.SUPPRESS)
    p.add_argument("--pilot-max-hours", type=float, default=.5, help=argparse.SUPPRESS)
    p.add_argument("--pilot-epochs", type=int, default=3)
    p.add_argument("--pilot-persons", type=int, default=12)
    p.add_argument("--runtime-safety-factor", type=float, default=1.2)
    p.add_argument("--search-configs", type=int, default=0, help="Je Kanalset: 0=Pilot waehlt 8..64 nach Laufzeitschaetzung. Feste Zahl >=8, durch 4 teilbar; wird unabhaengig von der Dauer ausgefuehrt.")
    p.add_argument("--min-search-configs", type=int, default=8)
    p.add_argument("--max-search-configs", type=int, default=64)
    p.add_argument("--split-seeds", type=int, nargs="+", default=[101, 307, 911])
    p.add_argument("--train-seeds", type=int, nargs="+", default=[1709, 2903])
    p.add_argument("--outer-folds", type=int, default=3)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--split-candidates", type=int, default=256)
    p.add_argument("--broad-epochs", type=int, default=4)
    p.add_argument("--stable-epochs", type=int, default=12)
    p.add_argument("--max-epochs", type=int, default=25)
    p.add_argument("--stable-patience", type=int, default=4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--person-batch-size", type=int, default=8)
    p.add_argument("--contact-microbatch", type=int, default=4, help="CNN-Featureextraktion, OOM-Halbierung bis 1.")
    p.add_argument("--head-microbatch", type=int, default=256, help="Kleine trainierbare Koepfe auf gecachten Features.")
    p.add_argument("--feature-cache-gb", type=float, default=8., help="Festplattenlimit des CPU-Memmap-Caches, NICHT GPU-Speicher.")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--log-every", type=int, default=4)
    p.add_argument("--noise-sigma", type=float, default=.02)
    p.add_argument("--search-seed", type=int, default=93871)
    p.add_argument("--seed", type=int, default=4243)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--force-threshold", type=float, default=0.)
    p.add_argument("--contact-duration-tolerance", type=float, default=.25)
    p.add_argument("--contact-zero-tolerance", type=float, default=0.)
    p.add_argument("--min-si-pairs-per-subject", type=int, default=3)
    p.add_argument("--extend-deadline-hours", type=float, default=0., help=argparse.SUPPRESS)
    for name in ("resume", "pilot-only", "smoke-test", "logic-test", "self-test", "check-model"):
        p.add_argument("--"+name, action="store_true")
    args = p.parse_args(argv)
    if args.data_dir is None and not (args.logic_test or args.self_test or args.check_model):
        p.error("Pfad zum CSV_Export-Ordner erforderlich.")
    if args.outer_folds < 2 or args.inner_folds < 2:
        p.error("Immer Nested-CV: mindestens zwei aeussere und zwei innere Folds.")
    if args.search_configs and (args.search_configs < 8 or args.search_configs % 4):
        p.error("search-configs: 0 oder >=8 und durch vier teilbar, jeweils PRO KANALSET.")
    if args.min_search_configs < 8 or args.min_search_configs % 4 or args.max_search_configs % 4 or args.max_search_configs < args.min_search_configs:
        p.error("Suchgrenzen muessen >=8, durch vier teilbar und aufsteigend sein.")
    if not 1 <= args.broad_epochs <= args.stable_epochs <= args.max_epochs:
        p.error("1 <= broad-epochs <= stable-epochs <= max-epochs erforderlich.")
    hours = [args.target_hours, args.reserve_hours, args.pilot_max_hours, args.extend_deadline_hours]
    if not all(math.isfinite(v) and v >= 0 for v in hours) or args.target_hours <= 0:
        p.error("Laufzeitziel muss endlich und >0 sein.")
    if min(args.pilot_epochs, args.stable_patience, args.patience, args.person_batch_size, args.contact_microbatch,
           args.head_microbatch, args.torch_threads, args.log_every, args.split_candidates, args.top_n, args.min_si_pairs_per_subject) < 1 or args.pilot_epochs < 3 or args.pilot_persons < 4:
        p.error("Positive Anzahlen erforderlich; pilot-epochs >=3, pilot-persons >=4.")
    if not math.isfinite(args.feature_cache_gb) or args.feature_cache_gb <= 0 or not math.isfinite(args.runtime_safety_factor) or args.runtime_safety_factor < 1:
        p.error("feature-cache-gb >0, runtime-safety-factor >=1 erforderlich.")
    if not math.isfinite(args.noise_sigma) or args.noise_sigma <= 0:
        p.error("noise-sigma muss endlich und >0 sein.")
    for seeds in (args.split_seeds, args.train_seeds):
        if not seeds or len(set(seeds)) != len(seeds):
            p.error("Seedlisten muessen eindeutig sein.")
    if any(v < 0 or v >= 2**32 for v in args.split_seeds+args.train_seeds+[args.seed, args.search_seed]):
        p.error("Seeds muessen in 0..2**32-1 liegen.")
    if args.pilot_only and args.smoke_test:
        p.error("pilot-only und smoke-test nicht kombinieren.")
    if args.smoke_test:
        args.outer_folds = args.inner_folds = 2
        args.split_seeds, args.train_seeds = args.split_seeds[:1], args.train_seeds[:1]
        args.split_candidates, args.search_configs = 16, 8
        args.broad_epochs = args.stable_epochs = args.max_epochs = 1
    args.criterion, args.simplicity_gap = "cohen_kappa", 0.
    args.precision, args.resolved_precision, args.tf32 = "fp32", "fp32", "off"
    args.runtime_budget, args.cost_models = None, {}
    args.run_started_wall = time.time()
    return args


def numpy_image_coverage(width):
    basis = np.eye(101)
    source, destination = np.arange(101), np.linspace(0, 100, width)
    return np.asarray([np.interp(destination, source, row).sum() > 0 for row in basis])


def create_test_csvs(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        curve = ("TH", "L", "THL")[i % 3]
        angle = ANGLE_CLASS_THRESHOLDS_DEG[curve] + (7 if i % 2 else -7)
        times = np.arange(380) / SAMPLING_RATE_HZ
        forces = {side: np.zeros(len(times)) for side in ("left", "right")}
        for contact, start in enumerate((10, 100, 190, 280)):
            pulse = np.sin(np.linspace(0, math.pi, 62))[1:-1] * (600 + 10 * i)
            forces["left" if contact % 2 == 0 else "right"][start:start + len(pulse)] = pulse
        pd.DataFrame({"time": times, "force_left": forces["left"], "force_right": forces["right"],
                      "cop_left_x": np.sin(times * 2.1) * (1 + i / 12), "cop_left_y": times + np.sin(times),
                      "cop_right_x": np.sin(times * 2.3) * (1 + i / 13), "cop_right_y": times + np.cos(times),
                      "cop_total_x": np.sin(times * 2.2), "cop_total_y": times + np.sin(times * .5)}).to_csv(
            folder / f"SYNTHETIC{i:02d}-test-{angle:g}-{curve}-measurement.csv", index=False)


def logic_test():
    import contextlib
    import io
    log("Technischer Logiktest ohne Torch, ohne Training auf echten Daten ...")
    defaults = parse_args(["SYNTHETIC"])
    assert defaults.target_hours == 10 and defaults.outer_folds == defaults.inner_folds == 3
    assert defaults.split_seeds == [101, 307, 911] and defaults.train_seeds == [1709, 2903]
    assert defaults.search_seed == 93871 and defaults.seed == 4243
    assert not ({*defaults.split_seeds, *defaults.train_seeds, defaults.search_seed, defaults.seed} & {42, 11, 2027, 91725})
    assert SELECTED_SOURCE_CHANNEL_INDICES == (3, 4, 8) and RAW_COP_INDICES == ()
    assert set(CHANNEL_SETS.values()) == set(itertools.chain.from_iterable(
        itertools.combinations(range(3), n) for n in (1, 2, 3)))
    for n in (8, 12, 16, 64):
        pool = sample_configs(n, 93871)
        assert len(pool) == n*7 and len({c.config_id for c in pool}) == len(pool)
        assert set(Counter(c.condition for c in pool).values()) == {n//2}
        templates = [{c.template_id for c in pool if c.channel_set == s} for s in CHANNEL_SETS]
        assert all(t == templates[0] for t in templates)
        assert all(c.transfer_mode == "frozen" and c.task == "angle_regression" for c in pool)
    assert {c.config_id for c in sample_configs(8, 93871)} <= {c.config_id for c in sample_configs(16, 93871)}
    assert all(numpy_image_coverage(160))
    for invalid in (("--outer-folds", "1"), ("--inner-folds", "1")):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parse_args(["SYNTHETIC", *invalid])
                raise AssertionError("Einfacher Holdout akzeptiert.")
            except SystemExit as e:
                assert e.code == 2
    records = synthetic_records()
    assert all(r.pairs.shape[1:] == (3, 101) for r in records)
    default_contexts = list(experiment_splits(records, defaults))
    validate_nested_contexts(records, default_contexts, defaults)
    assert len(default_contexts) == 9
    assert {c[1] for c in default_contexts} == {101, 307, 911}
    repeated_test_partitions = {tuple(sorted(tuple(sorted(map(int, c[4])))
        for c in default_contexts if c[1] == seed)) for seed in defaults.split_seeds}
    assert len(repeated_test_partitions) == 3
    cfg = next(c for c in sample_configs(8, 93871) if c.channel_set == "SI04")
    original = prepare_arrays(records[:16], records[16:], cfg, defaults, 99)
    changed_signals = []
    for r in records:
        measurements = []
        for m in r.measurements:
            values = m.pairs.copy()
            values[:, 1:] = np.nan
            measurements.append(replace(m, pairs=values))
        changed_signals.append(replace(r, measurements=measurements))
    altered = prepare_arrays(changed_signals[:16], changed_signals[16:], cfg, defaults, 99)
    assert np.array_equal(original[0], altered[0]) and np.array_equal(original[1], altered[1])
    assert not np.any(original[0][:, 1:]) and not np.any(original[1][:, 1:])
    full = replace(cfg, channel_set="SI04_SI05_SI09")
    full_arrays = prepare_arrays(records[:16], records[16:], full, defaults, 99)
    assert np.array_equal(original[0][:, 0], full_arrays[0][:, 0])
    truth = np.asarray([r.hard_reference for r in records])
    pred = truth.copy()
    for label in (0, 1):
        pred[np.flatnonzero(truth == label)[:3]] = 1-label
    angles = np.asarray([r.boundary_deg for r in records])+np.where(pred, 7., -7.)
    assert abs(metrics_for(records, angles)["cohen_kappa"]-.5) < 1e-12
    assert resumed_stale("broad", "stable", 9) == 0 and resumed_stale("stable", "stable", 9) == 9
    with tempfile.TemporaryDirectory(prefix="gtx1060_v38_logic_") as directory:
        folder = Path(directory)
        clock = [1000.]
        bargs = copy.copy(defaults)
        bargs.run_started_wall = clock[0]
        b = RuntimeBudget(folder/"clock", bargs, clock=lambda: clock[0])
        clock[0] += 100*3600
        b.in_pilot = True
        b.check(estimated_seconds=10**12, mandatory_seconds=10**12)
        b.in_pilot = False
        b.check()
        assert math.isinf(b.remaining_seconds) and b.state["deadline_unix"] is None
        choices = [{"n_search_configs_per_set": 8, "estimated_hours": 16.99},
                   {"n_search_configs_per_set": 12, "estimated_hours": 20.}]
        chosen, mode = choose_operation_count(choices, 10.)
        assert chosen["n_search_configs_per_set"] == 8 and mode == "minimum_operations_above_time_target"
        assert choose_operation_count(choices, 10., 12)[0]["n_search_configs_per_set"] == 12
        assert operation_counts(8, defaults)["head_transformer_epochs_upper_bound_total"] == 29610
        assert operation_counts(8, defaults)["distinct_trainable_model_initializations"] == 2016
        create_test_csvs(folder/"CSV_Export")
        with contextlib.redirect_stdout(io.StringIO()):
            imported, _, _, _ = load_standalone_si15_records(folder/"CSV_Export")
            saved_indices, saved_channels = SELECTED_SOURCE_CHANNEL_INDICES, EXPECTED_CHANNELS
            globals()["SELECTED_SOURCE_CHANNEL_INDICES"], globals()["EXPECTED_CHANNELS"] = tuple(range(15)), 15
            try:
                si15, _, _, _ = load_standalone_si15_records(folder/"CSV_Export")
            finally:
                globals()["SELECTED_SOURCE_CHANNEL_INDICES"], globals()["EXPECTED_CHANNELS"] = saved_indices, saved_channels
        assert len(imported) == 12
        assert all(np.array_equal(a.pairs, b.pairs[:, [3, 4, 8]], equal_nan=True) for a, b in zip(imported, si15))
        args = parse_args(["SYNTHETIC", "--device", "cpu", "--split-seeds", "101", "--outer-folds", "2", "--inner-folds", "2",
            "--split-candidates", "8", "--search-configs", "8", "--broad-epochs", "1", "--stable-epochs", "2", "--max-epochs", "3"])
        contexts = list(experiment_splits(records, args))
        jobs, stopped = [], [False]
        original_fit, original_atomic = globals()["fit_trial"], globals()["atomic_json"]

        class TestFactory:
            fingerprint, provenance = "numpy_v38", "SYNTHETIC_ORCHESTRATION_ONLY"

        def simulated_fit(train, evaluation, config, args, factory, cache, output,
                          job, seed, data_seed, stage, epochs, inner, metadata, patience=0):
            section = "inner_fit" if inner else "outer_member"
            completed = read_receipt(output, section, job, stage)
            if completed is not None:
                return completed
            assert not ({r.subject for r in train} & {r.subject for r in evaluation})
            if not inner:
                name = f"W{metadata['outer_repeat']}_F{metadata['outer_fold']}"
                assert read_receipt(output, "channel_selection", name) is not None
            key = (str(output), job, stage)
            assert key not in jobs
            jobs.append(key)
            arrays = prepare_arrays(train, evaluation, config, args, data_seed)
            active = CHANNEL_SETS[config.channel_set]
            signal = np.asarray([np.std(arrays[1][arrays[3] == i][:, active, :]) for i in range(len(evaluation))])
            center = np.mean([r.angle_deg-r.boundary_deg for r in train])
            values = np.asarray([r.boundary_deg for r in evaluation])+(signal-.55)*15+.02*center
            values += np.random.default_rng(seed).normal(0, .1, len(evaluation))
            row = {**metadata, **config.row(), "stage": stage, "best_epoch": min(epochs, 2), "epochs_run": epochs,
                   "duration_seconds": .003+.01*epochs, "q75_epoch_seconds": .01, "startup_seconds": .003,
                   "feature_seconds": .001, "train_contacts": len(arrays[0]), "eval_contacts": len(arrays[1]),
                   "trainable_parameters": 100+config.transformer_width, "total_parameters": 1000,
                   "cnn_trainable_parameters": 0, "encoder_parameters": config.transformer_width,
                   "small_component_parameters": 100+config.transformer_width,
                   **metrics_for(evaluation, values)}
            atomic_npz(prediction_path(output, job, stage), predictions=values,
                       evaluation_fingerprint=np.asarray(dataset_fingerprint(evaluation)))
            atomic_json(receipt_path(output, section, job, stage), row)
            return clean_json(row)

        def stop_after_one_outer(path, value):
            original_atomic(path, value)
            if path.parent.name == "outer" and value.get("method_id") in CHANNEL_SETS and not stopped[0]:
                stopped[0] = True
                raise KeyboardInterrupt("Technischer Resume-Test.")

        globals()["fit_trial"] = simulated_fit
        try:
            args.output_dir = folder/"run"
            globals()["atomic_json"] = stop_after_one_outer
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    run_hpo(records, args, TestFactory())
                except KeyboardInterrupt:
                    pass
                assert stopped[0]
                globals()["atomic_json"] = original_atomic
                args.resume = True
                summary = run_hpo(records, args, TestFactory())
                n_jobs = len(jobs)
                before = (args.output_dir/"outer_fold_metrics.csv").read_bytes()
                run_hpo(records, args, TestFactory())
            assert len(summary) == 8 and summary.complete_nested_cv.all()
            seed_summary = pd.read_csv(args.output_dir/"seed_summary.csv", sep=";")
            assert len(seed_summary) == 8 and seed_summary.complete_outer_repeat.all()
            assert len(jobs) == n_jobs and before == (args.output_dir/"outer_fold_metrics.csv").read_bytes()
            outer = pd.read_csv(args.output_dir/"outer_fold_metrics.csv", sep=";")
            assert len(outer) == 16 and outer.n_ensemble_members.eq(2).all()
            assert len(all_receipts(args.output_dir, "outer_member")) == 28
            for row in all_receipts(args.output_dir, "candidate"):
                assert row["n_training_initializations"] == (2 if row["stage"] == "final" else 1)
            plan = json.loads((args.output_dir/"search_plan.json").read_text())
            assert len(plan["cost_models"]) == 14 and plan["n_configurations"] == 56
            # Independent epoch-count check: all fits cost one second/epoch, no startup.
            dummy = {k: dict(v, epoch_seconds=1., startup_seconds=0., train_contacts=10**9, eval_contacts=10**9)
                     for k, v in plan["cost_models"].items()}
            count_args = copy.copy(defaults)
            count_args.runtime_safety_factor = 1.
            estimate = estimated_hours(12, dummy, count_args, list(experiment_splits(records, defaults)), records, sample_configs(12, 93871))
            assert abs(estimate["estimated_hours"]*3600-32634) < 1e-8
            old_estimator = globals()["estimated_hours"]
            def long_estimate(*positional, **keyword):
                result = old_estimator(*positional, **keyword)
                return dict(result, estimated_hours=16.99)
            globals()["estimated_hours"] = long_estimate
            overrun = copy.copy(args)
            overrun.output_dir, overrun.resume, overrun.runtime_budget = folder/"overrun", False, None
            overrun.search_configs, overrun.min_search_configs, overrun.max_search_configs = 0, 8, 8
            overrun.pilot_only = True
            overrun.run_started_wall = time.time()-100*3600
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_hpo(records, overrun, TestFactory())
                planned = json.loads((overrun.output_dir/"search_plan.json").read_text())
                assert planned["n_search_configs_per_set"] == 8 and planned["estimated_hours"] == 16.99
                assert planned["automatic_time_limit"] is False
                old_config = json.loads((overrun.output_dir/"run_config.json").read_text())
                old_config["contract"].update(version="38.1.0", script_sha256=LEGACY_V3810_SOURCE_SHA256)
                old_config["contract"].pop("time_policy")
                atomic_json(overrun.output_dir/"run_config.json", old_config)
                (overrun.output_dir/"search_plan.json").unlink()
                pilot_jobs_before = len([j for j in jobs if j[0] == str(overrun.output_dir/"_runtime_pilot")])
                assert pilot_jobs_before == 14
                overrun.resume, overrun.pilot_only, overrun.runtime_budget = True, False, None
                with contextlib.redirect_stdout(io.StringIO()):
                    overrun_result = run_hpo(records, overrun, TestFactory())
                assert overrun_result.complete_nested_cv.all()
                assert len([j for j in jobs if j[0] == str(overrun.output_dir/"_runtime_pilot")]) == pilot_jobs_before
                assert (overrun.output_dir/"_migrations"/"v38_1_0_before_time_fix"/"run_config.json").exists()
                assert upgrade_before_hpo(overrun.output_dir, old_config,
                    json.loads((overrun.output_dir/"run_config.json").read_text())["contract"]) is False
            finally:
                globals()["estimated_hours"] = old_estimator
            altered_records = list(records)
            for i in contexts[0][4]:
                r = records[i]
                altered_records[i] = replace(r, angle_deg=r.boundary_deg+(-7 if r.hard_reference else 7),
                                             hard_reference=1-r.hard_reference)
            changed_args = copy.copy(args)
            changed_args.output_dir, changed_args.resume, changed_args.runtime_budget = folder/"changed_outer_truth", False, None
            original_splits = globals()["experiment_splits"]
            globals()["experiment_splits"] = lambda records, args: iter(contexts)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_hpo(altered_records, changed_args, TestFactory())
            finally:
                globals()["experiment_splits"] = original_splits
            changed_outer = pd.read_csv(changed_args.output_dir/"outer_fold_metrics.csv", sep=";")
            a = outer[outer.outer_fold.eq(1)].sort_values("method_id")
            b = changed_outer[changed_outer.outer_fold.eq(1)].sort_values("method_id")
            assert a.config_id.tolist() == b.config_id.tolist()
            assert a.median_outer_epochs.tolist() == b.median_outer_epochs.tolist()
            assert a.mean_inner_primary_metric.tolist() == b.mean_inner_primary_metric.tolist()
            for p in args.output_dir.rglob("*.csv"):
                assert "SYNTHETIC_PERSON" not in p.read_text(encoding="utf-8-sig")
        finally:
            globals()["fit_trial"], globals()["atomic_json"] = original_fit, original_atomic
    log("LOGIKTEST BESTANDEN: sieben Sets, SI15-Zuordnung, Maskierung, 3 Wiederholungen x 3x3-Nested-CV, Kappa, Pilot, weiches Laufzeitziel, hohe ETA ohne Abbruch, Pilot-Migration, Resume und aeussere Auswahltrennung.")


def self_test():
    """Small real Torch fits with a synthetic backbone; no external weights."""
    logic_test()
    require_torch()
    torch.set_num_threads(2)
    args = parse_args(["SYNTHETIC", "--device", "cpu", "--smoke-test"])
    records, factory = synthetic_records(), BackboneFactory(args, synthetic=True)
    pool = sample_configs(8, args.search_seed)
    for condition in CONDITIONS:
        cfg = next(c for c in pool if c.condition == condition)
        x = torch.zeros(2, 3, 101)
        assert signal_images(x, 160).shape == (2, 3, 96, 160)
        model = TransferModel(factory, cfg)
        prediction = model(x)
        assert torch.isfinite(prediction).all()
        prediction.sum().backward()
        assert not any(p.grad is not None for p in model.backbone.parameters())
        assert parameter_breakdown(model)["encoder_parameters"] == cfg.row()["encoder_parameters_formula"]
    del model
    with tempfile.TemporaryDirectory(prefix="gtx1060_v38_torch_") as directory:
        args.output_dir = Path(directory)/"hpo"
        result = run_hpo(records, args, factory)
        assert result.complete_nested_cv.all() and len(result) == 8
        assert len(all_receipts(args.output_dir, "outer")) == 16
        args.resume = True
        before = (args.output_dir/"outer_fold_metrics.csv").read_bytes()
        run_hpo(records, args, factory)
        assert before == (args.output_dir/"outer_fold_metrics.csv").read_bytes()
    log("TORCH-SELBSTTEST BESTANDEN: Forward/Backward, CPU-Memmap-Features, Nested-CV und Resume. Keine CUDA-/Patientengueteaussage.")


def main(argv=None):
    args = parse_args(argv)
    if args.logic_test:
        logic_test()
        return 0
    if args.self_test:
        self_test()
        return 0
    require_torch()
    torch.set_num_threads(args.torch_threads)
    args.device = choose_device(args.device)
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.check_model:
        factory = BackboneFactory(args)
        pool = sample_configs(8, args.search_seed)
        for condition in CONDITIONS:
            cfg = next(c for c in pool if c.condition == condition)
            model = TransferModel(factory, cfg).to(args.device)
            x = torch.zeros(2, 3, 101, device=args.device)
            result = model(x)
            assert result.shape == (2,) and torch.isfinite(result).all()
            result.sum().backward()
            assert not any(p.requires_grad for p in model.backbone.parameters())
            log(f"Modelltest bestanden: {cfg.channel_set}/{cfg.architecture}.")
            del model
        return 0
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"CSV_Export-Ordner nicht gefunden: {args.data_dir.resolve()}")
    with output_lock(args.output_dir, args.resume):
        if not args.resume and any(p.name != "run.lock" for p in args.output_dir.iterdir()):
            raise RuntimeError("Ergebnisordner ist nicht leer. Neuen Ordner waehlen oder --resume verwenden.")
        try:
            args.runtime_budget = RuntimeBudget(args.output_dir, args)
            args.runtime_budget.in_pilot = not (args.output_dir/"search_plan.json").exists() and not args.smoke_test
            args.runtime_budget.check()
            records, dataset, audit, source = load_standalone_si15_records(
                args.data_dir, args.force_threshold, args.contact_duration_tolerance,
                args.contact_zero_tolerance, args.min_si_pairs_per_subject, budget=args.runtime_budget)
            atomic_csv(args.output_dir/"input_audit.csv", audit)
            atomic_json(args.output_dir/"cohort_qc_summary.json", {
                "n_retained_persons": len(records), "n_retained_contact_pairs": sum(r.n_pairs for r in records),
                "n_measurements": sum(len(r.measurements) for r in records),
                "n_failed_files": len(dataset["import_failures"]), "shared_cohort_all_seven_sets": True})
            args.runtime_budget.check()
            factory = BackboneFactory(args)
            result = run_hpo(records, args, factory)
        except BaseException as error:
            if isinstance(error, SystemExit):
                raise
            previous = {}
            status = args.output_dir/"run_status.json"
            if status.exists():
                previous = json.loads(status.read_text(encoding="utf-8"))
            state = "paused_deadline" if isinstance(error, BudgetExceeded) else "paused_user" if isinstance(error, KeyboardInterrupt) else "failed"
            atomic_json(status, {**previous, "status": state, "complete_nested_cv": False,
                                 "reason": str(error), "version": VERSION})
            aggregate_reports(args.output_dir, args)
            if isinstance(error, (KeyboardInterrupt, BudgetExceeded)):
                log(f"PAUSIERT: {error}. Kein vollstaendiger Nested-CV-Endscore.")
                log("Fortsetzen mit demselben Befehl und --resume. Es gibt keine automatische Zeitgrenze.")
                return 124 if isinstance(error, BudgetExceeded) else 130
            raise
    if result is None:
        log("Pilot abgeschlossen. Gleicher Befehl ohne --pilot-only, mit --resume startet die Suche.")
    else:
        log("ABGESCHLOSSEN: alle sieben Kanalsets, Nested-CV und aeussere Tests.")
        log(result[["method_id", "n_outer_tests", "complete_nested_cv", "mean_cohen_kappa", "mean_angle_mae_deg"]].to_string(index=False))
        log("Primaeres Verfahren: hpo_inner_selected; Kanalvergleiche sind deskriptiv.")
    return 0


class BudgetExceeded(RuntimeError):
    pass


class RuntimeBudget:
    """Compatibility runtime record. Target time NEVER interrupts any operation."""
    def __init__(self, output, args, clock=None):
        self.clock = clock or time.time
        self.path = output/"wall_budget.json"
        self.in_pilot = False
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if self.state.get("deadline_unix") is not None:
            self.state.setdefault("former_deadline_unix", self.state["deadline_unix"])
        start = self.state.get("started_unix", getattr(args, "run_started_wall", self.clock()))
        self.state.update({"started_unix": start, "target_hours": args.target_hours,
                           "started_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                           "time_policy": "planning_target_only", "automatic_time_limit": False,
                           "deadline_unix": None, "pilot_deadline_unix": None, "deadline_utc": None})
        atomic_json(self.path, self.state)

    @property
    def remaining_seconds(self):
        return math.inf

    def check(self, estimated_seconds=0., mandatory_seconds=0.):
        # Deliberately a no-op, also during import, pilot, caching and outer fits.
        return None


def sync_device(args):
    if torch is not None and args.device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))


def autocast_context(args):
    if torch is not None and args.device.startswith("cuda") and getattr(args, "resolved_precision", "fp32") == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def validate_nested_contexts(records, contexts, args):
    if args.outer_folds < 2 or args.inner_folds < 2 or not contexts:
        raise ValueError("Immer mindestens 2 Outer- und 2 Inner-Folds erforderlich.")
    if len({r.subject for r in records}) != len(records):
        raise ValueError("Personen muessen vor der CV zu genau einem Record zusammengefasst sein.")
    universe = set(range(len(records)))
    by_repeat = {}
    def check_classes(indices, role):
        if {records[int(i)].hard_reference for i in indices} != {0, 1}:
            raise ValueError(role + ": beide Klassen sind fuer definiertes Kappa erforderlich.")
    for repeat, seed, fold, train, test, inner in contexts:
        tr, te = set(map(int, train)), set(map(int, test))
        if tr & te or tr | te != universe:
            raise AssertionError("Outer-Train und aeusserer Test sind nicht disjunkt/vollstaendig.")
        check_classes(train, "Outer-Train")
        check_classes(test, "Aeussere Testdaten")
        if len(inner) != args.inner_folds:
            raise AssertionError("Unvollstaendige innere CV.")
        seen = Counter()
        for it, iv in inner:
            a, b = set(map(int, it)), set(map(int, iv))
            if a & b or a | b != tr or (a | b) & te:
                raise AssertionError("Inner-Fold beruehrt aeussere Testpersonen.")
            check_classes(it, "Inner-Train")
            check_classes(iv, "Innere Validierung")
            seen.update(b)
        if set(seen) != tr or set(seen.values()) != {1}:
            raise AssertionError("Jede Outer-Train-Person braucht genau eine innere OOF-Prognose.")
        by_repeat.setdefault(repeat, Counter()).update(te)
    if len(contexts) != len(args.split_seeds) * args.outer_folds:
        raise AssertionError("Outer-CV ist unvollstaendig.")
    for seen in by_repeat.values():
        if set(seen) != universe or set(seen.values()) != {1}:
            raise AssertionError("Jede Person muss pro Wiederholung genau einmal aeusserlich getestet werden.")


def workload(records, cap):
    # Laufzeitplanung darf keine aeusseren Zielwinkel zur Budgetwahl verwenden.
    # Strukturelle Obergrenze: Original plus hoechstens cap Ansichten je Person.
    return int(sum(len(r.pairs) for r in records) * (1 + cap))


def pilot_key(config):
    return config.channel_set+"__"+config.architecture


def cost_for_fit(config, train, evaluation, args, models, epochs):
    m = models[pilot_key(config)]
    ratio = max(1., workload(train, config.rare_angle_copy_cap)/max(1, m["train_contacts"]),
                sum(r.n_pairs for r in evaluation)/max(1, m["eval_contacts"]))
    # Deliberately includes a cold cache for every fit and checkpoint I/O.
    return args.runtime_safety_factor*ratio*(m["startup_seconds"] + max(0, epochs)*m["epoch_seconds"])


def outstanding_outer_seconds(records, args, contexts, configs, models, output):
    if not models:
        return 0.
    seconds = 0.
    for repeat, seed, fold, train, test, splits in contexts:
        for channel_set in CHANNEL_SETS:
            label = f"W{repeat}_F{fold}_"+channel_set
            if read_receipt(output, "outer", label, "outer") is not None:
                continue
            seconds += max(len(args.train_seeds)*cost_for_fit(c, [records[i] for i in train],
                [records[i] for i in test], args, models, args.max_epochs)
                for c in configs if c.channel_set == channel_set)
    return seconds


def check_candidate_budget(records, args, config, context, inner_splits, stage, epochs):
    # The operation plan is fixed before HPO; forecasts never veto a candidate.
    return None


def resumed_stale(saved_stage, current_stage, stale):
    if saved_stage == current_stage:
        return stale
    if (saved_stage, current_stage) in (("broad", "stable"), ("stable", "final")):
        return 0
    raise RuntimeError(f"Ungueltige Stufenfortsetzung: {saved_stage} -> {current_stage}")


def subset_configs(pool, count):
    result = []
    for condition in CONDITIONS:
        result.extend([c for c in pool if c.condition == condition][:count//2])
    if len(result) != 7*count:
        raise ValueError("Unvollstaendiger gemeinsamer Vorlagenpool.")
    return result


def prepare_arrays(train, evaluation, config, args, seed):
    """CPU only. Excluded SI channels are constant, including after augmentation."""
    if {r.subject for r in train} & {r.subject for r in evaluation}:
        raise AssertionError("Trainings-/Testpersonen ueberschneiden sich.")
    normalizer = FoldInputNormalizer(raw_indices=()).fit(train)
    copies = rare_angle_copy_counts(train, config.rare_angle_copy_cap, stable_seed(seed, "copies"))
    mask = np.asarray([i in CHANNEL_SETS[config.channel_set] for i in range(3)], np.float32)[None, :, None]
    chunks, owners = [], []
    for person, (record, n_extra) in enumerate(zip(train, copies)):
        original = normalizer.transform_pairs(record.pairs)
        for view in range(1+int(n_extra)):
            values = original.copy()
            for channel, source in enumerate(SELECTED_SOURCE_CHANNEL_INDICES):
                if mask[0, channel, 0]:
                    values[:, channel] += np.random.default_rng(stable_seed(
                        seed, record.subject, view, source)).normal(0, args.noise_sigma, values[:, channel].shape).astype(np.float32)
            values[:, mask[0, :, 0] == 0] = 0.
            chunks.append(values)
            owners.extend([person]*len(values))
    ev = [normalizer.transform_pairs(r.pairs) for r in evaluation]
    for values in ev:
        values[:, mask[0, :, 0] == 0] = 0.
    eval_owners = np.concatenate([np.full(len(v), i, np.int64) for i, v in enumerate(ev)])
    return np.concatenate(chunks), np.concatenate(ev), np.asarray(owners, np.int64), eval_owners, normalizer, copies


def stage_initializations(args, stage):
    return args.train_seeds if stage == "final" else args.train_seeds[:1]


def pilot_subset(records, limit):
    """Only the first inner split; reserve at least two people from each class."""
    if len(records) <= limit:
        return records
    indices = []
    for label in (0, 1):
        indices.extend([i for i, r in enumerate(records) if r.hard_reference == label][:2])
    for i in np.random.default_rng(913).permutation(len(records)):
        if len(indices) >= limit:
            break
        if int(i) not in indices:
            indices.append(int(i))
    return [records[i] for i in sorted(indices)]


def operation_counts(count, args):
    """Exact planned jobs and cumulative epoch upper bounds, without early stopping."""
    outer = len(args.split_seeds)*args.outer_folds
    sets, arms, inner, seeds = len(CHANNEL_SETS), 2, args.inner_folds, len(args.train_seeds)
    stable, final = sets*arms*2, sets*arms
    broad_fits = outer*sets*count*inner
    stable_fits = outer*stable*inner
    final_fits = outer*final*inner*seeds
    outer_fits = outer*sets*seeds
    epochs = {"broad": broad_fits*args.broad_epochs,
              "stable_additional": stable_fits*(args.stable_epochs-args.broad_epochs),
              "final_additional": outer*final*inner*((args.max_epochs-args.stable_epochs)
                                    +(seeds-1)*args.max_epochs),
              "outer_maximum": outer_fits*args.max_epochs}
    return {"configurations_per_channel_set": count, "outer_contexts": outer,
            "broad_candidates_per_outer": sets*count, "stable_candidates_per_outer": stable,
            "final_candidates_per_outer": final, "inner_stage_fit_calls": broad_fits+stable_fits+final_fits,
            "outer_model_fits": outer_fits, "outer_ensembles": outer*sets,
            "distinct_trainable_model_initializations": broad_fits+outer*final*inner*(seeds-1)+outer_fits,
            "outer_result_rows_including_selected_procedure": outer*(sets+1),
            "head_transformer_epochs_upper_bound_by_stage": epochs,
            "head_transformer_epochs_upper_bound_total": sum(epochs.values()),
            "note": "Fixed work plan. Early stopping may reduce epochs. Feature extraction and pilot are additional operations."}


def choose_operation_count(choices, target_hours, fixed_count=0):
    if not choices:
        raise ValueError("Keine Kandidaten fuer den Arbeitsplan.")
    if fixed_count:
        return next(c for c in choices if c["n_search_configs_per_set"] == fixed_count), "explicit_operation_count"
    within_target = [c for c in choices if c["estimated_hours"] <= target_hours]
    if within_target:
        return max(within_target, key=lambda c: c["n_search_configs_per_set"]), "estimated_operation_count"
    # A time estimate can exceed the target; it never vetoes a complete experiment.
    return min(choices, key=lambda c: c["n_search_configs_per_set"]), "minimum_operations_above_time_target"


def upgrade_before_hpo(output, saved, new_contract):
    """Reuse the failed V38.1.0 pilot only when the experiment itself has not begun."""
    old = saved.get("contract", {})
    if old.get("version") != "38.1.0" or old.get("script_sha256") != LEGACY_V3810_SOURCE_SHA256:
        return False
    ignored = {"version", "script_sha256", "time_policy"}
    if {k: v for k, v in old.items() if k not in ignored} != {
            k: v for k, v in clean_json(new_contract).items() if k not in ignored}:
        return False
    for section in ("inner_fit", "candidate", "outer_member", "outer", "selection", "channel_selection"):
        if any((output/"_state"/section).glob("*.json")):
            return False
    if any((output/"_checkpoints"/"inner_fit").glob("*.pt")):
        return False
    backup = output/"_migrations"/"v38_1_0_before_time_fix"
    atomic_json(backup/"run_config.json", saved)
    plan_path = output/"search_plan.json"
    if plan_path.exists():
        atomic_json(backup/"search_plan.json", json.loads(plan_path.read_text(encoding="utf-8")))
        plan_path.unlink()
    # Completed pilot receipts are reusable. Incomplete old-version checkpoints
    # must restart their own probe, because they retain the former code contract.
    pilot = output/"_runtime_pilot"
    for checkpoint in (pilot/"_checkpoints"/"inner_fit").glob("*.pt"):
        if read_receipt(pilot, "inner_fit", checkpoint.stem, "pilot") is None:
            destination = backup/"partial_pilot_checkpoints"/checkpoint.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(checkpoint, destination)
    atomic_json(output/"run_config.json", {**saved, "contract": clean_json(new_contract),
        "time_fix_migration": {"from_version": "38.1.0", "to_version": VERSION,
                               "completed_pilot_measurements_reused": True, "hpo_had_not_started": True}})
    log("Zeitsteuerung korrigiert: abgeschlossene Pilotmessungen werden uebernommen. Keine automatische Zeitgrenze.")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
