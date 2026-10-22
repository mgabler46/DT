#!/usr/bin/env python3
"""TimeGAN-SI7 V1.6: wiederholte patientenweise Nested-CV auf ALLEN Personen.

Start in der bestehenden CUDA-venv (keine sequences.npz, kein alter Ergebnisordner):
  python timegan_si7_nested_gtx1070_v1_6.py ./CSV_Export --device cuda --output-dir ./results_timegan_nested_v16
Fortsetzen nach Abbruch (identische Einstellungen, neues Zeitbudget):
  python timegan_si7_nested_gtx1070_v1_6.py ./CSV_Export --device cuda --output-dir ./results_timegan_nested_v16 --resume
Technischer Test ohne Gewichte oder Patientendaten:
  python timegan_si7_nested_gtx1070_v1_6.py --self-test --device cpu

Standard: 5 Wiederholungen x 3 aeussere x 3 innere Patientenfolds, 3 Seeds.
Alle Messungen einer Person bleiben zusammen. Kein separater Holdout wird
reserviert: jede Person wird pro Wiederholung genau einmal aussen getestet.
Innerhalb jedes aeusseren Trainingsfolds werden Modellvariante, GAN-Kopienzahl,
gemeinsamer Klassifikator-Checkpoint und Schwelle nach innerem OOF-Kappa
gewaehlt. Die 3 Seed-Wahrscheinlichkeiten werden fuer die primaere Auswertung
gemittelt; einzelne Seeds werden zusaetzlich ausgewertet, nicht selektiert.

Verfahren: real; random (Oversampling); noise_selected (Kontrolle);
timegan_selected. Letztere beiden duerfen keine Augmentation waehlen.
Standard-GAN: hidden=16, 2 GRU-Lagen, ratio=0.5, sigma=0.02,
Rauschkopien zur Auswahl: 0,3,4,5. Direkte Rauschkopien werden fuer 3,4,5
geprueft. Alle Klassifikatoren: dieselben Checkpoints 50,100,...,300 Updates.
Die max. Trainingsdauer ist gleich; der innen gewaehlte Checkpoint kann variieren.
Eine ratio von 0.5 bedeutet NICHT 50 Prozent synthetische Trainingsziehungen.
Mischanteil = ratio*n_Minderheit/(n_Patienten+ratio*n_Minderheit).

Originaler SI15->SI7-CSV-Import, Kontakt-QC und Normierung aus V36.1 bleiben
erhalten. ConvNeXt-Tiny (ImageNet) ist eingefroren; nur der kleine Transformer
und Klassifikationskopf werden trainiert. TimeGAN ist die bisherige kompakte
PyTorch-Portierung mit drei Trainingsphasen, keine unveraenderte Autoren-Datei.
Es werden SI-Kontaktpaare [101,7] erzeugt, keine vollstaendigen Patienten.
Nur Klassifikation: synthetische Paare erhalten KEINE erfundenen Zielwinkel.

Patientenweise Normierung, Imputation, GAN und Augmentation werden in JEDEM
inneren Fold neu ausschliesslich auf Trainingspersonen angepasst. Der aeussere
Test wird erst nach Speicherung der Auswahl verwendet. Gleiche Splits und
Klassifikator-Seeds fuer alle Verfahren; Generator-Seeds variieren je Seed-Slot.

Ausgabe: summary_patient_pooled.csv (Mittel der patientengepoolten Kennzahlen
pro Wiederholung, niemals 5*70 als unabhaengige Personen), outer_fold_metrics.csv,
outer_seed_metrics.csv, paired_comparison_summary.csv, selection_frequency.csv,
innere Auswahl, GAN-Lernkurven und Sequenzdiagnosen. Vollstaendige aeussere Folds
werden atomar gespeichert und bei --resume wiederverwendet. Ein unvollstaendiger
Fold wird neu gerechnet. Die Zusammenfassung trennt vollstaendige Wiederholungen.
Interne NPZ-Dateien fuer Wiederaufnahme enthalten nur lokale Personenindizes,
Labels und Vorhersagen; CSV-Berichte enthalten keine Patientenkennungen.

Die optionalen gepaarten Patienten-Bootstrapintervalle halten alle
Wiederholungen einer Person zusammen. Sie sind bedingt auf bereits angepasste
Modelle und ersetzen KEINE Unsicherheitsanalyse mit erneutem Training.
Fold-/Seed-Streuungen sind KEINE unabhaengigen Stichproben oder Konfidenzintervalle.
Da fruehere Ergebnisse die Suchraeume beeinflusst haben, bleibt dies eine
nachtraegliche Robustheitsanalyse. Die Nested-CV verhindert Leakage im neuen
Lauf; sie macht fruehere methodische Entscheidungen nicht rueckgaengig.

Abhaengigkeiten: bestehendes CUDA-PyTorch plus passende torchvision,
numpy, pandas, scipy, scikit-learn. GTX 1070: float32, keine AMP/TF32.
Offizielle ConvNeXt-Gewichte werden im bisherigen --weights-dir wiederverwendet.
Referenzen:
  https://github.com/jsyoon0823/TimeGAN
  https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import warnings
from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise SystemExit("CUDA-PyTorch fehlt in diesem Python. Vorhandene Projekt-venv verwenden.") from exc

IMPORT_SOURCE_SCRIPT_SHA256 = 'a448f436eb843ded9ee65ac9dc1c3be30c6038af4b91907f1919bbc1fa51a220'

# Originaler Import und Modellbausteine (V36.1).

SOURCE_CHANNELS = 15


EXPECTED_CHANNELS = 7


POINTS_PER_PAIR = 101


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


SELECTED_SOURCE_CHANNEL_INDICES = (2, 3, 8, 11, 12, 13, 14)


SELECTED_INPUT_CHANNELS = tuple(
    SOURCE_SI15_CHANNELS[index] for index in SELECTED_SOURCE_CHANNEL_INDICES
)


RAW_COP_INDICES = (3, 4, 5, 6)


@dataclass
class Measurement:
    measurement_id: str
    source_file: str
    subject: str
    pairs: np.ndarray  # n_contact_pairs x 7 x 101 nach SI15-Kanalauswahl


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


class FoldInputNormalizer:
    """Trainingsfold-sichere Skalierung ohne Z-Normierung der SI-Kanaele.

    Die vier Rohkanaele an Position 3--6 werden ausschliesslich mit einem aus
    dem jeweiligen Trainingsfold bestimmten Absolutmaximum skaliert. Die drei
    bereits als SI vorliegenden Kanaele 0--2 bleiben in ihren Originaleinheiten.
    Fehlende Werte werden mit trainingsfold-internen Medianen ersetzt.
    """

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
) -> tuple[list[SubjectRecord], dict[str, Any], pd.DataFrame, str]:
    """Rekonstruiert SI15 wie V29.10 und gibt die feste SI7-Teilmenge aus."""
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
        f"📂 {len(files)} CSV-Dateien gefunden. Eigenstaendiger SI15→SI7-Import mit "
        f"Kontaktdauer-QC (±{contact_duration_tolerance * 100:g}% vom "
        f"Messungsmittel), Nullrand-Toleranz={contact_zero_tolerance:g} ..."
    )
    raw_measurements: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in files:
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
                raise RuntimeError(f"Interne SI7-Paarform ist {pair.shape}.")
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


WEIGHTS_NAME = "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"


def encoder_parameter_formula(width, ff):
    # Q/K/V und Ausgabeprojektion, zwei FF-Linears, zwei LayerNorms; mit Bias.
    return 4 * width ** 2 + 2 * width * ff + 9 * width + ff


@dataclass(frozen=True)
class TransferConfig:
    architecture: str
    task: str
    transfer_mode: str
    image_width: int
    learning_rate: float
    backbone_lr_fraction: float
    weight_decay: float
    head_dropout: float
    feature_dropout: float
    head_hidden: int
    class_weight: str
    rare_angle_copy_cap: int
    warmup_epochs: int
    huber_beta: float
    transformer_width: int = 16
    attention_heads: int = 2
    transformer_dropout: float = .1
    feedforward_multiplier: int = 2
    tokens: int = 4
    stochastic_depth: float = .1

    @property
    def condition(self):
        return self.architecture + "_" + self.task + "__" + self.transfer_mode

    @property
    def config_id(self):
        return "tiny_" + digest(asdict(self))[:12]

    def row(self):
        d = self.transformer_width
        return {"config_id": self.config_id, "condition_id": self.condition,
                "pretrained_source": WEIGHTS_NAME, "image_height": 96,
                "transformer_layers": int(self.architecture == "cnn_transformer"),
                "input_channels": 7, "input_points": 101, "covered_input_points": 101,
                "encoder_parameters_formula": (encoder_parameter_formula(d, d * self.feedforward_multiplier)
                                                if self.architecture == "cnn_transformer" else 0),
                **asdict(self)}


def signal_images(pairs, width):
    if pairs.ndim != 3 or pairs.shape[1:] != (7, 101):
        raise ValueError(f"Erwartet [Kontakte,7,101], erhalten {tuple(pairs.shape)}")
    # Monotone, invertierbare Abbildung, keine abgeschnittenen Amplituden.
    values = .5 + torch.atan(pairs) / math.pi
    values = F.interpolate(values, size=width, mode="linear", align_corners=True)
    # Kein Mischen der Kanaele beim vertikalen Resize: feste Baender.
    band_index = (torch.arange(96, device=pairs.device) * 7 // 96).long()
    image = values[:, band_index].unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    rgb_mean = image.new_tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
    rgb_std = image.new_tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
    return (image - rgb_mean) / rgb_std


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
                    "torchvision konnte nicht importiert werden.\n"
                    f"Python: {sys.executable}\n"
                    f"PyTorch: {torch.__version__}; CUDA-Build: {torch.version.cuda}\n"
                    f"Urspruenglicher Importfehler: {type(error).__name__}: {error}\n"
                    "Die CUDA-Versionsnummer allein bestimmt nicht die passende torchvision-Version.\n"
                    "Diagnose im selben Python: python -m pip show torch torchvision\n"
                    "Funktionierenden CUDA-Torch-Build bei der Reparatur beibehalten."
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


def create_test_csvs(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(18):
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


# Kompakter TimeGAN-Kern und Auswertung.

class BudgetExpired(RuntimeError):
    pass


class Clock:
    def __init__(self, hours):
        self.start = time.monotonic()
        self.end = self.start + hours * 3600

    def check(self):
        if time.monotonic() >= self.end:
            raise BudgetExpired("Zeitbudget erreicht; unvollstaendige Ergebnisse sind als solche markiert.")

    def minutes(self):
        return (time.monotonic() - self.start) / 60


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Scaler:
    def fit(self, x):
        self.low = x.min(axis=(0, 1))
        self.span = x.max(axis=(0, 1)) - self.low
        self.denom = np.where(self.span > 1e-7, self.span, 1.0)
        return self

    def transform(self, x):
        return ((x - self.low) / self.denom).astype(np.float32)

    def inverse(self, x):
        return (x * self.span + self.low).astype(np.float32)


def row_probabilities(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


class SeqNet(nn.Module):
    def __init__(self, inputs, hidden, outputs, layers, sigmoid=True):
        super().__init__()
        self.rnn = nn.GRU(inputs, hidden, layers, batch_first=True)
        self.proj = nn.Linear(hidden, outputs)
        self.sigmoid = sigmoid

    def forward(self, x):
        z = self.proj(self.rnn(x)[0])
        return z.sigmoid() if self.sigmoid else z


class TimeGAN(nn.Module):
    def __init__(self, channels, hidden, layers):
        super().__init__()
        self.channels, self.hidden, self.layers = channels, hidden, layers
        self.E = SeqNet(channels, hidden, hidden, layers)
        self.R = SeqNet(hidden, hidden, channels, layers)
        self.G = SeqNet(channels, hidden, hidden, layers)
        self.S = SeqNet(hidden, hidden, hidden, max(1, layers - 1))
        self.D = SeqNet(hidden, hidden, 1, layers, sigmoid=False)


def active(model, names):
    for name, block in model.named_children():
        for p in block.parameters():
            p.requires_grad_(name in names)


def update(optimizer, loss, params):
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Nichtendlicher Loss; Lauf abgebrochen statt Ergebnisse zu verfaelschen.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(list(params), 5., error_if_nonfinite=True)
    optimizer.step()


def fit_timegan(x, groups, hidden, args, device, clock, seed, tag,
                history_rows=None, context=None, constant=None):
    """Alle drei Phasen trainieren; QC ausschliesslich mit Trainingssequenzen."""
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model = TimeGAN(x.shape[2], hidden, args.layers).to(device).train()
    er = list(model.E.parameters()) + list(model.R.parameters())
    gs = list(model.G.parameters()) + list(model.S.parameters())
    oe, os = torch.optim.Adam(er, lr=args.gan_lr), torch.optim.Adam(model.S.parameters(), lr=args.gan_lr)
    og, od = torch.optim.Adam(gs, lr=args.gan_lr), torch.optim.Adam(model.D.parameters(), lr=args.gan_lr)
    probs = row_probabilities(groups)
    bce = nn.BCEWithLogitsLoss()
    history_rows = history_rows if history_rows is not None else []
    context = context or {}
    constant = np.zeros(x.shape[2], bool) if constant is None else constant
    reference = x[rng.choice(len(x), min(128, len(x)), replace=False, p=probs)]
    global_step = 0

    def batch():
        ix = rng.choice(len(x), args.batch_size, replace=True, p=probs)
        return torch.from_numpy(x[ix]).to(device)

    def noise(b):
        return torch.rand_like(b)

    def supervise(h):
        return F.mse_loss(model.S(h)[:, :-1], h[:, 1:])

    def snapshot(phase, step, phase_steps, losses):
        # Separater torch.Generator: Monitoring veraendert nicht den Trainings-RNG.
        fake = generate(model, min(64, max(32, len(reference))), x.shape[1],
                        args, device, constant, seed + 900000)  # Fixes Rauschsample je Checkpoint
        with torch.no_grad():
            real = torch.from_numpy(reference).to(device)
            rec = F.mse_loss(model.R(model.E(real)), real).item()
            sup = supervise(model.E(real)).item()
        div = diversity_report(fake, reference)
        history_rows.append(dict(**context, hidden=hidden, gan_seed=seed,
                                 phase=phase, phase_step=step, phase_total=phase_steps,
                                 global_step=global_step, runtime_minutes=clock.minutes(),
                                 reconstruction_mse=float(rec), supervisor_mse=float(sup),
                                 generator_loss=losses.get('generator_loss'),
                                 adversarial_loss=losses.get('adversarial_loss'),
                                 moment_loss=losses.get('moment_loss'),
                                 discriminator_loss=losses.get('discriminator_loss'),
                                 **div))

    phases = [("AE", args.ae_steps), ("SUP", args.supervisor_steps), ("GAN", args.gan_steps)]
    for phase, steps in phases:
        phase_start = time.monotonic()
        for step in range(1, steps + 1):
            clock.check()
            global_step += 1
            real = batch()
            losses = {}
            if phase == "AE":
                active(model, {"E", "R"})
                loss = 10 * torch.sqrt(F.mse_loss(model.R(model.E(real)), real) + 1e-8)
                update(oe, loss, er)
            elif phase == "SUP":
                active(model, {"S"})
                with torch.no_grad():
                    h = model.E(real)
                loss = supervise(h)
                update(os, loss, model.S.parameters())
            else:
                # R/D bleiben im train-Modus fuer den cuDNN-GRU-Gradienten.
                active(model, {"G", "S"})
                with torch.no_grad():
                    h = model.E(real)
                eh = model.G(noise(real))
                hh = model.S(eh)
                fake = model.R(hh)
                adv = bce(model.D(hh), torch.ones_like(hh[:, :, :1]))
                adv = adv + bce(model.D(eh), torch.ones_like(eh[:, :, :1]))
                moment = (fake.mean(0) - real.mean(0)).abs().mean()
                moment = moment + (torch.sqrt(fake.var(0, unbiased=False) + 1e-6) -
                                   torch.sqrt(real.var(0, unbiased=False) + 1e-6)).abs().mean()
                loss = adv + 100 * torch.sqrt(supervise(h) + 1e-8) + 100 * moment
                losses.update(generator_loss=float(loss.item()),
                              adversarial_loss=float(adv.item()), moment_loss=float(moment.item()))
                update(og, loss, gs)
                active(model, {"E", "R"})
                h = model.E(real)
                eloss = 10 * torch.sqrt(F.mse_loss(model.R(h), real) + 1e-8) + .1 * supervise(h)
                update(oe, eloss, er)
                active(model, {"D"})
                with torch.no_grad():
                    h = model.E(real)
                    eh = model.G(noise(real))
                    hh = model.S(eh)
                dloss = bce(model.D(h), torch.ones_like(h[:, :, :1]))
                dloss = dloss + bce(model.D(hh), torch.zeros_like(hh[:, :, :1])) + bce(model.D(eh), torch.zeros_like(eh[:, :, :1]))
                losses['discriminator_loss'] = float(dloss.item())
                if dloss.item() > .15:
                    update(od, dloss, model.D.parameters())
            if step == 1 or step % args.diagnostic_every == 0 or step == steps:
                snapshot(phase, step, steps, losses)
                # generate() wechselt in eval(); cuDNN-GRU braucht fuer den naechsten
                # Backward zwingend train()-Modus, auch bei eingefrorenen Blocks.
                model.train()
            if step == 1 or step % args.log_every == 0 or step == steps:
                elapsed = time.monotonic() - phase_start
                eta = elapsed / step * (steps - step) / 60
                print(f"{tag} {phase} {step}/{steps} Loss={loss.item():.4f} | Phase-ETA {eta:.1f} min | Gesamt {clock.minutes():.1f} min", flush=True)
    active(model, set())
    return model.eval()


@torch.no_grad()
def generate(model, n, length, args, device, constant, seed):
    if n <= 0:
        raise ValueError('Mindestens eine synthetische Sequenz erforderlich.')
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    result = []
    for start in range(0, n, args.batch_size):
        z = torch.rand(min(args.batch_size, n - start), length, model.channels,
                       device=device, generator=generator)
        result.append(model.R(model.S(model.G(z))).cpu().numpy())
    out = np.concatenate(result)
    out[..., constant] = 0
    if not np.isfinite(out).all():
        raise FloatingPointError("Generator erzeugte NaN/Inf.")
    return out.astype(np.float32)


def metrics(y, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    return dict(accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                cohen_kappa=float(cohen_kappa_score(y, pred)), roc_auc=float(roc_auc_score(y, probabilities)),
                sensitivity=float(pred[y == 1].mean()), specificity=float((pred[y == 0] == 0).mean()))


def select_threshold(y, p):
    # Ausschliesslich auf echten inneren Validierungspatienten.
    grid = np.linspace(.1, .9, 17)
    scored = [(metrics(y, p, float(t)), float(t)) for t in grid]
    return max(scored, key=lambda v: (v[0]["cohen_kappa"], v[0]["balanced_accuracy"], -abs(v[1] - .5)))


def pairwise_rmse(a, b):
    """Distanzen auf [N, Zeit, Kanal] in fold-internen TimeGAN-Einheiten."""
    a = np.asarray(a, np.float64).reshape(len(a), -1)
    b = np.asarray(b, np.float64).reshape(len(b), -1)
    if a.shape[1] != b.shape[1]:
        raise ValueError('Inkompatible Sequenzlaengen/Kanaele.')
    a2 = np.mean(a * a, axis=1)[:, None]
    b2 = np.mean(b * b, axis=1)[None, :]
    return np.sqrt(np.maximum(0., a2 + b2 - 2 * (a @ b.T) / a.shape[1]))


def diversity_report(fake, real):
    f = np.asarray(fake[:128], np.float64)
    r = np.asarray(real[:512], np.float64)
    distances = pairwise_rmse(f, r)
    fst = float(f.std(axis=0).mean())
    rst = float(r.std(axis=0).mean())
    return dict(nearest_train_rmse_mean=float(distances.min(1).mean()),
                generated_std=fst, real_std=rst,
                std_ratio=fst / rst if rst > 1e-12 else None)


def _safe_corr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    am, bm = a - a.mean(), b - b.mean()
    denom = np.sqrt(np.sum(am * am) * np.sum(bm * bm))
    return float(np.dot(am, bm) / denom) if denom > 1e-12 else None


def quality_reports(fake, real, *, max_samples=128, real_groups=None):
    """Kanalmarginalien, Dynamik, LOO-Referenzdistanzen (Trainingsdaten)."""
    from scipy.stats import ks_2samp

    f = np.asarray(fake[:max_samples], np.float64)
    r = np.asarray(real[:max_samples], np.float64)
    if f.ndim != 3 or r.ndim != 3 or f.shape[1:] != r.shape[1:]:
        raise ValueError('QC erwartet zwei Arrays [N, Zeit, Kanaele].')
    if min(len(f), len(r)) < 2:
        raise ValueError('QC benoetigt mindestens zwei echte und synthetische Sequenzen.')
    channels = []
    for c in range(f.shape[-1]):
        a, b = f[:, :, c], r[:, :, c]
        row = dict(channel=c, channel_name=SELECTED_INPUT_CHANNELS[c],
                   fake_mean=float(a.mean()), real_mean=float(b.mean()),
                   fake_std=float(a.std()), real_std=float(b.std()),
                   fake_q05=float(np.quantile(a, .05)), real_q05=float(np.quantile(b, .05)),
                   fake_median=float(np.median(a)), real_median=float(np.median(b)),
                   fake_q95=float(np.quantile(a, .95)), real_q95=float(np.quantile(b, .95)),
                   ks_statistic=float(ks_2samp(a.ravel(), b.ravel()).statistic),
                   fake_mean_abs_delta=float(np.abs(np.diff(a, axis=1)).mean()),
                   real_mean_abs_delta=float(np.abs(np.diff(b, axis=1)).mean()))
        for lag in (1, 5, 10):
            row[f'fake_autocorr_lag{lag}'] = _safe_corr(a[:, :-lag], a[:, lag:])
            row[f'real_autocorr_lag{lag}'] = _safe_corr(b[:, :-lag], b[:, lag:])
        channels.append(row)
    fr = pairwise_rmse(f, r)
    rr = pairwise_rmse(r, r)
    ff = pairwise_rmse(f, f)
    np.fill_diagonal(rr, np.inf)  # Keine Selbstnachbarn als real-to-real-Referenz.
    # Zusaetzliche Referenz nur zwischen verschiedenen Patienten, um nicht
    # allein die Aehnlichkeit benachbarter Kontakte derselben Person zu messen.
    cross = None
    if real_groups is not None:
        real_groups = np.asarray(real_groups[:len(r)])
        if len(real_groups) != len(r):
            raise ValueError('Patientengruppen und QC-Real-Array stimmen nicht ueberein.')
        cross = np.where(real_groups[:, None] != real_groups[None, :], rr, np.inf)
    np.fill_diagonal(ff, np.inf)
    nearest_fr, nearest_rr, nearest_ff = fr.min(1), rr.min(1), ff.min(1)
    dist = dict(fake_to_real_nearest_rmse_mean=float(nearest_fr.mean()),
                fake_to_real_nearest_rmse_median=float(np.median(nearest_fr)),
                fake_to_real_nearest_rmse_q05=float(np.quantile(nearest_fr, .05)),
                real_to_real_loo_nearest_rmse_mean=float(nearest_rr.mean()),
                real_to_real_loo_nearest_rmse_median=float(np.median(nearest_rr)),
                real_to_real_loo_nearest_rmse_q05=float(np.quantile(nearest_rr, .05)),
                fake_to_fake_loo_nearest_rmse_mean=float(nearest_ff.mean()),
                fake_to_fake_loo_nearest_rmse_median=float(np.median(nearest_ff)),
                fraction_fake_below_real_real_q05=float(np.mean(nearest_fr < np.quantile(nearest_rr, .05))),
                n_fake=int(len(f)), n_real=int(len(r)))
    if cross is not None:
        cross_nearest = cross.min(axis=1)
        if np.isfinite(cross_nearest).all():
            dist['real_to_real_cross_patient_nearest_rmse_mean'] = float(cross_nearest.mean())
            dist['real_to_real_cross_patient_nearest_rmse_q05'] = float(np.quantile(cross_nearest, .05))
        else:
            dist['real_to_real_cross_patient_nearest_rmse_mean'] = None
            dist['real_to_real_cross_patient_nearest_rmse_q05'] = None
    return channels, dist


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class SI7Scaler(Scaler):
    """Originaler FoldNormalizer plus invertierbare TimeGAN-Skalierung."""
    def fit(self, x, groups):
        records = [SimpleNamespace(measurements=[SimpleNamespace(
            pairs=x[groups == owner].transpose(0, 2, 1))]) for owner in np.unique(groups)]
        self.normalizer = FoldInputNormalizer().fit(records)
        normalized = self.normalizer.transform_pairs(x.transpose(0, 2, 1)).transpose(0, 2, 1)
        return super().fit(normalized)

    def transform(self, x):
        values = self.normalizer.transform_pairs(x.transpose(0, 2, 1)).transpose(0, 2, 1)
        return super().transform(values)

    def inverse(self, x):
        values = super().inverse(x)
        # Rueckgabe in denselben SI7-Einheiten wie direkt nach CSV-Import.
        return (values * self.normalizer.raw_denominators_[None, None, :]).astype(np.float32)


def load_si7_csv(args):
    stream = io.StringIO()
    try:
        with redirect_stdout(stream):
            records, dataset, audit, _ = load_standalone_si15_records(
                args.data_dir, force_threshold=args.force_threshold,
                contact_duration_tolerance=args.contact_duration_tolerance,
                contact_zero_tolerance=args.contact_zero_tolerance,
                min_si_pairs_per_subject=args.min_si_pairs_per_subject)
    except Exception as exc:
        if args.show_import_log:
            print(stream.getvalue(), end='', flush=True)
            print('Originalfehler:', str(exc), flush=True)
        raise ValueError("CSV-Import fehlgeschlagen. Erwartet werden die ORIGINALEN CSV-Exporte mit force_left/force_right, "
                         "COP-Spalten und dem bisherigen Dateinamensschema. Details lokal mit --show-import-log anzeigen.") from exc
    if args.show_import_log:
        print(stream.getvalue(), end='', flush=True)
    x = np.concatenate([record.pairs.transpose(0, 2, 1) for record in records]).astype(np.float32)
    y = np.concatenate([np.full(record.n_pairs, record.hard_reference, np.int64) for record in records])
    groups = np.concatenate([np.full(record.n_pairs, i, np.int64) for i, record in enumerate(records)])
    gy = np.asarray([record.hard_reference for record in records], np.int64)
    names = np.asarray([record.subject for record in records])
    if x.shape[1:] != (101, 7):
        raise ValueError("Import muss die originale Form [SI-Paare,101,7] liefern.")
    if np.bincount(gy, minlength=2).min() < max(4, args.folds):
        raise ValueError("Nach QC zu wenige Patienten einer Klasse fuer den gewaehlten Vergleich.")
    # Es werden nur aggregierte Auditwerte exportiert, nicht dataset-QC-Tabellen.
    raw = audit.to_dict(orient='records')
    print(f"CSV-Import: {len(records)} Patienten, Klassen={np.bincount(gy).tolist()}, "
          f"{sum(len(r.measurements) for r in records)} Messungen, {len(x)} SI-Paare [7,101].", flush=True)
    print(f"Kontakt-QC +/-{args.contact_duration_tolerance * 100:g}%; Minimum {args.min_si_pairs_per_subject} Paare/Patient; "
          f"Importfehler={len(dataset['import_failures'])}.", flush=True)
    return x, y, groups, gy, names, raw


def probe_config():
    return TransferConfig(
        architecture='cnn_transformer', task='classification', transfer_mode='frozen',
        image_width=160, learning_rate=.001, backbone_lr_fraction=0., weight_decay=.01,
        head_dropout=.1, feature_dropout=0., head_hidden=0, class_weight='none',
        rare_angle_copy_cap=0, warmup_epochs=0, huber_beta=1., transformer_width=16,
        attention_heads=2, transformer_dropout=.1, feedforward_multiplier=2,
        tokens=4, stochastic_depth=0.)


class SI7Probe(nn.Module):
    def __init__(self, factory, scaler):
        super().__init__()
        self.base = TransferModel(factory, probe_config())
        self.register_buffer('gan_low', torch.as_tensor(scaler.low).reshape(1, 1, 7))
        # denom statt span: auch im Training konstante Kanaele duerfen bei
        # echten Testpatienten andere Werte annehmen.
        self.register_buffer('gan_denom', torch.as_tensor(scaler.denom).reshape(1, 1, 7))

    def raw_maps(self, x):
        original_fold_units = x * self.gan_denom + self.gan_low
        return self.base.maps(original_fold_units.transpose(1, 2))

    def from_maps(self, x):
        return self.base.from_maps(x)

    def forward(self, x):
        return self.from_maps(self.raw_maps(x))


@torch.no_grad()
def frozen_maps(model, x, args):
    # Cache nur fuer den unveraenderten vortrainierten CNN-Teil. Kein Training
    # auf Validierungs-/Testdaten; Skalierung stammt aus dem Trainingsfold.
    h = hashlib.sha256(np.ascontiguousarray(x).tobytes())
    h.update(model.gan_low.cpu().numpy().tobytes())
    h.update(model.gan_denom.cpu().numpy().tobytes())
    key = h.hexdigest()
    if key in args._map_cache:
        args._map_cache.move_to_end(key)
        return args._map_cache[key]
    model.eval()
    pieces, start = [], 0
    print(f"CNN-Maps vorberechnen: {len(x)} SI-Paare; Microbatch={args.contact_microbatch}", flush=True)
    while start < len(x):
        args._clock.check()
        stop = min(len(x), start + args.contact_microbatch)
        try:
            chunk = model.raw_maps(torch.from_numpy(x[start:stop]).to(args._device)).cpu()
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if args.contact_microbatch <= 1:
                raise
            args.contact_microbatch = max(1, args.contact_microbatch // 2)
            print('VRAM-Anpassung: Kontakt-Microbatch=', args.contact_microbatch, flush=True)
            continue
        pieces.append(chunk)
        start = stop
    result = torch.cat(pieces)
    args._map_cache[key] = result
    while len(args._map_cache) > 6:
        args._map_cache.popitem(last=False)
    return result


@torch.no_grad()
def predict_patients(model, x, y, groups, args, device):
    model.eval()
    maps = frozen_maps(model, x, args)
    chunks = []
    for start in range(0, len(x), args.batch_size):
        chunks.append(model.from_maps(maps[start:start + args.batch_size].to(device)).sigmoid().cpu().numpy())
    p = np.concatenate(chunks)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    pooled = np.bincount(inverse, weights=p) / counts
    labels = np.array([y[groups == v][0] for v in unique])
    return labels, pooled


def fit_classifier(x, y, groups, minority, mode, ratio, fake, args, device, clock, seed, tag,
                   valid=None, steps=None, collect_history=False):
    seed_all(seed)
    model = SI7Probe(args._factory, args._current_scaler).to(device)
    train_maps = frozen_maps(model, x, args)
    fake_maps = frozen_maps(model, fake, args) if mode in ('timegan_noise', 'noise') and ratio > 0 else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.classifier_lr, weight_decay=.01)
    seed_all(seed + 73)
    rng = np.random.default_rng(seed)
    probs = row_probabilities(groups)
    minority_indices = np.flatnonzero(y == minority)
    mprob = row_probabilities(groups[minority_indices])
    n_groups, n_min = len(np.unique(groups)), len(np.unique(groups[minority_indices]))
    if mode == 'noise':
        if fake is None or len(fake) != len(x) * int(ratio):
            raise ValueError('Rauschkopien muessen pro echter Sequenz genau copies-mal vorliegen.')
        # K reale Kopien je Original, aber ohne Pseudopatienten: Auswahl bleibt
        # patientengewichtet wie bei den echten Trainingssequenzen.
        mix = ratio / (1. + ratio)
    else:
        mix = ratio * n_min / (n_groups + ratio * n_min) if mode != 'real' else 0.
    n_steps = args.classifier_steps if steps is None else steps
    best, best_key, history = None, None, []
    for step in range(1, n_steps + 1):
        clock.check()
        model.train()
        ix = rng.choice(len(x), args.batch_size, replace=True, p=probs)
        xb, yb = train_maps[ix].clone(), y[ix].copy()
        replace_mask = rng.random(args.batch_size) < mix
        count = int(replace_mask.sum())
        if count:
            if mode in ('timegan_noise', 'noise'):
                if mode == 'noise':
                    # Bilde dieselbe Patientengewichtung ab: zuerst echtes Paar,
                    # dann eine der K vorher fest generierten Rauschkopien.
                    originals = rng.choice(len(x), count, replace=True, p=probs)
                    copy_numbers = rng.integers(0, int(ratio), count)
                    chosen = copy_numbers * len(x) + originals
                    yb[replace_mask] = y[originals]
                else:
                    chosen = rng.integers(0, len(fake_maps), count)
                    yb[replace_mask] = minority
                xb[replace_mask] = fake_maps[chosen]
            elif mode == 'random':
                chosen = rng.choice(minority_indices, count, replace=True, p=mprob)
                xb[replace_mask] = train_maps[chosen]
            if mode == 'random':
                yb[replace_mask] = minority
        logits = model.from_maps(xb.to(device)).reshape(-1)
        target = torch.from_numpy(yb.astype(np.float32)).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        update(opt, loss, params)
        if valid is not None and (step % args.eval_every == 0 or step == n_steps):
            vy, vp = predict_patients(model, *valid, args, device)
            if collect_history:
                history.append(dict(step=step, y=vy, p=vp))
                monitor = metrics(vy, vp, .5)
                print(f'{tag} {step}/{n_steps} Val@0.50 BA={monitor["balanced_accuracy"]:.3f} '
                      f'Accuracy={monitor["accuracy"]:.3f} Kappa={monitor["cohen_kappa"]:.3f} '
                      f'| {clock.minutes():.1f} min', flush=True)
            else:
                score, threshold = select_threshold(vy, vp)
                key = (score['cohen_kappa'], score['balanced_accuracy'])
                if best_key is None or key > best_key:
                    best_key = key
                    best = dict(metrics=score, threshold=threshold, steps=step, state=model.base.learned_state())
                print(f"{tag} {step}/{n_steps} Val BA={score['balanced_accuracy']:.3f} "
                      f"Accuracy={score['accuracy']:.3f} Kappa={score['cohen_kappa']:.3f} | {clock.minutes():.1f} min", flush=True)
        elif valid is None and (step == 1 or step % args.log_every == 0 or step == n_steps):
            print(f'{tag} {step}/{n_steps} Loss={loss.item():.4f} | {clock.minutes():.1f} min', flush=True)
    if valid is not None and not collect_history:
        model.base.restore_learned(best.pop('state'))
    return model.eval(), (history if collect_history else best)


def select_inner_candidate(histories, n_inner, n_seeds, threshold_grid=None):
    """Waehlt gemeinsamen Checkpoint und Schwelle aus allen inneren OOF-Folds/Seeds.

    histories[(inner_fold, seed_index)] = Liste von {step, y, p}.
    Jeder Validierungspatient ist in genau einem inneren Fold. Die
    Seed-Prognosen werden zuerst je Fold gemittelt; der Checkpoint/Threshold
    wird anschliessend nach patienten-gepooltem OOF-Kappa gewaehlt.
    """
    thresholds = np.linspace(.1, .9, 17) if threshold_grid is None else threshold_grid
    required = {(inner, seed) for inner in range(n_inner) for seed in range(n_seeds)}
    if set(histories) != required:
        raise ValueError('Unvollstaendige innere Fold/Seed-Kombination.')
    checkpoints = set(item['step'] for item in histories[(0, 0)])
    for values in histories.values():
        if {item['step'] for item in values} != checkpoints:
            raise ValueError('Unterschiedliche Checkpoints in inneren Folds/Seeds.')
    best, all_scores = None, []
    for step in sorted(checkpoints):
        by_fold = []
        seed_truth = [[] for _ in range(n_seeds)]
        seed_prob = [[] for _ in range(n_seeds)]
        for inner in range(n_inner):
            elements = [next(v for v in histories[(inner, k)] if v['step'] == step)
                        for k in range(n_seeds)]
            if any(not np.array_equal(elements[0]['y'], el['y']) for el in elements[1:]):
                raise ValueError('Seed-Prognosen besitzen unterschiedliche Validierungspatienten.')
            by_fold.append((elements[0]['y'], np.stack([el['p'] for el in elements]).mean(0)))
            for k, element in enumerate(elements):
                seed_truth[k].append(element['y'])
                seed_prob[k].append(element['p'])
        pooled_y = np.concatenate([v[0] for v in by_fold])
        pooled_p = np.concatenate([v[1] for v in by_fold])
        for threshold in thresholds:
            threshold = float(threshold)
            result = metrics(pooled_y, pooled_p, threshold)
            seed_kappa = [metrics(np.concatenate(seed_truth[k]), np.concatenate(seed_prob[k]), threshold)['cohen_kappa']
                          for k in range(n_seeds)]
            row = dict(steps=int(step), threshold=threshold, **result,
                       seed_kappa_mean=float(np.mean(seed_kappa)),
                       seed_kappa_std=float(np.std(seed_kappa)),
                       seed_kappa_min=float(np.min(seed_kappa)))
            all_scores.append(row)
            # Pooled OOF entscheidet; bessere Seed-Stabilitaet bei Gleichstand.
            key = (result['cohen_kappa'], result['balanced_accuracy'],
                   row['seed_kappa_min'], -abs(threshold - .5), -step)
            if best is None or key > best[0]:
                best = (key, row)
    return best[1], all_scores


VERSION = '1.6.0'
METHODS = ('real', 'random', 'noise_selected', 'timegan_selected')
METRICS = ('cohen_kappa', 'balanced_accuracy', 'accuracy', 'roc_auc', 'sensitivity', 'specificity')
BASELINE = ('real', 0, 0., 0)


def write_csv(path, rows):
    """Replace only complete files; never append partial rows on Windows."""
    path = Path(path)
    if not rows:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def derived_seed(seed, *parts):
    return int(digest([int(seed), *parts])[:8], 16) % (2**31 - 1)


def build_candidates(args):
    candidates = [BASELINE]
    candidates += [('random', 0, float(ratio), 0) for ratio in args.random_ratios]
    candidates += [('noise', 0, float(copies), copies)
                   for copies in args.noise_copies if copies > 0]
    candidates += [('timegan_noise', hidden, float(ratio), copies)
                   for hidden in args.hidden for ratio in args.gan_ratios
                   for copies in args.noise_copies]
    return candidates


def candidate_fields(candidate):
    mode, hidden, ratio, copies = candidate
    return dict(mode=mode, hidden=int(hidden), ratio=float(ratio), noise_copies=int(copies))


def candidate_key(choice):
    # Equal kappa/BA: prefer simpler augmentation, then seed stability and an
    # earlier checkpoint. No tie is resolved using outer test results.
    return (choice['cohen_kappa'], choice['balanced_accuracy'],
            choice['mode'] == 'real', -choice['noise_copies'], -choice['hidden'],
            -choice['ratio'], choice['seed_kappa_min'],
            -abs(choice['threshold'] - .5), -choice['steps'])


def select_methods(trials):
    selected = {}
    for method in METHODS:
        allowed = {
            'real': {'real'}, 'random': {'random'},
            'noise_selected': {'real', 'noise'},
            'timegan_selected': {'real', 'timegan_noise'},
        }[method]
        pool = [row for row in trials if row['mode'] in allowed]
        if not pool:
            raise ValueError(f'Keine inneren Kandidaten fuer {method}.')
        selected[method] = max(pool, key=candidate_key).copy()
    return selected


def validate_dataset(x, y, groups, gy, names):
    if x.ndim != 3 or x.shape[1:] != (101, 7):
        raise ValueError('Erwartet SI7-Paare [N,101,7].')
    if not (len(x) == len(y) == len(groups)) or len(gy) != len(names):
        raise ValueError('Inkonsistente Patienten-/Sequenzzuordnung.')
    if len(set(str(name) for name in names)) != len(names):
        raise ValueError('Patientennamen sind nicht eindeutig.')
    if set(np.unique(y)) != {0, 1} or set(np.unique(gy)) != {0, 1}:
        raise ValueError('Binaere Klassen 0 und 1 erforderlich.')
    if not np.array_equal(np.unique(groups), np.arange(len(gy))):
        raise ValueError('Jeder Patient muss mindestens ein Kontaktpaar besitzen.')
    if not np.array_equal(y, gy[groups]):
        raise ValueError('Sequenzlabels und Patientenlabels widersprechen sich.')


def dataset_fingerprint(x, y, groups, gy, names):
    result = hashlib.sha256()
    result.update(digest([str(name) for name in names]).encode())
    for values in (x, y, groups, gy):
        array = np.ascontiguousarray(values)
        result.update(str(array.shape).encode())
        result.update(str(array.dtype).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def build_split_plan(gy, args):
    """Split one row per patient; sequences are never directly split."""
    plan = []
    all_patients = np.arange(len(gy))
    for repeat in range(1, args.repeats + 1):
        splitter = StratifiedKFold(args.folds, shuffle=True,
                                  random_state=derived_seed(args.seed, 'outer', repeat))
        coverage = np.zeros(len(gy), np.int64)
        for fold, (train, test) in enumerate(splitter.split(all_patients, gy), 1):
            coverage[test] += 1
            if set(train) & set(test) or set(train) | set(test) != set(all_patients):
                raise AssertionError('Ungueltige aeussere Patiententrennung.')
            if np.bincount(gy[train], minlength=2).min() < args.inner_folds:
                raise ValueError('Zu wenige aeussere Trainingspatienten pro Klasse fuer innere CV.')
            inner_splitter = StratifiedKFold(args.inner_folds, shuffle=True,
                random_state=derived_seed(args.seed, 'inner', repeat, fold))
            inner_plan, inner_coverage = [], np.zeros(len(train), np.int64)
            for inner, (a, b) in enumerate(inner_splitter.split(train, gy[train]), 1):
                inner_coverage[b] += 1
                itrain, valid = train[a], train[b]
                if set(itrain) & set(valid) or set(test) & (set(itrain) | set(valid)):
                    raise AssertionError('Ungueltige innere Patiententrennung.')
                if np.bincount(gy[itrain], minlength=2).min() < 2:
                    raise ValueError('TimeGAN benoetigt mindestens zwei Trainingspersonen jeder Klasse in jedem inneren Fold.')
                if np.bincount(gy[valid], minlength=2).min() < 1:
                    raise ValueError('Jeder innere Validierungsfold muss beide Klassen enthalten.')
                inner_plan.append(dict(inner_fold=inner, train=itrain.tolist(), valid=valid.tolist()))
            if not np.all(inner_coverage == 1):
                raise AssertionError('Innere OOF-Abdeckung unvollstaendig.')
            plan.append(dict(repeat=repeat, fold=fold, train=train.tolist(),
                             test=test.tolist(), inner=inner_plan))
        if not np.all(coverage == 1):
            raise AssertionError('Jede Person muss genau einmal pro Wiederholung getestet werden.')
    return plan


def export_manifest(plan, gy, names, fingerprint, out):
    tokens = [hashlib.sha256(str(name).encode()).hexdigest() for name in names]
    units = []
    for unit in plan:
        units.append(dict(repeat=unit['repeat'], fold=unit['fold'],
            train_patient_sha256=[tokens[v] for v in unit['train']],
            test_patient_sha256=[tokens[v] for v in unit['test']],
            inner=[dict(inner_fold=item['inner_fold'],
                        train_patient_sha256=[tokens[v] for v in item['train']],
                        validation_patient_sha256=[tokens[v] for v in item['valid']])
                   for item in unit['inner']]))
    dump_json(out / 'patient_split_manifest.json', dict(dataset_sha256=fingerprint,
        n_unique_patients=len(gy), class_counts=np.bincount(gy, minlength=2).tolist(),
        separate_holdout=False, grouping='all measurements and pairs of each patient together',
        units=units))


def make_training_sources(train_x, train_y, train_groups, minority, copies,
                          args, seed, constant):
    minority_rows = train_y == minority
    noise_x = None
    gan_x, gan_groups = train_x[minority_rows], train_groups[minority_rows]
    if copies > 0:
        noise_x = make_noisy_copies(train_x, copies, args.noise_sigma, seed, constant)
        augmented_y = np.tile(train_y, copies)
        augmented_groups = np.tile(train_groups, copies)
        gan_x = np.concatenate([gan_x, noise_x[augmented_y == minority]])
        gan_groups = np.concatenate([gan_groups, augmented_groups[augmented_y == minority]])
    return noise_x, gan_x, gan_groups


def make_noisy_copies(x, copies, sigma, seed, constant=None):
    """K copies per original pair, in training-only MinMax units."""
    if copies < 1 or int(copies) != copies or sigma <= 0 or not np.isfinite(sigma):
        raise ValueError('copies >= 1 und sigma > 0 erforderlich.')
    original = np.asarray(x, np.float32)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0., sigma, size=(int(copies),) + original.shape).astype(np.float32)
    if constant is not None:
        noise[..., np.asarray(constant, bool)] = 0.
    return np.clip(original[None] + noise, 0., 1.).reshape((-1,) + original.shape[1:]).astype(np.float32)


def metric_record(truth, probability, threshold):
    binary = (np.asarray(probability) >= threshold).astype(np.int64)
    truth = np.asarray(truth)
    return dict(n_test=len(truth), true_positives=int(np.sum((truth == 1) & (binary == 1))),
                true_negatives=int(np.sum((truth == 0) & (binary == 0))),
                false_positives=int(np.sum((truth == 0) & (binary == 1))),
                false_negatives=int(np.sum((truth == 1) & (binary == 0))),
                n_positives=int(np.sum(truth == 1)), n_negatives=int(np.sum(truth == 0)),
                **metrics(truth, probability, threshold))


def train_outer_unit(unit, args, device, clock, x, y, groups, gy, unit_dir):
    """The indivisible evaluation unit includes every method and every seed."""
    repeat, fold = unit['repeat'], unit['fold']
    train_g, test_g = np.asarray(unit['train']), np.asarray(unit['test'])
    minority = int(np.argmin(np.bincount(gy[train_g], minlength=2)))
    candidates = build_candidates(args)
    histories = {candidate: {} for candidate in candidates}
    tables = {name: [] for name in ('inner_candidates', 'inner_checkpoint_metrics',
        'inner_split_sizes', 'generation_diagnostics', 'generation_channel_diagnostics',
        'generation_distance_diagnostics', 'timegan_training_history')}
    meta = dict(repeat=repeat, fold=fold)
    unit_dir.mkdir(parents=True, exist_ok=True)

    def flush():
        for name, rows in tables.items():
            write_csv(unit_dir / (name + '.csv'), rows)

    def generator_data(real_x, real_y, owners, scaler, copies, hidden,
                       inner_fold, seed_index, stage, ratio):
        noise_seed = derived_seed(args.seed, repeat, fold, inner_fold, seed_index,
                                  'noise', copies)
        noise_x, gan_x, gan_groups = make_training_sources(real_x, real_y, owners,
            minority, copies, args, noise_seed, scaler.span <= 1e-7)
        gan_seed = derived_seed(args.seed, repeat, fold, inner_fold, seed_index,
                                'generator', hidden, copies)
        context = dict(**meta, inner_fold=inner_fold, seed_index=seed_index,
            stage=stage, noise_copies=copies, noise_sigma=args.noise_sigma,
            n_original_minority_pairs=int(np.sum(real_y == minority)),
            n_gan_training_pairs=len(gan_x), n_gan_patients=len(np.unique(gan_groups)))
        model = fit_timegan(gan_x, gan_groups, hidden, args, device, clock,
            gan_seed, f'W{repeat}/F{fold}/I{inner_fold}/S{seed_index} GAN k={copies}',
            history_rows=tables['timegan_training_history'], context=context,
            constant=scaler.span <= 1e-7)
        reference, reference_groups = real_x[real_y == minority], owners[real_y == minority]
        fake = generate(model, max(32, int(math.ceil(len(reference) * max(ratio, .5)))),
                        x.shape[1], args, device, scaler.span <= 1e-7, gan_seed + 1)
        rng = np.random.default_rng(gan_seed + 201)
        subset = rng.choice(len(reference), min(128, len(reference)), replace=False,
                            p=row_probabilities(reference_groups))
        report_meta = dict(**context, hidden=hidden, gan_seed=gan_seed)
        tables['generation_diagnostics'].append(dict(**report_meta,
            **diversity_report(fake, reference[subset])))
        channel_rows, distance = quality_reports(fake, reference[subset],
                                                 real_groups=reference_groups[subset])
        tables['generation_channel_diagnostics'].extend(dict(**report_meta, **row)
                                                        for row in channel_rows)
        tables['generation_distance_diagnostics'].append(dict(**report_meta, **distance))
        if stage == 'outer_train' and args.save_models:
            torch.save(dict(state={key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                layers=args.layers, channels=x.shape[2], seq_len=x.shape[1],
                minority_class=minority, low=scaler.low.tolist(), span=scaler.span.tolist(),
                raw_denominators=scaler.normalizer.raw_denominators_.tolist(),
                fold_imputation=scaler.normalizer.impute_.tolist(), **report_meta,
                training_only=True), unit_dir / f'seed_{seed_index}_timegan.pt')
        if stage == 'outer_train' and args.save_synthetic:
            np.savez_compressed(unit_dir / f'seed_{seed_index}_synthetic_TRAIN_ONLY.npz',
                X=scaler.inverse(fake), y=np.full(len(fake), minority),
                channels=np.asarray(SELECTED_INPUT_CHANNELS), layout='NTC_RAW_SI7_UNITS',
                repeat=repeat, fold=fold, seed_index=seed_index, noise_copies=copies)
        del model, gan_x, gan_groups
        flush()
        return fake, noise_x

    print(f'\nW{repeat}/{args.repeats} F{fold}/{args.folds}: Train/Test={len(train_g)}/{len(test_g)} '
          f'Patienten; innen={args.inner_folds}; Seeds={args.training_seeds}; '
          f'Kandidaten={len(candidates)}', flush=True)
    for inner in unit['inner']:
        inner_fold = inner['inner_fold']
        train_inner, valid_inner = np.asarray(inner['train']), np.asarray(inner['valid'])
        tr, va = np.isin(groups, train_inner), np.isin(groups, valid_inner)
        assert not set(train_inner) & (set(valid_inner) | set(test_g))
        assert not set(valid_inner) & set(test_g)
        scaler = SI7Scaler().fit(x[tr], groups[tr])
        args._current_scaler = scaler
        args._map_cache.clear()
        tx, vx = scaler.transform(x[tr]), scaler.transform(x[va])
        valid = (vx, y[va], groups[va])
        tables['inner_split_sizes'].append(dict(**meta, inner_fold=inner_fold,
            train_patients=len(train_inner), val_patients=len(valid_inner),
            train_class0=int(np.sum(gy[train_inner] == 0)),
            train_class1=int(np.sum(gy[train_inner] == 1)),
            val_class0=int(np.sum(gy[valid_inner] == 0)),
            val_class1=int(np.sum(gy[valid_inner] == 1)), outer_test_patients=len(test_g)))
        for seed_index in range(1, args.training_seeds + 1):
            classifier_seed = derived_seed(args.seed, repeat, fold, inner_fold,
                                            seed_index, 'classifier')
            fake_cache = {}
            noise_cache = {}
            for candidate in candidates:
                mode, hidden, ratio, copies = candidate
                source = None
                if mode == 'noise':
                    noise_seed = derived_seed(args.seed, repeat, fold, inner_fold,
                                              seed_index, 'noise', copies)
                    source = make_noisy_copies(tx, copies, args.noise_sigma, noise_seed,
                                               scaler.span <= 1e-7)
                    noise_cache[copies] = source
                elif mode == 'timegan_noise':
                    source_key = (hidden, copies)
                    if source_key not in fake_cache:
                        fake_cache[source_key], _ = generator_data(tx, y[tr], groups[tr],
                            scaler, copies, hidden, inner_fold, seed_index, 'inner',
                            max(args.gan_ratios))
                    source = fake_cache[source_key]
                model, history = fit_classifier(tx, y[tr], groups[tr], minority,
                    mode, ratio, source, args, device, clock, classifier_seed,
                    f'W{repeat}/F{fold}/I{inner_fold}/S{seed_index} {mode} k={copies} r={ratio:g}',
                    valid=valid, collect_history=True)
                histories[candidate][(inner_fold - 1, seed_index - 1)] = history
                del model
            del fake_cache, noise_cache
        flush()
    for candidate, history in histories.items():
        best, checkpoint_rows = select_inner_candidate(history, args.inner_folds,
            args.training_seeds, args.thresholds)
        fields = candidate_fields(candidate)
        tables['inner_candidates'].append(dict(**meta, **fields, **best))
        tables['inner_checkpoint_metrics'].extend(dict(**meta, **fields, **row)
                                                   for row in checkpoint_rows)
    selected = select_methods(tables['inner_candidates'])
    # This file is finalized BEFORE even transforming the outer test set.
    dump_json(unit_dir / 'selection.json', dict(repeat=repeat, fold=fold,
        selected=selected, selection_complete_before_outer_prediction=True,
        criterion='pooled inner OOF ensemble Cohen kappa; common checkpoint and threshold',
        candidate_tie_break='BA; no augmentation; fewer copies; smaller hidden; smaller ratio; seed stability',
        max_classifier_updates_all_candidates=args.classifier_steps,
        classifier_seed_slots=args.training_seeds, minority_class=minority))
    flush()
    del histories
    for method, choice in selected.items():
        print(f'W{repeat}/F{fold} Auswahl {method}: {choice["mode"]}, k={choice["noise_copies"]}, '
              f'Updates={choice["steps"]}, Schwelle={choice["threshold"]:.2f}; '
              f'INNER BA={choice["balanced_accuracy"]:.3f} Accuracy={choice["accuracy"]:.3f} '
              f'Kappa={choice["cohen_kappa"]:.3f}', flush=True)

    tr, te = np.isin(groups, train_g), np.isin(groups, test_g)
    scaler = SI7Scaler().fit(x[tr], groups[tr])
    args._current_scaler = scaler
    args._map_cache.clear()
    tx, test_x = scaler.transform(x[tr]), scaler.transform(x[te])
    truth_expected = gy[np.unique(groups[te])]
    seed_probability = {method: [] for method in METHODS}
    seed_rows = []
    for seed_index in range(1, args.training_seeds + 1):
        classifier_seed = derived_seed(args.seed, repeat, fold, 0, seed_index, 'classifier')
        prediction_cache, fake_cache, noise_cache = {}, {}, {}
        for method, choice in selected.items():
            mode, hidden, ratio, copies = (choice[key] for key in ('mode', 'hidden', 'ratio', 'noise_copies'))
            model_key = (mode, hidden, ratio, copies, choice['steps'])
            if model_key not in prediction_cache:
                source = None
                if mode == 'noise':
                    noise_seed = derived_seed(args.seed, repeat, fold, 0, seed_index, 'noise', copies)
                    if copies not in noise_cache:
                        noise_cache[copies] = make_noisy_copies(tx, copies, args.noise_sigma,
                            noise_seed, scaler.span <= 1e-7)
                    source = noise_cache[copies]
                elif mode == 'timegan_noise':
                    source_key = (hidden, copies)
                    if source_key not in fake_cache:
                        fake_cache[source_key], _ = generator_data(tx, y[tr], groups[tr], scaler,
                            copies, hidden, 0, seed_index, 'outer_train', ratio)
                    source = fake_cache[source_key]
                model, _ = fit_classifier(tx, y[tr], groups[tr], minority, mode, ratio,
                    source, args, device, clock, classifier_seed,
                    f'W{repeat}/F{fold}/S{seed_index} {method} REFIT', steps=choice['steps'])
                if args.save_models:
                    torch.save(dict(learned_state=model.base.learned_state(), selected=choice,
                        backbone_provenance=args._factory.provenance,
                        backbone_fingerprint=args._factory.fingerprint,
                        low=scaler.low.tolist(), denom=scaler.denom.tolist(),
                        raw_denominators=scaler.normalizer.raw_denominators_.tolist(),
                        impute=scaler.normalizer.impute_.tolist(), repeat=repeat, fold=fold,
                        seed_index=seed_index, classifier_seed=classifier_seed),
                        unit_dir / f'seed_{seed_index}_{method}_head.pt')
                truth, probability = predict_patients(model, test_x, y[te], groups[te], args, device)
                if not np.array_equal(truth, truth_expected):
                    raise AssertionError('Aeussere Patientenreihenfolge geaendert.')
                prediction_cache[model_key] = probability
                del model
            probability = prediction_cache[model_key]
            seed_probability[method].append(probability)
            seed_rows.append(dict(**meta, seed_index=seed_index, classifier_seed=classifier_seed,
                method=method, mode=mode, hidden=hidden, ratio=ratio, noise_copies=copies,
                steps=choice['steps'], threshold=choice['threshold'],
                **metric_record(truth_expected, probability, choice['threshold'])))
        del prediction_cache, fake_cache, noise_cache
    fold_rows, arrays = [], dict(test_indices=np.unique(groups[te]), y=truth_expected)
    for method, choice in selected.items():
        seeds = np.stack(seed_probability[method])
        probability = seeds.mean(axis=0)
        score = metric_record(truth_expected, probability, choice['threshold'])
        mix = (choice['ratio'] * np.sum(gy[train_g] == minority) /
              (len(train_g) + choice['ratio'] * np.sum(gy[train_g] == minority))
              if choice['mode'] in ('timegan_noise', 'random') else
              choice['ratio'] / (1 + choice['ratio']) if choice['mode'] == 'noise' else 0.)
        fold_rows.append(dict(**meta, method=method, mode=choice['mode'],
            n_seeds=args.training_seeds, hidden=choice['hidden'], ratio=choice['ratio'],
            noise_copies=choice['noise_copies'], noise_sigma=args.noise_sigma,
            augmented_draw_probability=float(mix), steps=choice['steps'],
            threshold=choice['threshold'], evaluation_scope='outer_unseen_patients', **score))
        arrays[method + '_p'] = probability
        arrays[method + '_binary'] = (probability >= choice['threshold']).astype(np.int64)
        arrays[method + '_seed_p'] = seeds
        arrays[method + '_threshold'] = np.asarray(choice['threshold'])
        print(f'W{repeat}/F{fold} {method:18s} AUSSEN BA={score["balanced_accuracy"]:.3f} '
              f'Accuracy={score["accuracy"]:.3f} Kappa={score["cohen_kappa"]:.3f}', flush=True)
    prediction_path = unit_dir / 'predictions_LOCAL_ONLY.npz'
    temporary = prediction_path.with_name(prediction_path.name + '.writing')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, prediction_path)
    write_csv(unit_dir / 'outer_fold_metrics.csv', fold_rows)
    write_csv(unit_dir / 'outer_seed_metrics.csv', seed_rows)
    flush()
    # A marker is committed only after every comparison and seed has finished.
    dump_json(unit_dir / 'completed.json', dict(**meta, run_fingerprint=args._run_fingerprint,
        n_methods=len(METHODS), n_seeds=args.training_seeds, n_test=len(truth_expected),
        prediction_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        completed=True))
    args._map_cache.clear()
    gc.collect()
    return fold_rows


def read_table(path):
    return pd.read_csv(path).to_dict(orient='records') if path.is_file() else []


def completed_units(plan, args, out, gy):
    completed = []
    for unit in plan:
        folder = out / 'units' / f'repeat_{unit["repeat"]:02d}_fold_{unit["fold"]:02d}'
        marker = folder / 'completed.json'
        if not marker.is_file():
            continue
        state = json.loads(marker.read_text(encoding='utf-8'))
        if (state.get('run_fingerprint') != args._run_fingerprint or
                state.get('n_methods') != len(METHODS) or state.get('n_seeds') != args.training_seeds or
                not state.get('completed')):
            raise ValueError(f'Wiederaufnahme abgelehnt: inkonsistenter Fold {folder.name}.')
        path = folder / 'predictions_LOCAL_ONLY.npz'
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != state['prediction_sha256']:
            raise ValueError(f'Vorhersagedatei fehlt oder wurde veraendert: {folder.name}.')
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        expected = np.asarray(unit['test'])
        if (not np.array_equal(arrays['test_indices'], expected) or
                not np.array_equal(arrays['y'], gy[expected])):
            raise ValueError('Wiederaufnahme: Testpersonen oder Labels geaendert.')
        for method in METHODS:
            if arrays[method + '_seed_p'].shape != (args.training_seeds, len(expected)):
                raise ValueError('Wiederaufnahme: inkonsistente Seed-Prognosen.')
        completed.append((unit, folder, arrays))
    return completed


def pooled_metrics(truth, probability, binary):
    row = metrics(truth, np.asarray(binary, float), .5)
    row['roc_auc'] = float(roc_auc_score(truth, probability))
    return row


def cluster_bootstrap(complete_arrays, gy, args):
    """Class-stratified patient resampling; same sample for every repeat/method.

    This is a conditional interval on stored OOF predictions. It deliberately
    makes no claim to include uncertainty from refitting or model selection.
    """
    if not complete_arrays or args.bootstrap_samples == 0:
        return {}
    rng = np.random.default_rng(derived_seed(args.seed, 'patient_bootstrap'))
    classes = [np.flatnonzero(gy == label) for label in (0, 1)]
    indices = np.concatenate([rng.choice(rows, (args.bootstrap_samples, len(rows)), replace=True)
                              for rows in classes], axis=1)
    weights = np.zeros((args.bootstrap_samples, len(gy)), np.float64)
    for b, sampled in enumerate(indices):
        weights[b] = np.bincount(sampled, minlength=len(gy))
    n, positives, negatives = len(gy), int(np.sum(gy == 1)), int(np.sum(gy == 0))
    result = {}
    for method in METHODS:
        binary = np.stack([arrays[method + '_binary'] for arrays in complete_arrays])
        tp = weights @ ((binary == 1) & (gy[None] == 1)).T
        tn = weights @ ((binary == 0) & (gy[None] == 0)).T
        accuracy = (tp + tn) / n
        expected = ((tp + negatives - tn) * positives +
                    (tn + positives - tp) * negatives) / n**2
        values = dict(cohen_kappa=(accuracy - expected) / (1. - expected),
                      balanced_accuracy=.5 * (tp / positives + tn / negatives),
                      accuracy=accuracy, sensitivity=tp / positives, specificity=tn / negatives)
        result[method] = {key: value.mean(axis=1) for key, value in values.items()}
    return result


def export_results(plan, args, out, gy, *, bootstrap=False):
    completed = completed_units(plan, args, out, gy)
    fold_rows, seed_rows, selections = [], [], []
    table_names = ('inner_candidates', 'inner_checkpoint_metrics', 'inner_split_sizes',
        'generation_diagnostics', 'generation_channel_diagnostics',
        'generation_distance_diagnostics', 'timegan_training_history')
    tables = {name: [] for name in table_names}
    by_repeat = {}
    for unit, folder, arrays in completed:
        fold_rows.extend(read_table(folder / 'outer_fold_metrics.csv'))
        seed_rows.extend(read_table(folder / 'outer_seed_metrics.csv'))
        choice = json.loads((folder / 'selection.json').read_text(encoding='utf-8'))['selected']
        selections.extend(dict(row, method=method)
                          for method, row in choice.items())
        for name in table_names:
            tables[name].extend(read_table(folder / (name + '.csv')))
        by_repeat.setdefault(unit['repeat'], []).append((unit, arrays))
    repeat_rows, repeat_seed_rows, complete_arrays = [], [], []
    completed_repeats = []
    for repeat in sorted(by_repeat):
        units = sorted(by_repeat[repeat], key=lambda item: item[0]['fold'])
        indices = np.concatenate([arrays['test_indices'] for _, arrays in units])
        order = np.argsort(indices)
        if len(np.unique(indices)) != len(indices):
            raise AssertionError('Eine Testperson tritt doppelt innerhalb einer Wiederholung auf.')
        truth = np.concatenate([arrays['y'] for _, arrays in units])[order]
        is_complete = len(units) == args.folds
        if is_complete and not np.array_equal(np.sort(indices), np.arange(len(gy))):
            raise AssertionError('Vollstaendige Wiederholung deckt nicht alle Patienten ab.')
        combined = {}
        for method in METHODS:
            probability = np.concatenate([arrays[method + '_p'] for _, arrays in units])[order]
            binary = np.concatenate([arrays[method + '_binary'] for _, arrays in units])[order]
            combined[method + '_binary'] = binary
            repeat_rows.append(dict(repeat=repeat, method=method, n_completed_folds=len(units),
                repeat_complete=is_complete, n_unique_test_patients=len(truth),
                **pooled_metrics(truth, probability, binary)))
            for slot in range(args.training_seeds):
                sp = np.concatenate([arrays[method + '_seed_p'][slot] for _, arrays in units])[order]
                sb = np.concatenate([(arrays[method + '_seed_p'][slot] >=
                                      arrays[method + '_threshold']).astype(np.int64)
                                     for _, arrays in units])[order]
                repeat_seed_rows.append(dict(repeat=repeat, seed_index=slot + 1, method=method,
                    repeat_complete=is_complete, n_unique_test_patients=len(truth),
                    **pooled_metrics(truth, sp, sb)))
        if is_complete:
            completed_repeats.append(repeat)
            complete_arrays.append(combined)
    summary = []
    for method in METHODS:
        rows = [row for row in repeat_rows if row['method'] == method and row['repeat_complete']]
        if not rows:
            continue
        row = dict(method=method, n_completed_repeats=len(rows), planned_repeats=args.repeats,
            n_completed_outer_folds=len(completed), n_unique_patients=len(gy),
            n_seeds_per_ensemble=args.training_seeds,
            aggregation='mean of pooled OOF metrics per COMPLETE repeat; not independent replicates')
        for metric in METRICS:
            row[metric] = float(np.mean([item[metric] for item in rows]))
            row['sd_repeat_' + metric] = (float(np.std([item[metric] for item in rows], ddof=1))
                                          if len(rows) > 1 else None)
            row['mean_outer_fold_' + metric] = float(np.mean([item[metric] for item in fold_rows
                if item['method'] == method and int(item['repeat']) in completed_repeats]))
        seed_values = [item for item in repeat_seed_rows if item['method'] == method and item['repeat_complete']]
        row['mean_single_seed_kappa'] = float(np.mean([item['cohen_kappa'] for item in seed_values]))
        row['fraction_outer_folds_using_augmentation'] = float(np.mean([
            item['mode'] != 'real' for item in selections
            if item['method'] == method and int(item['repeat']) in completed_repeats]))
        summary.append(row)
    if summary:
        real = next(item for item in summary if item['method'] == 'real')
        for row in summary:
            for metric in ('cohen_kappa', 'balanced_accuracy', 'accuracy'):
                row['delta_' + metric + '_vs_real'] = row[metric] - real[metric]

    pairs = [('timegan_selected', 'real'), ('timegan_selected', 'random'),
             ('timegan_selected', 'noise_selected'), ('random', 'real'), ('noise_selected', 'real')]
    fold_differences, repeat_differences, comparisons = [], [], []
    by_fold_method = {(int(row['repeat']), int(row['fold']), row['method']): row for row in fold_rows}
    by_repeat_method = {(int(row['repeat']), row['method']): row for row in repeat_rows if row['repeat_complete']}
    boot = cluster_bootstrap(complete_arrays, gy, args) if bootstrap else {}
    for target, reference in pairs:
        for unit, _, _ in completed:
            a = by_fold_method[(unit['repeat'], unit['fold'], target)]
            b = by_fold_method[(unit['repeat'], unit['fold'], reference)]
            fold_differences.append(dict(repeat=unit['repeat'], fold=unit['fold'],
                target=target, reference=reference, **{'delta_' + metric: a[metric] - b[metric]
                                                    for metric in METRICS}))
        for repeat in completed_repeats:
            a, b = by_repeat_method[(repeat, target)], by_repeat_method[(repeat, reference)]
            repeat_differences.append(dict(repeat=repeat, target=target, reference=reference,
                **{'delta_' + metric: a[metric] - b[metric] for metric in METRICS}))
        for metric in METRICS:
            deltas = [row['delta_' + metric] for row in repeat_differences
                      if row['target'] == target and row['reference'] == reference]
            if not deltas:
                continue
            folds = [row['delta_' + metric] for row in fold_differences
                     if row['target'] == target and row['reference'] == reference
                     and int(row['repeat']) in completed_repeats]
            ci = [None, None]
            if boot and metric != 'roc_auc':
                ci = np.quantile(boot[target][metric] - boot[reference][metric], [.025, .975]).tolist()
            comparisons.append(dict(target=target, reference=reference, metric=metric,
                mean_delta=float(np.mean(deltas)), n_complete_repeats=len(deltas),
                n_unique_patients=len(gy), n_paired_outer_folds=len(folds),
                fraction_repeats_target_better=float(np.mean(np.asarray(deltas) > 1e-12)),
                fraction_folds_target_better=float(np.mean(np.asarray(folds) > 1e-12)),
                fraction_folds_tied=float(np.mean(np.abs(folds) <= 1e-12)),
                conditional_patient_bootstrap_low=ci[0], conditional_patient_bootstrap_high=ci[1],
                bootstrap_samples=args.bootstrap_samples if boot and metric != 'roc_auc' else 0,
                uncertainty='conditional on fitted OOF models; no retraining; folds/repeats/seeds dependent'))
    frequency = {}
    for row in selections:
        key = tuple(row[name] for name in ('method', 'mode', 'hidden', 'ratio', 'noise_copies', 'steps'))
        frequency.setdefault(key, []).append(row)
    frequency_rows = []
    for key, rows in frequency.items():
        frequency_rows.append(dict(zip(('method', 'mode', 'hidden', 'ratio', 'noise_copies', 'steps'), key),
            selected_outer_folds=len(rows), total_completed_outer_folds=len(completed),
            fraction_selected=len(rows) / len(completed),
            mean_selected_threshold=float(np.mean([item['threshold'] for item in rows]))))
    outputs = dict(tables, outer_fold_metrics=fold_rows, outer_seed_metrics=seed_rows,
        repeat_patient_pooled=repeat_rows, repeat_seed_patient_pooled=repeat_seed_rows,
        summary_patient_pooled=summary, paired_outer_fold_differences=fold_differences,
        paired_repeat_differences=repeat_differences, paired_comparison_summary=comparisons,
        selected_configurations=selections, selection_frequency=frequency_rows)
    for name, rows in outputs.items():
        write_csv(out / (name + '.csv'), rows)
    return dict(n_completed_folds=len(completed), completed_repeats=completed_repeats,
                summary=summary)


def run(args, device, clock, x, y, groups, gy, names, out, plan):
    existing = completed_units(plan, args, out, gy)
    done = {(unit['repeat'], unit['fold']) for unit, _, _ in existing}
    status, added = 'running', 0
    state = dict(n_completed_folds=len(done), completed_repeats=[], summary=[])

    def save_status():
        dump_json(out / 'run_status.json', dict(status=status,
            completed_outer_folds=state['n_completed_folds'], planned_outer_folds=len(plan),
            completed_repeats=state['completed_repeats'], planned_repeats=args.repeats,
            inner_folds=args.inner_folds, training_seeds=args.training_seeds,
            runtime_minutes_this_session=clock.minutes(), resumed_outer_folds=len(done),
            total_qc_patients=len(gy), patient_class_counts=np.bincount(gy, minlength=2).tolist(),
            all_patients_used=True, separate_final_holdout=False,
            evaluation_scope='repeated nested CV on all patients; outer tests exclude training and selection',
            technical_self_test=args.self_test, run_fingerprint=args._run_fingerprint,
            post_hoc_robustness_analysis=True, optimization_metric='inner pooled OOF ensemble Cohen kappa',
            primary_evaluation='mean per-repeat patient-pooled OOF metrics of the seed ensemble',
            caution='repeated folds and seeds are dependent; conditional bootstrap omits retraining uncertainty'))

    try:
        state = export_results(plan, args, out, gy)
        save_status()
        for unit in plan:
            key = unit['repeat'], unit['fold']
            if key in done:
                print(f'W{key[0]}/F{key[1]} vollstaendig gespeichert: uebersprungen.', flush=True)
                continue
            if added >= args.fold_limit:
                break
            clock.check()
            folder = out / 'units' / f'repeat_{key[0]:02d}_fold_{key[1]:02d}'
            train_outer_unit(unit, args, device, clock, x, y, groups, gy, folder)
            added += 1
            state = export_results(plan, args, out, gy)
            remaining = len(plan) - state['n_completed_folds']
            print(f'Fortschritt: {state["n_completed_folds"]}/{len(plan)} aeussere Folds; '
                  f'grobe Restzeit {clock.minutes() / added * remaining / 60:.2f} h '
                  f'(aus dieser Sitzung geschaetzt).', flush=True)
            save_status()
        status = 'complete' if state['n_completed_folds'] == len(plan) else 'partial_requested'
    except (BudgetExpired, KeyboardInterrupt) as exc:
        status = 'budget_exceeded' if isinstance(exc, BudgetExpired) else 'interrupted'
        print(str(exc) or 'Unterbrochen; vollstaendige aeussere Folds bleiben gespeichert.', flush=True)
    except Exception:
        status = 'failed'
        raise
    finally:
        # Only committed folds enter aggregate metrics, even on interruption.
        state = export_results(plan, args, out, gy, bootstrap=True)
        save_status()
        print(f'\nStatus={status}; Nested-CV={state["n_completed_folds"]}/{len(plan)} aeussere Folds; '
              f'vollstaendige Wiederholungen={len(state["completed_repeats"])}/{args.repeats}; '
              f'{clock.minutes():.1f} min in dieser Sitzung.', flush=True)
        for row in state['summary']:
            print(f'{row["method"]:18s} BA={row["balanced_accuracy"]:.3f} '
                  f'Accuracy={row["accuracy"]:.3f} Kappa={row["cohen_kappa"]:.3f} '
                  f'(Mittel ueber {row["n_completed_repeats"]} vollstaendige Wiederholungen)', flush=True)
        if not state['summary']:
            print('Noch keine vollstaendige Wiederholung; Fold-Ergebnisse sind separat gespeichert.', flush=True)
        print('Ausgabe:', out.resolve(), flush=True)
    return 0 if status in ('complete', 'partial_requested') else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', nargs='?', type=Path, help='Bisheriger Ordner mit Original-CSV-Exporten.')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output-dir', '--out', dest='out', type=Path)
    parser.add_argument('--weights-dir', type=Path, default=Path('pretrained_weights_v36'))
    parser.add_argument('--resume', action='store_true', help='Identische Konfiguration fortsetzen; fertige aeussere Folds wiederverwenden.')
    parser.add_argument('--hours', type=float, default=24., help='Maximales Zeitbudget DIESER Sitzung in Stunden; kein Soll-Trainingsumfang.')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--folds', '--outer-folds', type=int, default=3)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--training-seeds', type=int, default=3)
    parser.add_argument('--fold-limit', type=int, default=999999, help='Hoechstens N noch fehlende aeussere Folds je Sitzung rechnen.')
    parser.add_argument('--seed', type=int, default=42, help='Masterseed fuer unabhaengige Split-/Training-/Generator-Streams.')
    parser.add_argument('--hidden', default='16', help='GAN-Breiten als Zahlenliste; Standard 16.')
    parser.add_argument('--noise-copies', default='0,3,4,5', help='GAN-Rauschkopien zur inneren Auswahl; 0 prueft GAN ohne Rauschkopien.')
    parser.add_argument('--noise-sigma', type=float, default=.02)
    parser.add_argument('--gan-ratios', '--gan-ratio', dest='gan_ratios', default='0.5')
    parser.add_argument('--random-ratios', '--random-ratio', dest='random_ratios', default='0.5')
    parser.add_argument('--thresholds', default='0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--contact-microbatch', type=int, default=16)
    parser.add_argument('--ae-steps', type=int, default=500)
    parser.add_argument('--supervisor-steps', type=int, default=500)
    parser.add_argument('--gan-steps', type=int, default=1500)
    parser.add_argument('--classifier-steps', type=int, default=300)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--diagnostic-every', type=int, default=250)
    parser.add_argument('--gan-lr', type=float, default=.001)
    parser.add_argument('--classifier-lr', type=float, default=.001)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--bootstrap-samples', type=int, default=2000,
                        help='Gepaarte Patienten-Bootstraps auf gespeicherten Prognosen; 0 deaktiviert.')
    parser.add_argument('--save-synthetic', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--check-data', action='store_true', help='Nur Import, QC und vollstaendigen Splitplan pruefen.')
    parser.add_argument('--show-import-log', action='store_true')
    parser.add_argument('--self-test', action='store_true', help='Kleiner kompletter CPU-/CUDA-Test mit erzeugten CSVs und technischem Backbone.')
    parser.add_argument('--force-threshold', type=float, default=0.)
    parser.add_argument('--contact-duration-tolerance', type=float, default=.25)
    parser.add_argument('--contact-zero-tolerance', type=float, default=0.)
    parser.add_argument('--min-si-pairs-per-subject', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        args.hidden = sorted(set(int(v) for v in args.hidden.split(',')))
        args.noise_copies = sorted(set(int(v) for v in args.noise_copies.split(',')))
        args.gan_ratios = sorted(set(float(v) for v in args.gan_ratios.split(',')))
        args.random_ratios = sorted(set(float(v) for v in args.random_ratios.split(',')))
        args.thresholds = sorted(set(float(v) for v in args.thresholds.split(',')))
    except ValueError:
        parser.error('Suchraeume als kommagetrennte Zahlenlisten angeben.')
    positive = [args.hours, args.repeats, args.training_seeds, args.fold_limit,
        args.batch_size, args.contact_microbatch, args.ae_steps, args.supervisor_steps,
        args.gan_steps, args.classifier_steps, args.eval_every, args.log_every,
        args.diagnostic_every, args.gan_lr, args.classifier_lr, args.threads,
        args.noise_sigma, args.min_si_pairs_per_subject]
    if not all(np.isfinite(positive)) or min(positive) <= 0:
        parser.error('Trainingsparameter und Budgets muessen endlich und positiv sein.')
    if args.folds < 2 or args.inner_folds < 2 or args.layers < 2:
        parser.error('folds, inner-folds und layers muessen >=2 sein.')
    if not args.hidden or min(args.hidden) < 2 or not args.noise_copies or min(args.noise_copies) < 0:
        parser.error('--hidden >=2 und --noise-copies >=0 erforderlich.')
    if not any(v > 0 for v in args.noise_copies):
        parser.error('Mindestens eine positive Kopienanzahl fuer die Rauschkontrolle erforderlich.')
    for name, values in (('gan-ratios', args.gan_ratios), ('random-ratios', args.random_ratios)):
        if not values or not all(np.isfinite(values)) or min(values) <= 0:
            parser.error(f'--{name}: endliche positive Zahlen erforderlich.')
    if not args.thresholds or not all(np.isfinite(args.thresholds)) or min(args.thresholds) <= 0 or max(args.thresholds) >= 1:
        parser.error('--thresholds muessen innerhalb (0,1) liegen.')
    if not 0 <= args.seed < 2**32 or args.bootstrap_samples < 0:
        parser.error('--seed in 0..2**32-1 und --bootstrap-samples >=0 erforderlich.')
    qc = [args.contact_duration_tolerance, args.contact_zero_tolerance, args.force_threshold]
    if not all(np.isfinite(qc)) or min(qc) < 0:
        parser.error('QC-Toleranzen muessen endlich und nichtnegativ sein.')
    if args.out is None:
        stem = 'results_timegan_nested_selftest' if args.self_test else 'results_timegan_nested_v16'
        args.out = Path(stem + '_' + time.strftime('%Y%m%d_%H%M%S'))
    if args.out.exists() and not args.out.is_dir():
        parser.error('--output-dir muss ein Ordner sein.')
    nonempty = args.out.is_dir() and any(args.out.iterdir())
    if nonempty and not args.resume:
        parser.error('Ausgabeordner ist belegt: fuer denselben Lauf --resume, sonst neuen --output-dir verwenden.')
    if args.resume and not (args.out / 'configuration.json').is_file():
        parser.error('--resume benoetigt den Ergebnisordner genau dieses V1.6-Laufs mit configuration.json.')
    torch.set_num_threads(args.threads)
    device = torch.device('cpu')
    if args.device == 'cuda' and not args.check_data:
        if not torch.cuda.is_available():
            parser.error('CUDA-PyTorch fehlt in diesem Python; bestehende CUDA-venv verwenden.')
        device = torch.device('cuda')
        probe = nn.GRU(2, 4, batch_first=True).to(device)
        probe(torch.ones(2, 8, 2, device=device))[0].sum().backward()
        torch.cuda.synchronize()
        del probe
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(f'GPU={torch.cuda.get_device_name(0)}; CUDA-PyTorch={torch.version.cuda}; float32', flush=True)
    else:
        print('Nur Import-/Splitpruefung.' if args.check_data else 'Expliziter CPU-Modus; keine GTX-1070-Laufzeitmessung.', flush=True)
    temporary = None
    try:
        if args.self_test:
            temporary = tempfile.TemporaryDirectory(prefix='nested_v16_test_', dir=Path.cwd())
            args.data_dir = Path(temporary.name)
            create_test_csvs(args.data_dir)
            args.repeats, args.training_seeds = min(args.repeats, 2), min(args.training_seeds, 2)
            args.hidden, args.noise_copies = [4], [0, 1]
            args.ae_steps = args.supervisor_steps = args.gan_steps = 1
            args.classifier_steps, args.eval_every = 2, 1
            args.thresholds = [.25, .5, .75]
            args.batch_size = min(args.batch_size, 4)
            args.bootstrap_samples = min(args.bootstrap_samples, 50)
            print('SELBSTTEST: erzeugte CSVs, technischer Backbone, reduzierte Updates; keine Aussage zur Modellguete.', flush=True)
        if args.data_dir is None:
            args.data_dir = Path('CSV_Export')
        if not args.data_dir.is_dir():
            raise FileNotFoundError(f'CSV-Ordner fehlt: {args.data_dir}. Als ersten Parameter bisherigen CSV-Export-Ordner angeben.')
        x, y, groups, gy, names, audit = load_si7_csv(args)
        validate_dataset(x, y, groups, gy, names)
        plan = build_split_plan(gy, args)
        fingerprint = dataset_fingerprint(x, y, groups, gy, names)
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if not key.startswith('_')}
        config.update(version=VERSION, dataset_sha256=fingerprint,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            import_source_script_sha256=IMPORT_SOURCE_SCRIPT_SHA256,
            torch_version=str(torch.__version__), input_shape=list(x.shape),
            n_unique_patients=len(gy), patient_class_counts=np.bincount(gy, minlength=2).tolist(),
            channel_names=list(SELECTED_INPUT_CHANNELS), model=probe_config().row(), precision='float32',
            synthetic_angles=False, separate_holdout=False, all_patients_used=True,
            selection_metric='pooled inner OOF ensemble Cohen kappa',
            checkpoint_grid=sorted(set(list(range(args.eval_every, args.classifier_steps + 1,
                                                args.eval_every)) + [args.classifier_steps])),
            candidate_grid=[candidate_fields(candidate) for candidate in build_candidates(args)],
            primary_prediction='mean probability across all seed slots; common inner-selected threshold',
            raw_input_scaling='original FoldInputNormalizer; GAN MinMax undone before signal_images',
            post_hoc_robustness_analysis=True)
        ignore = {'out', 'data_dir', 'weights_dir', 'resume', 'hours', 'fold_limit',
                  'contact_microbatch', 'check_data', 'show_import_log', 'threads',
                  'log_every', 'save_models', 'save_synthetic', 'bootstrap_samples'}
        identity = {key: value for key, value in config.items() if key not in ignore}
        args._run_fingerprint = digest(identity)
        config['run_fingerprint'] = args._run_fingerprint
        if args.resume:
            saved = json.loads((args.out / 'configuration.json').read_text(encoding='utf-8'))
            if saved.get('run_fingerprint') != args._run_fingerprint:
                raise ValueError('Wiederaufnahme abgelehnt: Daten, Suchraum, Splits, Seeds, Code oder Trainingsparameter unterscheiden sich. Identische Parameter oder neuen Ergebnisordner verwenden.')
        args.out.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            dump_json(args.out / 'configuration.json', config)
        write_csv(args.out / 'import_audit_aggregate.csv', audit)
        export_manifest(plan, gy, names, fingerprint, args.out)
        if args.check_data:
            print(f'Import-/Splitpruefung bestanden: {len(plan)} aeussere Folds; ALLE {len(gy)} Patienten; kein separater Holdout.', flush=True)
            return 0
        clock = Clock(args.hours)
        args._device, args._clock, args._map_cache = device, clock, OrderedDict()
        args._factory = BackboneFactory(args, synthetic=args.self_test)
        backbone_path = args.out / 'backbone.json'
        backbone = dict(provenance=args._factory.provenance, fingerprint=args._factory.fingerprint)
        if backbone_path.is_file() and json.loads(backbone_path.read_text(encoding='utf-8')) != backbone:
            raise ValueError('Wiederaufnahme abgelehnt: Backbone-Gewichte geaendert.')
        dump_json(backbone_path, backbone)
        print(f'ALLE {len(gy)} Personen, Klassen={np.bincount(gy, minlength=2).tolist()}; '
              f'{args.repeats} Wiederholungen x {args.folds} aeussere x {args.inner_folds} innere Folds x '
              f'{args.training_seeds} Seeds; Kappa-Auswahl; {len(build_candidates(args))} Kandidaten.', flush=True)
        print(f'ConvNeXt-Tiny eingefroren + kleiner Transformer; max. {args.classifier_steps} Updates '
              f'je Klassifikator; Sitzungsbudget {args.hours:g} h; Fortsetzung mit --resume.', flush=True)
        return run(args, device, clock, x, y, groups, gy, names, args.out, plan)
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise SystemExit(str(exc)) from exc


