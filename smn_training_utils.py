#!/usr/bin/env python3
r"""SI7-TimeGAN V1.7: drei vorab festgelegte Trainingsvarianten vergleichen.

Diese Datei neben timegan_si7_nested_gtx1070_v1_6.py ablegen. Sie laedt die
vorhandene V1.6-Pipeline direkt; CSV-Import, SI7-Bildung, Kontakt-QC,
patientenweise Nested-CV und ConvNeXt-Transformer stammen aus dieser Datei.
Es ist kein NPZ-Export und keine Aenderung an der V1.6-Datei erforderlich.

PowerShell, in der bisherigen CUDA-venv:
  python .\timegan_si7_training_compare_gtx1070_v1_7.py .\CSV_Export --device cuda --output-dir .\results_timegan_training_v17
Fortsetzen:
  python .\timegan_si7_training_compare_gtx1070_v1_7.py .\CSV_Export --device cuda --output-dir .\results_timegan_training_v17 --resume
Technischer Test nur des neuen Trainingskerns (ohne V1.6 und CSV-Dateien):
  python .\timegan_si7_training_compare_gtx1070_v1_7.py --self-test --device cpu

Standard: EINE Wiederholung einer vollstaendigen 3x3-Patienten-Nested-CV,
3 Seed-Slots, fuer jede der drei Bedingungen. --repeats 5 erweitert den
Vergleich auf fuenf Wiederholungen. Alle Bedingungen bekommen IDENTISCHE
Patientenaufteilungen und Klassifikator-Seeds. Kein separater Holdout.

  reference_1500_1g: 1500 GAN-Zyklen, 1 G/S- und 1 E/R-Update je Zyklus
  longer_3000_1g:    3000 GAN-Zyklen, 1 G/S- und 1 E/R-Update je Zyklus
  ratio_1500_2g:     1500 GAN-Zyklen, 2 G/S- und 2 E/R-Updates je Zyklus

Pro GAN-Zyklus gibt es maximal ein D-Update. Wie bisher wird es nur bei
D-Loss > 0.15 ausgefuehrt; tatsaechliche Updatezaehler werden gespeichert.
Die dritte Bedingung uebernimmt das Update-VERHAELTNIS des TimeGAN-Autorencodes.
Die bestehende kompakte PyTorch-Portierung bleibt die Grundlage.

AE-/Supervisor-Vortraining standardmaessig je 500 Updates. GAN-Breite 16,
2 GRU-Lagen, Rauschkopien 0/3/4/5 und Beimischung 0.5 bleiben im Vergleich
gleich. Die GAN-Kopienzahl und die Klassifikationsschwelle werden jeweils
innen nach Patienten-Kappa ausgewaehlt; die Baseline darf gewaehlt werden.
ALLE Klassifikatoren werden exakt 300 Updates trainiert (anpassbar), damit
die Auswahl der Trainingsdauer den Augmentationsvergleich nicht ueberlagert.
Nur der abschliessende Klassifikator-Checkpoint steht zur inneren Auswahl.

Erweiterte Diagnosen: Varianz von Encoder, Generator und Supervisor,
Ausgabesensitivitaet gegenueber unterschiedlichen Zufallseingaben,
Gradientennormen vor Clipping und tatsaechliche Optimierungszaehler.
Sehr kleine generierte Varianz vor dem adversarialen Training ist allein
kein Kollapsnachweis: G wurde in AE/SUP noch nicht adversarial trainiert.
Nur der Trainingsstand am Ende wird zur Klassifikator-Augmentation verwendet.

Ausgabe je Bedingung in einem eigenen Unterordner. Uebergreifend:
  training_variant_summary.csv
  training_variant_fold_differences.csv
  training_variant_repeat_differences.csv
  generator_training_summary.csv
  generator_trajectory_summary.csv
  baseline_consistency.csv
Unvollstaendige aeussere Folds werden nicht in Vergleiche aufgenommen.
Das gemeinsame --hours-Budget (Standard 24 h) gilt fuer ALLE Bedingungen.
Gespeicherte komplette aeussere Folds werden bei --resume wiederverwendet.

Primaere Auswertung: Drei-Seed-Ensemble, Patientenmetriken je kompletter
Wiederholung. Wiederholungen, Folds und Seeds sind abhaengig. Die aeusseren
Testergebnisse werden nicht fuer eine erneute innere Auswahl verwendet.
Dies bleibt eine nachtraegliche Robustheitsanalyse auf dem bekannten Datensatz.

Referenz zum 2:1-Verhaeltnis:
  https://github.com/jsyoon0823/TimeGAN/blob/master/timegan.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace


VERSION = '1.7.0'
BASE_FILENAME = 'timegan_si7_nested_gtx1070_v1_6.py'
METHODS = ('real', 'random', 'noise_selected', 'timegan_selected')
SCHEME_NAMES = ('reference_1500_1g', 'longer_3000_1g', 'ratio_1500_2g')
METRICS = ('cohen_kappa', 'balanced_accuracy', 'accuracy', 'roc_auc', 'sensitivity', 'specificity')


@dataclass(frozen=True)
class TrainingScheme:
    name: str
    gan_cycles: int
    generator_updates_per_cycle: int


def build_schemes(reference_cycles=1500):
    if reference_cycles < 1:
        raise ValueError('reference-cycles muss positiv sein.')
    return [TrainingScheme(SCHEME_NAMES[0], reference_cycles, 1),
            TrainingScheme(SCHEME_NAMES[1], 2 * reference_cycles, 1),
            TrainingScheme(SCHEME_NAMES[2], reference_cycles, 2)]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        if path.is_file():
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


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def number(value):
    return float(value) if value not in (None, '', 'None', 'nan') else None


def yes(value):
    return value is True or str(value).lower() == 'true'


def load_base(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f'Die vorhandene Pipeline fehlt: {path}\n'
            f'Diese neue Datei neben {BASE_FILENAME} ablegen oder '
            '--base-script mit dem Pfad zu dieser Datei angeben.')
    spec = importlib.util.spec_from_file_location('_si7_v16_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    validate_base(module)
    return module


def validate_base(base):
    required = ('main', 'TimeGAN', 'generate', 'seed_all', 'row_probabilities',
                'active', 'diversity_report', 'candidate_fields', 'fit_timegan',
                'build_split_plan', 'train_outer_unit', 'select_inner_candidate',
                'np', 'torch', 'F')
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        raise ValueError('Die Basisdatei hat nicht die V1.6-Schnittstelle: ' + ', '.join(missing))
    version = str(getattr(base, 'VERSION', ''))
    if not version.startswith('1.6.'):
        raise ValueError(f'Erwartet wird die SI7-Nested-CV-Pipeline V1.6, gefunden: {version or "ohne Versionskennung"}.')
    signature = inspect.signature(base.fit_timegan)
    required_parameters = ('x', 'groups', 'hidden', 'args', 'device', 'clock',
                           'seed', 'tag', 'history_rows', 'context', 'constant')
    if any(name not in signature.parameters for name in required_parameters):
        raise ValueError('fit_timegan besitzt eine andere Schnittstelle. Bitte die V1.6-Datei verwenden.')


def make_timegan_fit(base, scheme, *, session_id, live_path=None):
    """Preserve V1.6 input/output; vary only joint cycles and G/E update count."""
    np, torch, F = base.np, base.torch, base.F

    def fit_timegan(x, groups, hidden, args, device, clock, seed, tag,
                    history_rows=None, context=None, constant=None):
        base.seed_all(seed)
        rng = np.random.default_rng(seed)
        model = base.TimeGAN(x.shape[2], hidden, args.layers).to(device).train()
        er = list(model.E.parameters()) + list(model.R.parameters())
        gs = list(model.G.parameters()) + list(model.S.parameters())
        oe = torch.optim.Adam(er, lr=args.gan_lr)
        osup = torch.optim.Adam(model.S.parameters(), lr=args.gan_lr)
        og = torch.optim.Adam(gs, lr=args.gan_lr)
        od = torch.optim.Adam(model.D.parameters(), lr=args.gan_lr)
        probabilities = base.row_probabilities(groups)
        bce = torch.nn.BCEWithLogitsLoss()
        history_rows = history_rows if history_rows is not None else []
        context = dict(context or {})
        constant = np.zeros(x.shape[2], bool) if constant is None else constant
        reference = x[rng.choice(len(x), min(128, len(x)), replace=False, p=probabilities)]
        counts = dict(generator_updates=0, embedder_updates=0,
                      supervisor_pretrain_updates=0, discriminator_updates=0)
        global_step = 0

        def batch():
            indices = rng.choice(len(x), args.batch_size, replace=True, p=probabilities)
            return torch.from_numpy(x[indices]).to(device)

        def supervise(h):
            return F.mse_loss(model.S(h)[:, :-1], h[:, 1:])

        def optimize(optimizer, loss, parameters):
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Nichtendlicher GAN-Loss; Lauf wird abgebrochen.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(parameters), 5., error_if_nonfinite=True)
            optimizer.step()
            return float(norm.item())

        def spread(tensor):
            return float(tensor.std(dim=0, unbiased=False).mean().item())

        def snapshot(phase, step, phase_steps, losses):
            fake = base.generate(model, min(64, max(32, len(reference))), x.shape[1],
                                 args, device, constant, seed + 900000)
            with torch.no_grad():
                real = torch.from_numpy(reference).to(device)
                h_real = model.E(real)
                reconstruction = F.mse_loss(model.R(h_real), real).item()
                supervision = supervise(h_real).item()
                # Separate RNG: diagnostics never advance the training RNG.
                generator = torch.Generator(device=device)
                generator.manual_seed(int(seed + 900001))
                z = torch.rand(min(32, max(8, len(reference))), x.shape[1], model.channels,
                               device=device, generator=generator)
                eh = model.G(z)
                hh = model.S(eh)
                decoded = model.R(hh)
                decoded[..., torch.as_tensor(constant, dtype=torch.bool, device=device)] = 0
                z_distance = torch.mean(torch.abs(z - z.flip(0))).item()
                output_distance = torch.mean(torch.abs(decoded - decoded.flip(0))).item()
                extra = dict(real_embedding_std=spread(h_real),
                             fake_embedding_std=spread(eh),
                             fake_supervisor_std=spread(hh),
                             decoded_probe_std=spread(decoded),
                             noise_sensitivity_l1=float(output_distance / max(z_distance, 1e-12)))
            row = dict(context)
            row.update(training_variant=scheme.name,
                generator_updates_per_cycle=scheme.generator_updates_per_cycle,
                planned_gan_cycles=scheme.gan_cycles, session_id=session_id,
                hidden=hidden, gan_seed=seed, phase=phase, phase_step=step,
                phase_total=phase_steps, global_step=global_step,
                runtime_minutes=clock.minutes(), reconstruction_mse=float(reconstruction),
                supervisor_mse=float(supervision), **counts, **extra,
                generator_loss=losses.get('generator_loss'),
                adversarial_loss=losses.get('adversarial_loss'),
                moment_loss=losses.get('moment_loss'),
                discriminator_loss=losses.get('discriminator_loss'),
                generator_grad_norm=losses.get('generator_grad_norm'),
                embedder_grad_norm=losses.get('embedder_grad_norm'),
                discriminator_grad_norm=losses.get('discriminator_grad_norm'),
                **base.diversity_report(fake, reference))
            history_rows.append(row)
            if live_path is not None:
                path = Path(live_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
            # cuDNN GRU backward requires train mode even through frozen blocks.
            model.train()

        phases = [('AE', args.ae_steps), ('SUP', args.supervisor_steps), ('GAN', scheme.gan_cycles)]
        for phase, steps in phases:
            phase_start = time.monotonic()
            for step in range(1, steps + 1):
                clock.check()
                global_step += 1
                real = batch()
                losses = {}
                if phase == 'AE':
                    base.active(model, {'E', 'R'})
                    loss = 10 * torch.sqrt(F.mse_loss(model.R(model.E(real)), real) + 1e-8)
                    losses['embedder_grad_norm'] = optimize(oe, loss, er)
                    counts['embedder_updates'] += 1
                elif phase == 'SUP':
                    base.active(model, {'S'})
                    with torch.no_grad():
                        h = model.E(real)
                    loss = supervise(h)
                    optimize(osup, loss, model.S.parameters())
                    counts['supervisor_pretrain_updates'] += 1
                else:
                    for g_update in range(scheme.generator_updates_per_cycle):
                        clock.check()
                        if g_update:
                            real = batch()
                        base.active(model, {'G', 'S'})
                        with torch.no_grad():
                            h = model.E(real)
                        eh = model.G(torch.rand_like(real))
                        hh = model.S(eh)
                        generated = model.R(hh)
                        adversarial = bce(model.D(hh), torch.ones_like(hh[:, :, :1]))
                        adversarial = adversarial + bce(model.D(eh), torch.ones_like(eh[:, :, :1]))
                        moment = (generated.mean(0) - real.mean(0)).abs().mean()
                        moment = moment + (torch.sqrt(generated.var(0, unbiased=False) + 1e-6) -
                                           torch.sqrt(real.var(0, unbiased=False) + 1e-6)).abs().mean()
                        loss = adversarial + 100 * torch.sqrt(supervise(h) + 1e-8) + 100 * moment
                        losses.update(generator_loss=float(loss.item()),
                                      adversarial_loss=float(adversarial.item()),
                                      moment_loss=float(moment.item()))
                        losses['generator_grad_norm'] = optimize(og, loss, gs)
                        counts['generator_updates'] += 1
                        base.active(model, {'E', 'R'})
                        h = model.E(real)
                        embedder_loss = 10 * torch.sqrt(F.mse_loss(model.R(h), real) + 1e-8) + .1 * supervise(h)
                        losses['embedder_grad_norm'] = optimize(oe, embedder_loss, er)
                        counts['embedder_updates'] += 1
                    base.active(model, {'D'})
                    with torch.no_grad():
                        h = model.E(real)
                        eh = model.G(torch.rand_like(real))
                        hh = model.S(eh)
                    discriminator_loss = bce(model.D(h), torch.ones_like(h[:, :, :1]))
                    discriminator_loss = discriminator_loss + bce(model.D(hh), torch.zeros_like(hh[:, :, :1]))
                    discriminator_loss = discriminator_loss + bce(model.D(eh), torch.zeros_like(eh[:, :, :1]))
                    losses['discriminator_loss'] = float(discriminator_loss.item())
                    if discriminator_loss.item() > .15:
                        losses['discriminator_grad_norm'] = optimize(od, discriminator_loss, model.D.parameters())
                        counts['discriminator_updates'] += 1
                if step == 1 or step % args.diagnostic_every == 0 or step == steps:
                    snapshot(phase, step, steps, losses)
                if step == 1 or step % args.log_every == 0 or step == steps:
                    elapsed = time.monotonic() - phase_start
                    eta = elapsed / step * (steps - step) / 60
                    print(f'{scheme.name} | {tag} {phase} {step}/{steps} '
                          f'Loss={loss.item():.4f} G-Updates={counts["generator_updates"]} '
                          f'D-Updates={counts["discriminator_updates"]} '
                          f'Phase-ETA={eta:.1f} min | {clock.minutes():.1f} min', flush=True)
        base.active(model, set())
        return model.eval()

    return fit_timegan


def install_scheme(base, scheme, *, runner_sha256, base_sha256, session_id, folder):
    original_fit, original_fields = base.fit_timegan, base.candidate_fields

    def fields(candidate):
        row = original_fields(candidate)
        row.update(training_variant=scheme.name, planned_gan_cycles=scheme.gan_cycles,
                   generator_updates_per_cycle=scheme.generator_updates_per_cycle,
                   training_extension_sha256=runner_sha256, training_base_sha256=base_sha256)
        return row

    base.candidate_fields = fields
    base.fit_timegan = make_timegan_fit(base, scheme, session_id=session_id,
                                      live_path=Path(folder)/'gan_training_live.jsonl')
    return original_fit, original_fields


def restore_scheme(base, originals):
    base.fit_timegan, base.candidate_fields = originals


def assert_shared_splits(out, schemes):
    """Compare every available full patient/inner split manifest, not just sizes."""
    available = []
    for scheme in schemes:
        path = Path(out) / scheme.name / 'patient_split_manifest.json'
        if not path.is_file():
            continue
        manifest = read_json(path)
        identity = dict(dataset_sha256=manifest['dataset_sha256'], units=manifest['units'],
                        n_unique_patients=manifest['n_unique_patients'],
                        class_counts=manifest['class_counts'])
        available.append((scheme.name, identity))
    if available and any(identity != available[0][1] for _, identity in available[1:]):
        raise ValueError('Die Trainingsvarianten besitzen unterschiedliche Patientenaufteilungen. '
                         'Gepaarte Ergebnisse werden deshalb nicht ausgegeben.')
    return (digest(available[0][1]) if available else None,
            available[0][1]['n_unique_patients'] if available else None)


def collect_comparison(out, schemes):
    """Read only committed fold tables produced by V1.6's normal exporter."""
    out = Path(out)
    split_hash, n_patients = assert_shared_splits(out, schemes)
    summaries, fold_tables, repeats, diagnostics, trajectory_rows = [], {}, {}, [], []
    all_folds, all_seeds, selected_rows = [], [], []
    for scheme in schemes:
        folder = out / scheme.name
        for row in read_csv(folder / 'summary_patient_pooled.csv'):
            summaries.append(dict(training_variant=scheme.name, **row))
        folded = read_csv(folder / 'outer_fold_metrics.csv')
        fold_tables[scheme.name] = {(int(row['repeat']), int(row['fold']), row['method']): row
                                     for row in folded}
        all_folds.extend(dict(training_variant=scheme.name, **row) for row in folded)
        all_seeds.extend(dict(training_variant=scheme.name, **row)
                         for row in read_csv(folder / 'outer_seed_metrics.csv'))
        selected_rows.extend(dict(row, training_variant=scheme.name)
                             for row in read_csv(folder / 'selected_configurations.csv'))
        repeats[scheme.name] = {(int(row['repeat']), row['method']): row
            for row in read_csv(folder / 'repeat_patient_pooled.csv') if yes(row['repeat_complete'])}
        qc = read_csv(folder / 'generation_diagnostics.csv')
        for stage in ('inner', 'outer_train'):
            values = [number(row['std_ratio']) for row in qc
                      if row['stage'] == stage and number(row.get('std_ratio')) is not None]
            if values:
                import statistics
                diagnostics.append(dict(training_variant=scheme.name, stage=stage,
                    n_generator_fits=len(values), mean_std_ratio=statistics.mean(values),
                    median_std_ratio=statistics.median(values),
                    n_std_ratio_below_0_1=sum(value < .1 for value in values),
                    fraction_std_ratio_below_0_1=sum(value < .1 for value in values) / len(values),
                    n_std_ratio_below_0_01=sum(value < .01 for value in values),
                    reference='original minority training sequences; terminal generated samples'))
        training = read_csv(folder / 'timegan_training_history.csv')
        curves = {}
        id_columns = ('repeat', 'fold', 'inner_fold', 'seed_index', 'stage', 'noise_copies', 'gan_seed')
        for row in training:
            if row['phase'] == 'GAN':
                curves.setdefault(tuple(row[key] for key in id_columns), []).append(row)
        by_stage = {}
        for key, curve in curves.items():
            curve.sort(key=lambda row: int(row['phase_step']))
            last = curve[-1]
            if int(last['phase_step']) != int(last['phase_total']):
                continue
            ratio = number(last['std_ratio'])
            if ratio is None:
                continue
            prior = [number(row['std_ratio']) for row in curve[:-1]]
            prior = [value for value in prior if value is not None]
            category = ('final_at_least_10pct' if ratio >= .1 else
                        'previously_at_least_10pct_then_below' if prior and max(prior) >= .1 else
                        'never_reached_10pct_at_logged_checkpoints')
            by_stage.setdefault((last['stage'], category), []).append(last)
        for (stage, category), rows in by_stage.items():
            import statistics
            trajectory_rows.append(dict(training_variant=scheme.name, stage=stage,
                trajectory=category, n_generator_fits=len(rows),
                median_generator_updates=statistics.median(float(row['generator_updates']) for row in rows),
                median_discriminator_updates=statistics.median(float(row['discriminator_updates']) for row in rows),
                reference='GAN-training snapshot reference; fixed diagnostic noise; logged checkpoints only'))

    reference_name = schemes[0].name
    fold_differences, repeat_differences, consistency = [], [], []
    reference_folds, reference_repeats = fold_tables.get(reference_name, {}), repeats.get(reference_name, {})
    for scheme in schemes[1:]:
        current_folds = fold_tables.get(scheme.name, {})
        for key in sorted(reference_folds.keys() & current_folds.keys()):
            repeat, fold, method = key
            a, b = current_folds[key], reference_folds[key]
            differences = {'delta_' + metric: float(a[metric]) - float(b[metric]) for metric in METRICS}
            if method == 'timegan_selected':
                fold_differences.append(dict(training_variant=scheme.name,
                    reference_variant=reference_name, repeat=repeat, fold=fold, method=method,
                    **differences))
            else:
                maximum = max(abs(value) for value in differences.values())
                same_choice = all(str(a[field]) == str(b[field]) for field in
                                  ('mode', 'hidden', 'noise_copies', 'steps', 'threshold'))
                consistency.append(dict(training_variant=scheme.name, reference_variant=reference_name,
                    repeat=repeat, fold=fold, method=method, max_absolute_metric_difference=maximum,
                    identical_selected_settings=same_choice,
                    consistent_within_tolerance=maximum <= 1e-7 and same_choice))
        current_repeats = repeats.get(scheme.name, {})
        for key in sorted(reference_repeats.keys() & current_repeats.keys()):
            repeat, method = key
            if method != 'timegan_selected':
                continue
            a, b = current_repeats[key], reference_repeats[key]
            repeat_differences.append(dict(training_variant=scheme.name,
                reference_variant=reference_name, repeat=repeat, method=method,
                n_unique_patients=n_patients,
                **{'delta_' + metric: float(a[metric]) - float(b[metric]) for metric in METRICS}))
    for filename, rows in (
        ('training_variant_summary.csv', summaries),
        ('training_variant_outer_fold_metrics.csv', all_folds),
        ('training_variant_outer_seed_metrics.csv', all_seeds),
        ('training_variant_selected_configurations.csv', selected_rows),
        ('training_variant_fold_differences.csv', fold_differences),
        ('training_variant_repeat_differences.csv', repeat_differences),
        ('generator_training_summary.csv', diagnostics),
        ('generator_trajectory_summary.csv', trajectory_rows),
        ('baseline_consistency.csv', consistency)):
        write_csv(out / filename, rows)
    states = {}
    for scheme in schemes:
        path = out / scheme.name / 'run_status.json'
        states[scheme.name] = read_json(path) if path.is_file() else {'status': 'not_started'}
    return dict(states=states, summaries=summaries,
                split_manifest_sha256=split_hash, n_unique_patients=n_patients,
                baseline_consistency_passed=(all(yes(row['consistent_within_tolerance']) for row in consistency)
                                             if consistency else None))


def base_arguments(args, scheme, folder, hours, resume):
    result = [str(args.data_dir), '--device', args.device, '--output-dir', str(folder),
              '--weights-dir', str(args.weights_dir), '--repeats', str(args.repeats),
              '--folds', str(args.folds), '--inner-folds', str(args.inner_folds),
              '--training-seeds', str(args.training_seeds), '--seed', str(args.seed),
              '--hours', f'{hours:.12g}', '--fold-limit', str(args.fold_limit),
              '--hidden', args.hidden, '--noise-copies', args.noise_copies,
              '--noise-sigma', str(args.noise_sigma), '--gan-ratios', args.gan_ratios,
              '--random-ratios', args.random_ratios,
              '--layers', str(args.layers), '--batch-size', str(args.batch_size),
              '--contact-microbatch', str(args.contact_microbatch),
              '--ae-steps', str(args.ae_steps), '--supervisor-steps', str(args.supervisor_steps),
              '--gan-steps', str(scheme.gan_cycles), '--classifier-steps', str(args.classifier_steps),
              '--eval-every', str(args.classifier_steps), '--diagnostic-every', str(args.diagnostic_every),
              '--log-every', str(args.log_every), '--gan-lr', str(args.gan_lr),
              '--classifier-lr', str(args.classifier_lr), '--threads', str(args.threads),
              '--bootstrap-samples', str(args.bootstrap_samples),
              '--force-threshold', str(args.force_threshold),
              '--contact-duration-tolerance', str(args.contact_duration_tolerance),
              '--contact-zero-tolerance', str(args.contact_zero_tolerance),
              '--min-si-pairs-per-subject', str(args.min_si_pairs_per_subject)]
    for flag, enabled in (('--resume', resume), ('--save-synthetic', args.save_synthetic),
                          ('--save-models', args.save_models), ('--show-import-log', args.show_import_log)):
        if enabled:
            result.append(flag)
    return result


def prepare_manifest(args, schemes):
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    ignore = {'out', 'resume', 'hours', 'fold_limit', 'variants', 'show_import_log',
              'save_models', 'save_synthetic', 'bootstrap_samples', 'log_every',
              'contact_microbatch', 'threads'}
    identity = {key: value for key, value in configuration.items() if key not in ignore}
    identity.update(version=VERSION, extension_sha256=sha256_file(__file__),
                    base_sha256=sha256_file(args.base_script),
                    schemes=[asdict(scheme) for scheme in schemes],
                    classifier_checkpoint='fixed updates; inner threshold selection only')
    fingerprint = digest(identity)
    path = args.out / 'run_manifest.json'
    if args.out.exists() and not args.out.is_dir():
        raise ValueError('--output-dir muss ein Ordner sein.')
    nonempty = args.out.is_dir() and any(args.out.iterdir())
    if nonempty and not args.resume:
        raise ValueError('Ausgabeordner ist belegt. Zum Fortsetzen --resume verwenden oder neuen --output-dir waehlen.')
    if args.resume:
        if not path.is_file():
            raise ValueError('Wiederaufnahme benoetigt einen V1.7-Vergleichsordner mit run_manifest.json.')
        previous = read_json(path)
        if previous.get('fingerprint') != fingerprint:
            raise ValueError('Wiederaufnahme abgelehnt: Basisdatei, Erweiterung, Suchraum, Seeds oder Trainingsparameter wurden geaendert.')
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_json(path, dict(fingerprint=fingerprint, identity=identity,
                              initial_arguments=configuration))
    return identity, fingerprint


def run_comparison(base, args):
    schemes = build_schemes(args.reference_cycles)
    identity, fingerprint = prepare_manifest(args, schemes)
    started, deadline = time.monotonic(), time.monotonic() + args.hours * 3600
    session_id = uuid.uuid4().hex
    status, result_code = 'running', 0
    requested = {name.strip() for name in args.variants.split(',')}
    state = collect_comparison(args.out, schemes)

    def save_status():
        atomic_json(args.out / 'run_status.json', dict(status=status, version=VERSION,
            fingerprint=fingerprint, session_id=session_id,
            runtime_minutes_this_session=(time.monotonic() - started) / 60,
            planned_variants=[scheme.name for scheme in schemes],
            requested_variants=sorted(requested), variant_states=state['states'],
            completed_variants=sum(item.get('status') == 'complete' for item in state['states'].values()),
            planned_outer_folds_per_variant=args.repeats * args.folds,
            completed_outer_folds=sum(item.get('completed_outer_folds', 0) for item in state['states'].values()),
            n_unique_patients=state['n_unique_patients'],
            split_manifest_sha256=state['split_manifest_sha256'],
            baseline_consistency_passed=state['baseline_consistency_passed'],
            classifier_updates_fixed=args.classifier_steps,
            interpretation='paired predefined training conditions; primary metrics per complete repeat; dependent folds/seeds'))

    print(f'Vergleich: {len(schemes)} Bedingungen; je {args.repeats} Wiederholung(en) '
          f'x {args.folds} aeussere x {args.inner_folds} innere Patientenfolds '
          f'x {args.training_seeds} Seeds. Klassifikator: exakt {args.classifier_steps} Updates.', flush=True)
    print(f'Gemeinsames Zeitbudget: {args.hours:g} h. Basis: {args.base_script.name}', flush=True)
    try:
        save_status()
        for scheme in schemes:
            if scheme.name not in requested:
                continue
            remaining = (deadline - time.monotonic()) / 3600
            if remaining <= 0:
                status, result_code = 'budget_exceeded', 2
                break
            folder = args.out / scheme.name
            resume_variant = args.resume and (folder / 'configuration.json').is_file()
            originals = install_scheme(base, scheme, runner_sha256=identity['extension_sha256'],
                base_sha256=identity['base_sha256'], session_id=session_id, folder=folder)
            try:
                print(f'\n=== {scheme.name}: {scheme.gan_cycles} Zyklen, '
                      f'{scheme.generator_updates_per_cycle} G/E-Updates pro Zyklus ===', flush=True)
                code = base.main(base_arguments(args, scheme, folder, remaining, resume_variant))
            finally:
                restore_scheme(base, originals)
                gc.collect()
                if base.torch.cuda.is_available():
                    base.torch.cuda.empty_cache()
            state = collect_comparison(args.out, schemes)
            save_status()
            if code:
                status = state['states'][scheme.name].get('status', 'failed')
                result_code = int(code)
                break
        else:
            status = ('complete' if all(item.get('status') == 'complete' for item in state['states'].values())
                      else 'partial_requested')
        if status == 'running':
            status = 'partial_requested'
    except KeyboardInterrupt:
        status, result_code = 'interrupted', 2
        print('Unterbrochen. Vollstaendige aeussere Folds bleiben gespeichert.', flush=True)
    except BaseException:
        status, result_code = 'failed', 2
        raise
    finally:
        state = collect_comparison(args.out, schemes)
        save_status()
        print(f'\nGesamtstatus={status}; Laufzeit dieser Sitzung={(time.monotonic()-started)/60:.1f} min', flush=True)
        for row in state['summaries']:
            if row['method'] == 'timegan_selected':
                print(f'{row["training_variant"]:22s} BA={float(row["balanced_accuracy"]):.3f} '
                      f'Accuracy={float(row["accuracy"]):.3f} Kappa={float(row["cohen_kappa"]):.3f} '
                      f'({row["n_completed_repeats"]} vollstaendige Wiederholung(en))', flush=True)
        if state['baseline_consistency_passed'] is False:
            print('Die Kontrollverfahren unterscheiden sich zwischen Bedingungen. '
                  'baseline_consistency.csv vor Interpretation pruefen.', flush=True)
            if result_code == 0:
                result_code = 3
        print('Ausgabe:', args.out.resolve(), flush=True)
    return result_code


def make_technical_base():
    """Minimal CPU/CUDA harness for the new GAN core, not a CSV fallback."""
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as error:
        raise RuntimeError('Fuer den technischen GAN-Test werden numpy und PyTorch benoetigt.') from error

    class SeqNet(nn.Module):
        def __init__(self, inputs, hidden, outputs, layers, sigmoid=True):
            super().__init__()
            self.rnn = nn.GRU(inputs, hidden, layers, batch_first=True)
            self.proj = nn.Linear(hidden, outputs)
            self.sigmoid = sigmoid

        def forward(self, values):
            result = self.proj(self.rnn(values)[0])
            return result.sigmoid() if self.sigmoid else result

    class TimeGAN(nn.Module):
        def __init__(self, channels, hidden, layers):
            super().__init__()
            self.channels, self.hidden, self.layers = channels, hidden, layers
            self.E = SeqNet(channels, hidden, hidden, layers)
            self.R = SeqNet(hidden, hidden, channels, layers)
            self.G = SeqNet(channels, hidden, hidden, layers)
            self.S = SeqNet(hidden, hidden, hidden, max(1, layers - 1))
            self.D = SeqNet(hidden, hidden, 1, layers, sigmoid=False)

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def row_probabilities(groups):
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        weights = 1. / counts[inverse]
        return weights / weights.sum()

    def active(model, names):
        for name, block in model.named_children():
            for parameter in block.parameters():
                parameter.requires_grad_(name in names)

    @torch.no_grad()
    def generate(model, n, length, args, device, constant, seed):
        model.eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        results = []
        for start in range(0, n, args.batch_size):
            z = torch.rand(min(args.batch_size, n-start), length, model.channels,
                           device=device, generator=generator)
            results.append(model.R(model.S(model.G(z))).cpu().numpy())
        fake = np.concatenate(results).astype(np.float32)
        fake[..., constant] = 0
        if not np.isfinite(fake).all():
            raise FloatingPointError('Nichtendliche synthetische Werte.')
        return fake

    def diversity_report(fake, real):
        f, r = np.asarray(fake[:128], np.float64), np.asarray(real[:512], np.float64)
        ff, rr = f.reshape(len(f), -1), r.reshape(len(r), -1)
        distances = np.sqrt(np.maximum(0., np.mean(ff*ff, axis=1)[:, None] +
            np.mean(rr*rr, axis=1)[None] - 2*(ff @ rr.T)/ff.shape[1]))
        generated_std, real_std = float(f.std(axis=0).mean()), float(r.std(axis=0).mean())
        return dict(nearest_train_rmse_mean=float(distances.min(axis=1).mean()),
                    generated_std=generated_std, real_std=real_std,
                    std_ratio=generated_std/real_std if real_std > 1e-12 else None)

    return SimpleNamespace(np=np, torch=torch, F=F, TimeGAN=TimeGAN,
        seed_all=seed_all, row_probabilities=row_probabilities, active=active,
        generate=generate, diversity_report=diversity_report)


class TechnicalClock:
    def __init__(self, seconds=120):
        self.start = time.monotonic()
        self.end = self.start + seconds

    def check(self):
        if time.monotonic() >= self.end:
            raise TimeoutError('Technischer Test: Zeitbudget erreicht.')

    def minutes(self):
        return (time.monotonic() - self.start) / 60


def technical_self_test(args):
    base = make_technical_base()
    np, torch = base.np, base.torch
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA fehlt in diesem Python. Fuer CPU-Test --device cpu angeben.')
    device = torch.device(args.device)
    schemes = build_schemes(2)
    rng = np.random.default_rng(23)
    x = rng.uniform(.1, .9, (12, 101, 7)).astype(np.float32)
    x[..., 6] = 0
    groups = np.repeat(np.arange(4), 3)
    options = SimpleNamespace(layers=2, gan_lr=.001, ae_steps=2, supervisor_steps=2,
        batch_size=4, diagnostic_every=1, log_every=1000)
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise ValueError('Technischer Test benoetigt einen leeren Ausgabeordner.')
    out.mkdir(parents=True, exist_ok=True)
    history, result = [], []
    for scheme in schemes:
        rows = []
        fit = make_timegan_fit(base, scheme, session_id='technical_test')
        model = fit(x, groups, 4, options, device, TechnicalClock(), 42, 'SELBSTTEST',
                    history_rows=rows, context={'stage': 'technical_test'},
                    constant=np.array([False]*6+[True]))
        final = rows[-1]
        expected_g = scheme.gan_cycles * scheme.generator_updates_per_cycle
        assert final['generator_updates'] == expected_g
        assert final['embedder_updates'] == options.ae_steps + expected_g
        assert final['supervisor_pretrain_updates'] == options.supervisor_steps
        assert 0 <= final['discriminator_updates'] <= scheme.gan_cycles
        assert final['phase_step'] == scheme.gan_cycles
        fake = base.generate(model, 8, 101, options, device, np.array([False]*6+[True]), 71)
        assert fake.shape == (8, 101, 7) and np.isfinite(fake).all()
        assert (fake[..., 6] == 0).all()
        assert all(np.isfinite(final[key]) for key in ('reconstruction_mse', 'supervisor_mse',
            'real_embedding_std', 'fake_embedding_std', 'fake_supervisor_std',
            'noise_sensitivity_l1', 'generator_grad_norm'))
        history.extend(rows)
        result.append(dict(training_variant=scheme.name, planned_gan_cycles=scheme.gan_cycles,
            generator_updates=final['generator_updates'], embedder_updates=final['embedder_updates'],
            discriminator_updates=final['discriminator_updates'], result='passed'))
        del model
    write_csv(out/'technical_training_history.csv', history)
    write_csv(out/'technical_update_counts.csv', result)
    atomic_json(out/'run_status.json', dict(status='complete', technical_test=True,
        tested='new GAN training core, update counts and diagnostics only; no original CSV pipeline',
        torch_version=str(torch.__version__), device=str(device), variants=3))
    print('SELBSTTEST BESTANDEN: drei Trainingsvarianten, Optimierungszaehler, Gradienten und Generierung.', flush=True)
    print('Dieser Test bewertet weder CSV-Import noch Klassifikationsguete.', flush=True)
    print('Ausgabe:', out.resolve(), flush=True)
    return 0


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', nargs='?', type=Path, default=Path('CSV_Export'))
    parser.add_argument('--base-script', type=Path, help='Vorhandene V1.6-Datei; Standard: neben dieser Erweiterung.')
    parser.add_argument('--output-dir', '--out', dest='out', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--weights-dir', type=Path, default=Path('pretrained_weights_v36'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--hours', type=float, default=24., help='Gesamtbudget fuer alle drei Bedingungen, in Stunden.')
    parser.add_argument('--variants', default=','.join(SCHEME_NAMES), help='Kommagetrennte Bedingungsnamen; Standard alle drei.')
    parser.add_argument('--reference-cycles', type=int, default=1500,
                        help='Referenzzyklen; langer Lauf verwendet das Doppelte.')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--folds', '--outer-folds', type=int, default=3)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--training-seeds', type=int, default=3)
    parser.add_argument('--fold-limit', type=int, default=999999,
                        help='Hoechstens N noch fehlende aeussere Folds je Bedingung/Sitzung.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden', default='16')
    parser.add_argument('--noise-copies', default='0,3,4,5')
    parser.add_argument('--noise-sigma', type=float, default=.02)
    parser.add_argument('--gan-ratios', '--gan-ratio', default='0.5')
    parser.add_argument('--random-ratios', '--random-ratio', default='0.5')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--contact-microbatch', type=int, default=16)
    parser.add_argument('--ae-steps', type=int, default=500)
    parser.add_argument('--supervisor-steps', type=int, default=500)
    parser.add_argument('--classifier-steps', type=int, default=300)
    parser.add_argument('--diagnostic-every', type=int, default=250)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--gan-lr', type=float, default=.001)
    parser.add_argument('--classifier-lr', type=float, default=.001)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--save-synthetic', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--show-import-log', action='store_true')
    parser.add_argument('--force-threshold', type=float, default=0.)
    parser.add_argument('--contact-duration-tolerance', type=float, default=.25)
    parser.add_argument('--contact-zero-tolerance', type=float, default=0.)
    parser.add_argument('--min-si-pairs-per-subject', type=int, default=3)
    parser.add_argument('--self-test', action='store_true', help='Nur technischen GAN-Kern mit kuenstlichen Sequenzen pruefen.')
    args = parser.parse_args(argv)
    positive = [args.hours, args.reference_cycles, args.repeats, args.training_seeds, args.fold_limit,
        args.batch_size, args.contact_microbatch, args.ae_steps, args.supervisor_steps,
        args.classifier_steps, args.diagnostic_every, args.log_every, args.gan_lr,
        args.classifier_lr, args.threads, args.noise_sigma, args.min_si_pairs_per_subject]
    if not all(math.isfinite(value) for value in positive) or min(positive) <= 0:
        parser.error('Budgets und Trainingsparameter muessen endlich und positiv sein.')
    if args.folds < 2 or args.inner_folds < 2 or args.layers < 2:
        parser.error('folds, inner-folds und layers muessen >=2 sein.')
    if not 0 <= args.seed < 2**32 or args.bootstrap_samples < 0:
        parser.error('seed in 0..2**32-1 und bootstrap-samples >=0 erforderlich.')
    qc = (args.force_threshold, args.contact_duration_tolerance, args.contact_zero_tolerance)
    if not all(math.isfinite(value) and value >= 0 for value in qc):
        parser.error('QC-Werte muessen endlich und nichtnegativ sein.')
    requested = {name.strip() for name in args.variants.split(',')}
    if not requested or requested - set(SCHEME_NAMES):
        parser.error('--variants erwartet: ' + ','.join(SCHEME_NAMES))
    if args.out is None:
        stem = 'results_timegan_training_selftest' if args.self_test else 'results_timegan_training_v17'
        args.out = Path(stem + '_' + time.strftime('%Y%m%d_%H%M%S'))
    args.out = args.out.resolve()
    args.data_dir = args.data_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    if args.base_script is None:
        adjacent = Path(__file__).resolve().with_name(BASE_FILENAME)
        local = Path.cwd()/BASE_FILENAME
        args.base_script = adjacent if adjacent.is_file() else local
    args.base_script = args.base_script.resolve()
    return args


def main(argv=None):
    args = parse_arguments(argv)
    if args.self_test:
        return technical_self_test(args)
    base = load_base(args.base_script)
    return run_comparison(base, args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, PermissionError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
