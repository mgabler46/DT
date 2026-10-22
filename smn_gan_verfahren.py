#!/usr/bin/env python3
r"""SI7 TimeGAN V2.1: Paper-Ablationen, 5000/6000/7000, GTX 1070, Nested-CV.

Neben den unveraenderten Dateien timegan_si7_nested_gtx1070_v1_6.py und
timegan_si7_training_compare_gtx1070_v1_7.py ablegen. Original-CSV-Import,
SI7, Kontakt-QC, Patiententrennung und ConvNeXt-Transformer bleiben erhalten.

  python .\timegan_si7_paper_compare_gtx1070_v2_1.py .\CSV_Export --device cuda --hours 24
  python .\timegan_si7_paper_compare_gtx1070_v2_1.py .\CSV_Export --device cuda --hours 24 --resume
  python .\timegan_si7_paper_compare_gtx1070_v2_1.py --plan-only
  python .\timegan_si7_paper_compare_gtx1070_v2_1.py --self-test --device cpu --cycles 5000

Standard: 1 Wiederholung x 3 aeussere x 3 innere Folds x 6 Seed-Slots,
24 Stunden SITZUNGSBUDGET; CUDA float32; feste 300 Klassifikator-Updates. Ein Zeitbudget
garantiert keinen vollstaendigen Lauf; --resume setzt denselben Lauf fort.
GAN-Kopien 0,3 (bewusst kleiner als der bisherige Suchraum 0,3,4,5).
Mit --noise-copies 0,3,4,5 kann der bisherige Suchraum verwendet werden.

18 vorab festgelegte Bedingungen, jede bei 5000, 6000 und 7000 GAN-Zyklen:
  legacy_1g        Bisheriges V2.0-Schema, 1 G/S- und 1 E/R-Update.
  timegan_2g       Autoren-Ablauf: 2 G/S- und 2 E/R-Updates, frische D-Minibatch.
  instance_noise   timegan_2g + unabhaengiges Gaussrauschen auf echten und
                   kuenstlichen LATENTEN D-Eingaengen, bei G- UND D-Schritten.
  mode_seeking     timegan_2g - lambda * L1(RSG(z1),RSG(z2))/L1(z1,z2).
  pacgan           timegan_2g + PacGAN2: zwei unabhaengige latente Sequenzen
                   entlang der Merkmale verketten; EIN Urteil pro Paket/Zeit.
  ttur             timegan_2g + getrennte G/S- und D-Lernraten.
Je Zyklus maximal ein D-Update, wie bisher nur bei D-Loss > 0.15.
Das timegan_2g-Schema verwendet jetzt auch die frische D-Minibatch des Autorencodes.
Die bisherige kompakte PyTorch-Architektur bleibt erhalten; kein Anspruch auf
vollstaendige TensorFlow-Reproduktion. Alle vier Paper-Erweiterungen werden
EINZELN gegen timegan_2g derselben Zyklenzahl verglichen, nicht miteinander vermischt.

Voreinstellungen sind vorab festgelegte SI7-Experimente, keine behaupteten
Optimalwerte fuer Gangdaten: Instance-Sigma 0.05 -> 0.0 linear bis Zyklus
7000; dieser GEMEINSAME Horizont gilt auch fuer 5000/6000, sodass Fortsetzungen
dieselben Trainingspraefixe besitzen. Mode-Seeking lambda=1 und L1-Distanzen
nach Mao et al.; Vorzeichen fuer einen MINIMIERENDEN Optimierer negativ.
Die vollstaendige Kette G -> S -> R bleibt differenzierbar. Konstante
Trainingskanaele sind von der Diversitaetsbelohnung ausgeschlossen.
TTUR: G/S=0.0005, D=0.001; E/R und Supervisor-Vortraining behalten gan-lr=0.001.
Anpassbar mit --instance-sigma-start/end, --instance-anneal-cycles,
--mode-seeking-lambda, --ttur-g-lr und --ttur-d-lr; neue Werte = neuer Lauf.
Dies sind Uebertragungen auf SI7-TimeGAN, keine Wirksamkeitsgarantien der Papers.

Originalquellen:
https://github.com/jsyoon0823/TimeGAN/blob/master/timegan.py
https://proceedings.mlr.press/v80/mescheder18a.html
https://arxiv.org/abs/1903.05628
https://proceedings.neurips.cc/paper/2018/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html
https://papers.nips.cc/paper_files/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html

Alle sechs Seeds werden jetzt AUCH innen trainiert. Kandidat und Schwelle
werden auf den gemittelten inneren Sechs-Seed-OOF-Wahrscheinlichkeiten nach
Patienten-Kappa gewaehlt. Dieselbe Ensemblegroesse und dieselbe Schwelle
werden aussen angewandt. Alle Varianten nutzen identische Patientenfolds,
Seed-Slots und Kontrollverfahren. Seeds und CV-Wiederholungen sind abhaengig.
Ein Gewinner nach aeusserem Kappa ist explorativ, keine neue innere Auswahl.

Wiederaufnahme spart Arbeit innerhalb unvollstaendiger aeusserer Folds:
* GAN-Zustaende MIT Optimierern und Zufallszustaenden alle 250 GAN-Zyklen
  sowie an jedem Varianten-Endpunkt; Fortsetzung 5000 -> 6000 -> 7000.
* abgeschlossene Klassifikator-Fits einschliesslich innerer OOF-Prognosen;
* gemeinsame Kontroll-Fits werden zwischen Varianten wiederverwendet.
Unterbrochene Klassifikator-Fits beginnen erneut (max. 300 Updates).
Nur vollstaendig abgeschlossene aeussere Folds gehen in Vergleichswerte ein.
Zuerst werden die 5000er-Bedingungen mit vollstaendiger CV abgearbeitet,
danach 6000, dann 7000. Der groessere Umfang kann mehrere 24h-Sitzungen
benoetigen. --resume setzt exakt dieselbe Konfiguration fort; kein neuer
Seed wird anhand aeusserer Ergebnisse ausgesucht. --families und --cycles
koennen fuer einen NEUEN Lauf den vorab festgelegten Umfang eingrenzen.
Technischer --self-test verwendet ausdruecklich kuenstliche Daten, kurze
2/3/4-Zyklen und einen technischen Backbone; keine Aussage zur Modellguete.

BA, Accuracy und Cohen-Kappa erscheinen innen, aussen und in der Zusammen-
fassung. timegan_v21_aggregate.zip enthaelt ausschliesslich explizit erlaubte
aggregierte Tabellen, Konfigurationen ohne Patientennamen und Beschreibungen.
Patientenprognosen, Patientensplit-Hashes, Modelle und Cache bleiben lokal.
paper_ablation_*.csv vergleichen jede Erweiterung mit timegan_2g bei
GLEICHER Zyklenzahl. timegan_2g wird mit legacy_1g verglichen. Nur vollstaendige
Wiederholungen gehen in paper_ablation_summary.csv ein. Mehr G/E-Updates und
zusaetzliche Verluste bedeuten unterschiedliche Rechenkosten; keine Gleichheit
des Rechenbudgets behauptet. Vorherige V2.0-Ergebnisse werden nicht ueberschrieben.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

VERSION = '2.1.0'
BASE_FILENAME = 'timegan_si7_nested_gtx1070_v1_6.py'
COMPARE_FILENAME = 'timegan_si7_training_compare_gtx1070_v1_7.py'
FAMILIES = ('legacy_1g', 'timegan_2g', 'instance_noise', 'mode_seeking', 'pacgan', 'ttur')
CYCLES = (5000, 6000, 7000)
PAPER_SOURCES = {
    'timegan_2g': 'https://github.com/jsyoon0823/TimeGAN/blob/master/timegan.py',
    'instance_noise': 'https://proceedings.mlr.press/v80/mescheder18a.html',
    'mode_seeking': 'https://arxiv.org/abs/1903.05628',
    'pacgan': 'https://proceedings.neurips.cc/paper/2018/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html',
    'ttur': 'https://papers.nips.cc/paper_files/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html'}


@dataclass(frozen=True)
class PaperScheme:
    name: str
    gan_cycles: int
    generator_updates_per_cycle: int
    family: str
    production_cycles: int
    fresh_discriminator_batch: bool
    instance_sigma_start: float
    instance_sigma_end: float
    instance_anneal_cycles: int
    mode_seeking_lambda: float
    packing: int
    generator_lr: float
    discriminator_lr: float


def build_schemes(args):
    families = [v.strip() for v in args.families.split(',')]
    cycles = sorted({int(v) for v in args.cycles.split(',')})
    result = []
    for total in cycles:
        for family in families:
            test_cycles = {5000: 2, 6000: 3, 7000: 4}[total] if args.self_test else total
            result.append(PaperScheme(name=f'{family}_{total}', gan_cycles=test_cycles,
                generator_updates_per_cycle=1 if family == 'legacy_1g' else 2,
                family=family, production_cycles=total,
                fresh_discriminator_batch=family != 'legacy_1g',
                instance_sigma_start=args.instance_sigma_start if family == 'instance_noise' else 0.,
                instance_sigma_end=args.instance_sigma_end if family == 'instance_noise' else 0.,
                instance_anneal_cycles=args.instance_anneal_cycles,
                mode_seeking_lambda=args.mode_seeking_lambda if family == 'mode_seeking' else 0.,
                packing=2 if family == 'pacgan' else 1,
                generator_lr=args.ttur_g_lr if family == 'ttur' else args.gan_lr,
                discriminator_lr=args.ttur_d_lr if family == 'ttur' else args.gan_lr))
    return result


def instance_sigma(scheme, cycle):
    """One fixed horizon for all durations: extending a fit preserves its prefix."""
    fraction = min(1., max(0., (cycle - 1) / max(1, scheme.instance_anneal_cycles - 1)))
    return scheme.instance_sigma_start + fraction * (scheme.instance_sigma_end - scheme.instance_sigma_start)


def discriminator_input(values, packing=1, sigma=0., generator=None):
    """Instance noise on latent values; PacGAN concatenates independent samples.

    [B,T,H] -> [B/packing,T,packing*H]. No real/fake mixing, no temporal
    concatenation, and no loss of members through averaging. Input batches
    are independently drawn from patient-weighted training distributions.
    """
    if values.ndim != 3 or packing < 1 or values.shape[0] % packing:
        raise ValueError('PacGAN benoetigt [B,T,H] und durch packing teilbare Batchgroesse.')
    if sigma:
        import torch
        values = values + sigma * torch.randn(values.shape, dtype=values.dtype,
            device=values.device, generator=generator)
    b, t, h = values.shape
    return values.reshape(b // packing, packing, t, h).permute(0, 2, 1, 3).reshape(b // packing, t, packing * h)


def mode_seeking_ratio(output_a, output_b, noise_a, noise_b, active_channels):
    """Mao et al. Eq.1, L1 distances; maximize ratio by subtracting it in SGD.

    Outputs are R(S(G(z))) in fold-normalized SI7 units, not just latent G(z).
    Constant input channels are excluded so diversity cannot be invented there.
    """
    import torch
    active = torch.as_tensor(active_channels, dtype=torch.bool, device=output_a.device)
    if active.numel() != output_a.shape[-1]:
        raise ValueError('Mode-Seeking-Kanalmaske besitzt eine falsche Laenge.')
    if not active.any().item():
        return (output_a.sum() + output_b.sum()) * 0.
    distance_x = (output_a[..., active] - output_b[..., active]).abs().mean()
    distance_z = (noise_a - noise_b).abs().mean().clamp_min(1e-6)
    return distance_x / distance_z


def load_compare(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Die vorhandene V1.7-Datei fehlt: {path}')
    spec = importlib.util.spec_from_file_location('_si7_overnight_v17', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not str(module.VERSION).startswith('1.7.'):
        raise ValueError('Erwartet wird die unveraenderte Vergleichspipeline V1.7.')
    return module


def array_digest(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        value = array if hasattr(array, 'dtype') else None
        if value is None:
            raise TypeError('Cache-Identitaet benoetigt NumPy-Arrays.')
        result.update(str(value.dtype).encode())
        result.update(str(value.shape).encode())
        result.update(value.tobytes(order='C'))
    return result.hexdigest()


class FitCache:
    """Local, authenticated-by-hash cache; never included in the share ZIP.

    A complete entry is one atomic torch.save envelope, so a power failure
    cannot expose a new file paired with an old checksum. The outer envelope
    is loaded with weights_only=True; only its SHA-verified inner bytes use
    full deserialization. Files are generated locally by this exact run.
    """
    def __init__(self, base, helper, root, identity):
        self.b, self.h = base, helper
        self.root, self.identity = Path(root), identity
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = dict(classifier=0, gan=0)

    def key(self, kind, identity):
        return self.h.digest(dict(kind=kind, identity=identity, run=self.identity))

    def path(self, kind, key):
        return self.root / kind / (key + '.pt')

    def save(self, path, value):
        import io
        buffer = io.BytesIO()
        self.b.torch.save(dict(cache_identity=self.identity, value=value), buffer)
        raw = buffer.getvalue()
        envelope = dict(format='si7_v21_sha256', sha256=hashlib.sha256(raw).hexdigest(),
                        payload=self.b.torch.from_numpy(self.b.np.frombuffer(raw, dtype='uint8').copy()))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.writing')
        self.b.torch.save(envelope, temporary)
        os.replace(temporary, path)

    def load(self, path):
        import io
        envelope = self.b.torch.load(path, map_location='cpu', weights_only=True)
        if envelope.get('format') != 'si7_v21_sha256':
            raise ValueError(f'Cacheformat widerspricht V2.1: {Path(path).name}')
        payload = envelope['payload']
        if payload.dtype != self.b.torch.uint8 or payload.ndim != 1:
            raise ValueError('Ungueltiger Cache-Payload.')
        raw = payload.numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != envelope['sha256']:
            raise ValueError(f'Cache-Pruefsumme stimmt nicht: {Path(path).name}')
        result = self.b.torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        if result.get('cache_identity') != self.identity:
            raise ValueError('Cache gehoert zu anderen Daten, Seeds oder Parametern.')
        return result['value']


def cached_classifier(base, cache, original):
    """Cache exactly the final fixed-update model and its inner OOF values."""
    np, torch = base.np, base.torch

    def fit(x, y, groups, minority, mode, ratio, fake, args, device, clock, seed,
            tag, valid=None, steps=None, collect_history=False):
        scaler = args._current_scaler
        scaling = array_digest(scaler.low, scaler.span,
            scaler.normalizer.raw_denominators_, scaler.normalizer.impute_)
        data = array_digest(x, y, groups)
        source = array_digest(fake) if fake is not None else None
        validation = array_digest(*valid) if valid is not None else None
        identity = dict(data=data, scaling=scaling, source=source, validation=validation,
            minority=int(minority), mode=mode, ratio=float(ratio), seed=int(seed),
            steps=int(args.classifier_steps if steps is None else steps),
            collect_history=bool(collect_history), backbone=args._factory.fingerprint)
        key = cache.key('classifier', identity)
        path = cache.path('classifier', key)
        if path.is_file():
            clock.check()
            saved = cache.load(path)
            if saved['identity'] != identity:
                raise ValueError('Klassifikatorcache ist inkonsistent.')
            base.seed_all(seed)
            model = base.SI7Probe(args._factory, scaler).to(device)
            model.base.restore_learned(saved['learned_state'])
            cache.hits['classifier'] += 1
            print(f'{tag}: fertigen Klassifikator-Fit wiederverwendet.', flush=True)
            return model.eval(), saved['history']
        model, history = original(x, y, groups, minority, mode, ratio, fake, args,
            device, clock, seed, tag, valid=valid, steps=steps,
            collect_history=collect_history)
        state = {k: v.detach().cpu().clone() for k, v in model.base.learned_state().items()}
        cache.save(path, dict(identity=identity, learned_state=state, history=history))
        return model, history
    return fit


def checkpoint_fit(base, scheme, cache, session_id, live_path):
    """Five requested paper ideas, with independent ablations and saved RNGs."""
    np, torch, F = base.np, base.torch, base.F

    def fit(x, groups, hidden, args, device, clock, seed, tag,
            history_rows=None, context=None, constant=None):
        history_rows = history_rows if history_rows is not None else []
        context = dict(context or {})
        constant = np.zeros(x.shape[2], bool) if constant is None else constant
        method_identity = asdict(scheme)
        for field in ('name', 'gan_cycles', 'production_cycles'):
            method_identity.pop(field)
        identity = dict(data=array_digest(x, groups, constant), hidden=int(hidden),
            seed=int(seed), layers=args.layers, gan_lr=args.gan_lr,
            batch_size=args.batch_size, ae_steps=args.ae_steps,
            supervisor_steps=args.supervisor_steps,
            method=method_identity,
            diagnostic_every=args.diagnostic_every)
        key = cache.key('gan', identity)
        folder = cache.root / 'gan' / key
        target_path = folder / f'GAN_{scheme.gan_cycles:06d}.pt'
        progress_path = folder / 'progress.pt'
        base.seed_all(seed)
        rng = np.random.default_rng(seed)
        model = base.TimeGAN(x.shape[2], hidden, args.layers).to(device).train()
        if scheme.packing != 1:
            # Replace only D's input dimension. Preserve E/R/G/S initialization
            # and the main latent-noise RNG when constructing the packed D.
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(cache.h.digest([seed, 'packed discriminator'])[:8], 16))
                model.D = type(model.D)(hidden * scheme.packing, hidden, 1,
                    args.layers, sigmoid=False).to(device)
        aux_rngs = {}
        for purpose in ('instance_noise', 'mode_seeking'):
            aux_rngs[purpose] = torch.Generator(device=device)
            aux_rngs[purpose].manual_seed(int(cache.h.digest([seed, purpose])[:8], 16))
        er = list(model.E.parameters()) + list(model.R.parameters())
        gs = list(model.G.parameters()) + list(model.S.parameters())
        oe = torch.optim.Adam(er, lr=args.gan_lr)
        osup = torch.optim.Adam(model.S.parameters(), lr=args.gan_lr)
        og = torch.optim.Adam(gs, lr=scheme.generator_lr)
        od = torch.optim.Adam(model.D.parameters(), lr=scheme.discriminator_lr)
        optimizers = dict(oe=oe, osup=osup, og=og, od=od)
        probabilities = base.row_probabilities(groups)
        reference = x[rng.choice(len(x), min(128, len(x)), replace=False, p=probabilities)]
        bce = torch.nn.BCEWithLogitsLoss()
        counts = dict(generator_updates=0, embedder_updates=0,
            supervisor_pretrain_updates=0, discriminator_updates=0,
            training_batches=0, mode_seeking_forward_pairs=0)
        trace, start_phase, start_step, global_step = [], 0, 0, 0

        def restore(saved):
            if saved['identity'] != identity:
                raise ValueError('GAN-Checkpoint hat eine andere Trainingsidentitaet.')
            model.load_state_dict(saved['model'])
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(saved['optimizers'][name])
            rng.bit_generator.state = saved['numpy_rng']
            torch.set_rng_state(saved['torch_rng'])
            random.setstate(saved['python_rng'])
            for purpose, generator in aux_rngs.items():
                generator.set_state(saved['aux_rngs'][purpose])
            if device.type == 'cuda':
                if saved['cuda_rng'] is None:
                    raise ValueError('CUDA-Zufallszustand fehlt im GAN-Checkpoint.')
                torch.cuda.set_rng_state(saved['cuda_rng'], device)

        def append_history():
            for original_row in trace:
                row = dict(original_row)
                row.update(context)
                row.update(training_variant=scheme.name, session_id=session_id,
                    planned_gan_cycles=scheme.gan_cycles,
                    paper_family=scheme.family, packing=scheme.packing,
                    generator_lr=scheme.generator_lr, discriminator_lr=scheme.discriminator_lr,
                    reused_training_trajectory=bool(start_phase or start_step))
                if row['phase'] == 'GAN':
                    row['phase_total'] = scheme.gan_cycles
                history_rows.append(row)

        if target_path.is_file():
            saved = cache.load(target_path)
            restore(saved)
            trace = saved['trace']
            start_phase, start_step = 2, scheme.gan_cycles
            append_history()
            cache.hits['gan'] += 1
            base.active(model, set())
            print(f'{scheme.name} | {tag}: GAN-Endpunkt {scheme.gan_cycles} wiederverwendet.', flush=True)
            return model.eval()
        if progress_path.is_file():
            saved = cache.load(progress_path)
            if saved['phase_index'] == 2 and saved['step'] > scheme.gan_cycles:
                raise ValueError('Frueher GAN-Endpunkt fehlt trotz spaeterem Trainingsstand.')
            restore(saved)
            start_phase, start_step = saved['phase_index'], saved['step']
            global_step, counts = saved['global_step'], saved['counts']
            trace = saved['trace']
            cache.hits['gan'] += 1
            print(f'{scheme.name} | {tag}: fortsetzen bei {saved["phase"]} {start_step}.', flush=True)

        def save(phase_index, step, path):
            state = dict(identity=identity, phase_index=phase_index,
                phase=('AE', 'SUP', 'GAN')[phase_index], step=step,
                global_step=global_step, counts=dict(counts), trace=list(trace),
                model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                optimizers={k: v.state_dict() for k, v in optimizers.items()},
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                python_rng=random.getstate(),
                aux_rngs={name: generator.get_state() for name, generator in aux_rngs.items()},
                cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
            cache.save(path, state)

        def batch():
            counts['training_batches'] += 1
            indices = rng.choice(len(x), args.batch_size, replace=True, p=probabilities)
            return torch.from_numpy(x[indices]).to(device)

        def supervise(h):
            return F.mse_loss(model.S(h)[:, :-1], h[:, 1:])

        def discriminate(values, cycle):
            return model.D(discriminator_input(values, scheme.packing,
                instance_sigma(scheme, cycle), aux_rngs['instance_noise']))

        def adversarial_term(values, is_real, cycle):
            logits = discriminate(values, cycle)
            target = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
            return bce(logits, target)

        def optimize(optimizer, loss, parameters):
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Nichtendlicher GAN-Loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(parameters), 5., error_if_nonfinite=True)
            optimizer.step()
            return float(norm.item())

        def spread(values):
            return float(values.std(dim=0, unbiased=False).mean().item())

        def snapshot(phase, step, total, losses):
            fake = base.generate(model, min(64, max(32, len(reference))), x.shape[1],
                args, device, constant, seed + 900000)
            with torch.no_grad():
                real = torch.from_numpy(reference).to(device)
                h = model.E(real)
                reconstruction = F.mse_loss(model.R(h), real).item()
                supervision = supervise(h).item()
                generator = torch.Generator(device=device)
                generator.manual_seed(int(seed + 900001))
                z = torch.rand(min(32, max(8, len(reference))), x.shape[1],
                    model.channels, device=device, generator=generator)
                eh, hh = model.G(z), None
                hh = model.S(eh)
                decoded = model.R(hh)
                decoded[..., torch.as_tensor(constant, dtype=torch.bool, device=device)] = 0
                zd = torch.mean(torch.abs(z - z.flip(0))).item()
                xd = torch.mean(torch.abs(decoded - decoded.flip(0))).item()
                extra = dict(real_embedding_std=spread(h), fake_embedding_std=spread(eh),
                    fake_supervisor_std=spread(hh), decoded_probe_std=spread(decoded),
                    noise_sensitivity_l1=float(xd / max(zd, 1e-12)))
            row = dict(context, hidden=hidden, gan_seed=seed, phase=phase,
                phase_step=step, phase_total=total, global_step=global_step,
                runtime_minutes=clock.minutes(), reconstruction_mse=float(reconstruction),
                supervisor_mse=float(supervision), **counts, **extra,
                generator_updates_per_cycle=scheme.generator_updates_per_cycle,
                paper_family=scheme.family, packing=scheme.packing,
                generator_lr=scheme.generator_lr, discriminator_lr=scheme.discriminator_lr,
                instance_sigma=instance_sigma(scheme, step) if phase == 'GAN' else 0.,
                mode_seeking_lambda=scheme.mode_seeking_lambda,
                **losses, **base.diversity_report(fake, reference))
            trace.append(row)
            live = dict(row, training_variant=scheme.name, session_id=session_id,
                planned_gan_cycles=scheme.gan_cycles)
            live_path.parent.mkdir(parents=True, exist_ok=True)
            with live_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(live, allow_nan=False) + '\n')
            model.train()  # cuDNN GRU backward also needs this for frozen blocks.

        phases = [('AE', args.ae_steps), ('SUP', args.supervisor_steps), ('GAN', scheme.gan_cycles)]
        for phase_index, (phase, total) in enumerate(phases):
            if phase_index < start_phase:
                continue
            first = start_step + 1 if phase_index == start_phase else 1
            began = time.monotonic()
            for step in range(first, total + 1):
                clock.check()  # No budget check inside one indivisible update cycle.
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
                    for update_index in range(scheme.generator_updates_per_cycle):
                        if update_index:
                            real = batch()
                        base.active(model, {'G', 'S'})
                        with torch.no_grad():
                            h = model.E(real)
                        z_primary = torch.rand_like(real)
                        eh = model.G(z_primary)
                        hh = model.S(eh)
                        generated = model.R(hh)
                        adv = adversarial_term(hh, True, step)
                        adv = adv + adversarial_term(eh, True, step)
                        moment = (generated.mean(0) - real.mean(0)).abs().mean()
                        moment = moment + (torch.sqrt(generated.var(0, unbiased=False) + 1e-6)
                            - torch.sqrt(real.var(0, unbiased=False) + 1e-6)).abs().mean()
                        loss = adv + 100 * torch.sqrt(supervise(h) + 1e-8) + 100 * moment
                        if scheme.mode_seeking_lambda:
                            z_second = torch.rand(real.shape, dtype=real.dtype,
                                device=device, generator=aux_rngs['mode_seeking'])
                            second_output = model.R(model.S(model.G(z_second)))
                            ratio_ms = mode_seeking_ratio(generated, second_output,
                                z_primary, z_second, ~constant)
                            counts['mode_seeking_forward_pairs'] += 1
                            # Eq.1 maximizes the L1 ratio; optimizer minimizes.
                            loss = loss - scheme.mode_seeking_lambda * ratio_ms
                            losses.update(mode_seeking_ratio=float(ratio_ms.item()),
                                mode_seeking_loss=float((-scheme.mode_seeking_lambda * ratio_ms).item()))
                        losses.update(generator_loss=float(loss.item()),
                            adversarial_loss=float(adv.item()), moment_loss=float(moment.item()))
                        losses['generator_grad_norm'] = optimize(og, loss, gs)
                        counts['generator_updates'] += 1
                        base.active(model, {'E', 'R'})
                        h = model.E(real)
                        embedder = 10 * torch.sqrt(F.mse_loss(model.R(h), real) + 1e-8) + .1 * supervise(h)
                        losses['embedder_grad_norm'] = optimize(oe, embedder, er)
                        counts['embedder_updates'] += 1
                    base.active(model, {'D'})
                    if scheme.fresh_discriminator_batch:
                        real = batch()  # Separate D minibatch as in the authors' code.
                    with torch.no_grad():
                        h = model.E(real)
                        eh = model.G(torch.rand_like(real))
                        hh = model.S(eh)
                    dloss = adversarial_term(h, True, step)
                    dloss = dloss + adversarial_term(hh, False, step)
                    dloss = dloss + adversarial_term(eh, False, step)
                    losses['discriminator_loss'] = float(dloss.item())
                    if dloss.item() > .15:
                        losses['discriminator_grad_norm'] = optimize(od, dloss, model.D.parameters())
                        counts['discriminator_updates'] += 1
                if step == 1 or step % args.diagnostic_every == 0 or step == total:
                    snapshot(phase, step, total, losses)
                if step == 1 or step % args.log_every == 0 or step == total:
                    eta = (time.monotonic() - began) / (step - first + 1) * (total - step) / 60
                    print(f'{scheme.name} | {tag} {phase} {step}/{total} '
                        f'Loss={loss.item():.4f} G={counts["generator_updates"]} '
                        f'D={counts["discriminator_updates"]} ETA={eta:.1f} min '
                        f'| {clock.minutes():.1f} min', flush=True)
                if step % args._checkpoint_every == 0 or step == total:
                    save(phase_index, step, progress_path)
                if phase == 'GAN' and step == total:
                    save(phase_index, step, target_path)
        # A process can stop after progress.pt was committed but before the
        # terminal file. Recover that endpoint even when no loop step remains.
        if not target_path.is_file():
            save(2, scheme.gan_cycles, target_path)
        append_history()
        base.active(model, set())
        return model.eval()
    return fit


def collect_paper_effects(helper, out, schemes):
    """Paired comparisons at the SAME duration, using committed folds only."""
    lookup = {(s.family, s.production_cycles): s for s in schemes}
    folds = helper.read_csv(out / 'training_variant_outer_fold_metrics.csv')
    by_fold = {(r['training_variant'], int(r['repeat']), int(r['fold'])): r
               for r in folds if r['method'] == 'timegan_selected'}
    repeats = {}
    for scheme in schemes:
        repeats[scheme.name] = {int(r['repeat']): r
            for r in helper.read_csv(out / scheme.name / 'repeat_patient_pooled.csv')
            if r['method'] == 'timegan_selected' and helper.yes(r['repeat_complete'])}
    fold_rows, repeat_rows, summaries = [], [], []
    for scheme in schemes:
        if scheme.family == 'legacy_1g':
            continue
        reference_family = 'legacy_1g' if scheme.family == 'timegan_2g' else 'timegan_2g'
        reference = lookup.get((reference_family, scheme.production_cycles))
        if reference is None:
            continue
        metadata = dict(family=scheme.family, target_variant=scheme.name,
            reference_variant=reference.name, gan_cycles=scheme.production_cycles,
            interpretation='paired same-duration ablation; dependent folds/seeds; descriptive')
        for key, target in by_fold.items():
            if key[0] != scheme.name:
                continue
            ref = by_fold.get((reference.name, key[1], key[2]))
            if ref is None:
                continue
            fold_rows.append(dict(metadata, repeat=key[1], fold=key[2],
                **{'delta_' + metric: float(target[metric]) - float(ref[metric])
                   for metric in helper.METRICS}))
        paired = sorted(set(repeats[scheme.name]) & set(repeats[reference.name]))
        specific = []
        for repeat in paired:
            a, b = repeats[scheme.name][repeat], repeats[reference.name][repeat]
            row = dict(metadata, repeat=repeat,
                **{'delta_' + metric: float(a[metric]) - float(b[metric]) for metric in helper.METRICS},
                target_cohen_kappa=float(a['cohen_kappa']), reference_cohen_kappa=float(b['cohen_kappa']))
            repeat_rows.append(row)
            specific.append(row)
        if specific:
            summaries.append(dict(metadata, n_completed_paired_repeats=len(specific),
                **{'mean_delta_' + metric: sum(r['delta_' + metric] for r in specific) / len(specific)
                   for metric in helper.METRICS}))
    helper.write_csv(out / 'paper_ablation_fold_differences.csv', fold_rows)
    helper.write_csv(out / 'paper_ablation_repeat_differences.csv', repeat_rows)
    helper.write_csv(out / 'paper_ablation_summary.csv', summaries)


def write_aggregate_zip(helper, out, schemes):
    # Exact allowlist; never recursively zip an output tree.
    root_names = ('training_variant_summary.csv', 'training_variant_outer_fold_metrics.csv',
        'training_variant_outer_seed_metrics.csv', 'training_variant_selected_configurations.csv',
        'training_variant_fold_differences.csv',
        'training_variant_repeat_differences.csv', 'generator_training_summary.csv',
        'generator_trajectory_summary.csv', 'baseline_consistency.csv', 'run_status.json',
        'protocol.json', 'paper_method_plan.csv', 'paper_ablation_fold_differences.csv',
        'paper_ablation_repeat_differences.csv', 'paper_ablation_summary.csv')
    variant_names = ('summary_patient_pooled.csv', 'outer_fold_metrics.csv', 'outer_seed_metrics.csv',
        'repeat_patient_pooled.csv', 'repeat_seed_patient_pooled.csv', 'selected_configurations.csv',
        'paired_outer_fold_differences.csv', 'paired_repeat_differences.csv',
        'paired_comparison_summary.csv', 'selection_frequency.csv',
        'inner_candidates.csv', 'inner_checkpoint_metrics.csv', 'inner_split_sizes.csv',
        'generation_diagnostics.csv', 'generation_channel_diagnostics.csv',
        'generation_distance_diagnostics.csv', 'timegan_training_history.csv',
        'run_status.json', 'import_audit_aggregate.csv', 'paper_method.json')
    paths = [out / name for name in root_names]
    paths += [out / scheme.name / name for scheme in schemes for name in variant_names]
    path = out / 'timegan_v21_aggregate.zip'
    temporary = path.with_name(path.name + '.writing')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in paths:
            if source.is_file():
                archive.write(source, source.relative_to(out).as_posix())
    os.replace(temporary, path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', nargs='?', type=Path, default=Path('CSV_Export'))
    parser.add_argument('--base-script', type=Path, default=Path(__file__).with_name(BASE_FILENAME))
    parser.add_argument('--compare-script', type=Path, default=Path(__file__).with_name(COMPARE_FILENAME))
    parser.add_argument('--output-dir', '--out', dest='out', type=Path, default=Path('results_timegan_papers_v21'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--weights-dir', type=Path, default=Path('pretrained_weights_v36'))
    parser.add_argument('--hours', type=float, default=24.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--folds', '--outer-folds', type=int, default=3)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--training-seeds', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-copies', default='0,3')
    parser.add_argument('--hidden', default='16')
    parser.add_argument('--gan-ratios', default='0.5')
    parser.add_argument('--random-ratios', default='0.5')
    parser.add_argument('--noise-sigma', type=float, default=.02)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--contact-microbatch', type=int, default=16)
    parser.add_argument('--ae-steps', type=int, default=500)
    parser.add_argument('--supervisor-steps', type=int, default=500)
    parser.add_argument('--classifier-steps', type=int, default=300)
    parser.add_argument('--gan-lr', type=float, default=.001)
    parser.add_argument('--cycles', default='5000,6000,7000', help='Nur 5000,6000,7000; Teilmengen erlaubt.')
    parser.add_argument('--families', default=','.join(FAMILIES), help='Kommagetrennte Teilmenge: ' + ','.join(FAMILIES))
    parser.add_argument('--instance-sigma-start', type=float, default=.05)
    parser.add_argument('--instance-sigma-end', type=float, default=0.)
    parser.add_argument('--instance-anneal-cycles', type=int, default=7000,
        help='Fester Rausch-Horizont fuer ALLE Trainingsdauern; macht Fortsetzung vergleichbar.')
    parser.add_argument('--mode-seeking-lambda', type=float, default=1.)
    parser.add_argument('--ttur-g-lr', type=float, default=.0005)
    parser.add_argument('--ttur-d-lr', type=float, default=.001)
    parser.add_argument('--classifier-lr', type=float, default=.001)
    parser.add_argument('--log-every', type=int, default=250)
    parser.add_argument('--diagnostic-every', type=int, default=250)
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--bootstrap-samples', type=int, default=0)
    parser.add_argument('--force-threshold', type=float, default=0.)
    parser.add_argument('--contact-duration-tolerance', type=float, default=.25)
    parser.add_argument('--contact-zero-tolerance', type=float, default=0.)
    parser.add_argument('--min-si-pairs-per-subject', type=int, default=3)
    parser.add_argument('--show-import-log', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--save-synthetic', action='store_true')
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--plan-only', action='store_true', help='Plan anzeigen, ohne Daten/GPU/Training.')
    args = parser.parse_args(argv)
    import math
    positive = [args.hours, args.repeats, args.folds, args.inner_folds,
        args.training_seeds, args.batch_size, args.contact_microbatch, args.ae_steps,
        args.supervisor_steps, args.classifier_steps, args.gan_lr, args.classifier_lr,
        args.log_every, args.diagnostic_every, args.checkpoint_every, args.threads,
        args.noise_sigma, args.min_si_pairs_per_subject, args.ttur_g_lr, args.ttur_d_lr]
    if not all(math.isfinite(v) and v > 0 for v in positive):
        parser.error('Budgets und Trainingsparameter muessen endlich und positiv sein.')
    if min(args.folds, args.inner_folds, args.layers) < 2:
        parser.error('folds, inner-folds und layers muessen mindestens 2 sein.')
    if not 0 <= args.seed < 2**32 or args.bootstrap_samples < 0:
        parser.error('Ungueltiger Masterseed oder Bootstrap-Umfang.')
    try:
        requested_cycles = {int(v.strip()) for v in args.cycles.split(',')}
        requested_families = {v.strip() for v in args.families.split(',')}
        hidden = {int(v) for v in args.hidden.split(',')}
        copies = {int(v) for v in args.noise_copies.split(',')}
        ratios = [float(v) for v in (args.gan_ratios + ',' + args.random_ratios).split(',')]
    except ValueError:
        parser.error('Suchraeume als kommagetrennte Zahlen angeben.')
    if not requested_cycles or not requested_cycles.issubset(CYCLES):
        parser.error('--cycles darf ausschliesslich 5000,6000,7000 enthalten.')
    if not requested_families or not requested_families.issubset(FAMILIES):
        parser.error('Unbekannte --families; gueltig: ' + ','.join(FAMILIES))
    if not hidden or min(hidden) < 2 or not copies or min(copies) < 0 or max(copies) == 0:
        parser.error('GAN-Breite >=2, Kopien >=0 und mindestens eine positive Kopienzahl erforderlich.')
    if not ratios or not all(math.isfinite(v) and v > 0 for v in ratios):
        parser.error('Augmentationsverhaeltnisse muessen endlich und positiv sein.')
    extras = [args.instance_sigma_start, args.instance_sigma_end, args.mode_seeking_lambda,
              args.force_threshold, args.contact_duration_tolerance, args.contact_zero_tolerance]
    if not all(math.isfinite(v) and v >= 0 for v in extras):
        parser.error('Rauschstaerken, Mode-Seeking-Gewicht und QC-Toleranzen muessen endlich und nichtnegativ sein.')
    if args.instance_anneal_cycles < 2 or args.instance_sigma_start < args.instance_sigma_end:
        parser.error('Instance Noise: anneal-cycles >=2 und sigma-start >= sigma-end erforderlich.')
    if 'pacgan' in requested_families and args.batch_size % 2:
        parser.error('PacGAN2 benoetigt eine gerade --batch-size.')
    if 'ttur' in requested_families and args.ttur_g_lr == args.ttur_d_lr:
        parser.error('TTUR benoetigt unterschiedliche Generator-/Diskriminator-Lernraten.')
    args.cycles = ','.join(str(v) for v in sorted(requested_cycles))
    args.families = ','.join(v for v in FAMILIES if v in requested_families)
    if args.out.exists() and not args.out.is_dir():
        parser.error('--output-dir muss ein Ordner sein.')
    return args


def run(base, helper, args, schemes):
    """Finish each condition's CV, starting at 5000; share all control fits."""
    args.reference_cycles = schemes[0].gan_cycles
    args.fold_limit = 1
    args.variants = ','.join(s.name for s in schemes)
    old_version = helper.VERSION
    helper.VERSION = VERSION
    try:
        identity, fingerprint = helper.prepare_manifest(args, schemes)
    finally:
        helper.VERSION = old_version
    # V1.7 hashes its own __file__; add this V2.1 runner explicitly too.
    cache = FitCache(base, helper, args.out / 'fit_cache_LOCAL_ONLY', fingerprint)
    original_classifier = base.fit_classifier
    original_train_unit = base.train_outer_unit
    original_clock = base.Clock
    base.fit_classifier = cached_classifier(base, cache, original_classifier)

    def train_unit(unit, base_args, *a, **kw):
        base_args._checkpoint_every = args.checkpoint_every
        return original_train_unit(unit, base_args, *a, **kw)
    base.train_outer_unit = train_unit
    began, deadline = time.monotonic(), time.monotonic() + args.hours * 3600
    class SessionClock(original_clock):
        def __init__(self, hours):
            super().__init__(hours)
            self.end = min(self.end, deadline)
    base.Clock = SessionClock
    session_id = uuid.uuid4().hex
    status, code = 'running', 0
    validated_complete_variants = set()
    state = helper.collect_comparison(args.out, schemes)
    protocol = dict(version=VERSION, schemes=[asdict(s) for s in schemes],
        training_seeds=args.training_seeds, seed_slots=list(range(1, args.training_seeds + 1)),
        repeats=args.repeats, outer_folds=args.folds, inner_folds=args.inner_folds,
        noise_copies=args.noise_copies, hidden=args.hidden, gan_ratios=args.gan_ratios,
        classifier_updates_fixed=args.classifier_steps,
        optimization='pooled inner OOF ensemble Cohen kappa; same seed count inside and outside',
        threshold_grid='0.10..0.90 in steps of 0.05; unchanged V1.6 tie-breaking',
        checkpoint_every=args.checkpoint_every,
        allowed_changes=['5000/6000/7000 cycles', 'author two-G/E updates and fresh discriminator minibatch',
                         'latent instance noise', 'output mode-seeking L1 ratio',
                         'PacGAN2 latent packing', 'separate TTUR G and D learning rates'],
        paper_sources=PAPER_SOURCES, technical_self_test=args.self_test,
        adaptations='paper ideas adapted to compact PyTorch TimeGAN/SI7; no guarantee from image benchmarks',
        schedule='cycles ascending; complete CV of each condition before next condition',
        primary_method_reference='timegan_2g with same cycle count; legacy_1g compares training schema',
        TTUR_defaults='G=0.0005; D=0.001; E/R and S pretraining keep gan-lr; predefined hypothesis, not HPO',
        instance_noise_schedule='linear, fixed common horizon; independent Gaussian on real/fake D inputs in G and D steps',
        mode_seeking_objective='L_G minus lambda*mean|RSG(z1)-RSG(z2)|/mean|z1-z2|; constant channels excluded',
        supplementary_variant_comparison='exploratory; no outer-test model selection',
        interpretation='same patient dataset; dependent folds and seeds; no new independent cohort')
    helper.atomic_json(args.out / 'protocol.json', protocol)
    helper.write_csv(args.out / 'paper_method_plan.csv', [asdict(s) for s in schemes])

    def save_status():
        helper.atomic_json(args.out / 'run_status.json', dict(version=VERSION, status=status,
            fingerprint=fingerprint, session_id=session_id,
            runtime_minutes_this_session=(time.monotonic() - began) / 60,
            planned_variants=[s.name for s in schemes], variant_states=state['states'],
            planned_outer_folds_per_variant=args.repeats * args.folds,
            completed_outer_folds=sum(v.get('completed_outer_folds', 0) for v in state['states'].values()),
            n_unique_patients=state['n_unique_patients'],
            baseline_consistency_passed=state['baseline_consistency_passed'],
            reused_classifier_fits=cache.hits['classifier'], reused_gan_trajectories=cache.hits['gan'],
            training_seeds=args.training_seeds, budget_hours_this_session=args.hours))

    print(f'V{VERSION}: {len(schemes)} Bedingungen; {args.repeats} x {args.folds} aussen '
        f'x {args.inner_folds} innen x {args.training_seeds} Seeds; Budget {args.hours:g} h.', flush=True)
    print('GAN-Kopien:', args.noise_copies,
          '| Gemeinsame innere Kappa-Auswahl auf ALLEN Ensemble-Seeds.', flush=True)
    try:
        save_status()
        for scheme in schemes:
            for round_index in range(args.repeats * args.folds):
                variant_state = state['states'][scheme.name]
                completed = variant_state.get('completed_outer_folds', 0)
                validate_complete = (args.resume and completed == args.repeats * args.folds
                    and scheme.name not in validated_complete_variants)
                if completed > round_index and not validate_complete:
                    continue
                remaining = (deadline - time.monotonic()) / 3600
                if remaining <= 0:
                    status, code = 'budget_exceeded', 2
                    break
                folder = args.out / scheme.name
                resume_variant = (folder / 'configuration.json').is_file()
                originals = helper.install_scheme(base, scheme,
                    runner_sha256=identity['overnight_sha256'],
                    base_sha256=identity['base_sha256'], session_id=session_id, folder=folder)
                original_fields = base.candidate_fields
                def fields(candidate, original=original_fields, method=scheme):
                    result = original(candidate)
                    result.update(paper_family=method.family, packing=method.packing,
                        generator_lr=method.generator_lr, discriminator_lr=method.discriminator_lr,
                        instance_sigma_start=method.instance_sigma_start,
                        instance_sigma_end=method.instance_sigma_end,
                        instance_anneal_cycles=method.instance_anneal_cycles,
                        mode_seeking_lambda=method.mode_seeking_lambda,
                        fresh_discriminator_batch=method.fresh_discriminator_batch)
                    return result
                base.candidate_fields = fields
                base.fit_timegan = checkpoint_fit(base, scheme, cache, session_id,
                    folder / 'gan_training_live.jsonl')
                argv = helper.base_arguments(args, scheme, folder, remaining, resume_variant)
                if args.check_data:
                    argv.append('--check-data')
                if args.self_test:
                    argv.append('--self-test')
                try:
                    print(f'\n=== {scheme.name}; naechster aeusserer Fold '
                        f'{completed + 1}/{args.repeats * args.folds} ===', flush=True)
                    code = base.main(argv)
                    helper.atomic_json(folder / 'paper_method.json', dict(**asdict(scheme),
                        sources=PAPER_SOURCES, E_R_pretraining_lr=args.gan_lr,
                        reconstruction='PacGAN D has packing*hidden input features; generator dimensions unchanged'))
                    validated_complete_variants.add(scheme.name)
                finally:
                    helper.restore_scheme(base, originals)
                    gc.collect()
                    if base.torch.cuda.is_available():
                        base.torch.cuda.empty_cache()
                state = helper.collect_comparison(args.out, schemes)
                collect_paper_effects(helper, args.out, schemes)
                save_status()
                write_aggregate_zip(helper, args.out, schemes)
                if code:
                    status = state['states'][scheme.name].get('status', 'failed')
                    break
                if args.check_data:
                    status = 'checked_data_only'
                    break
            if code or args.check_data:
                break
        if not code and not args.check_data:
            status = 'complete' if all(v.get('status') == 'complete'
                for v in state['states'].values()) else 'partial'
    except KeyboardInterrupt:
        status, code = 'interrupted', 2
        print('Unterbrochen. Fertige Fits und GAN-Checkpoints bleiben gespeichert.', flush=True)
    except BaseException:
        status, code = 'failed', 2
        raise
    finally:
        base.fit_classifier, base.train_outer_unit, base.Clock = original_classifier, original_train_unit, original_clock
        state = helper.collect_comparison(args.out, schemes)
        collect_paper_effects(helper, args.out, schemes)
        if state['baseline_consistency_passed'] is False:
            status, code = 'baseline_mismatch', 3
        save_status()
        write_aggregate_zip(helper, args.out, schemes)
        print(f'\nGesamtstatus={status}; {(time.monotonic() - began) / 60:.1f} min in dieser Sitzung.', flush=True)
        for row in state['summaries']:
            print(f'{row["training_variant"]:20s} {row["method"]:18s} '
                f'BA={float(row["balanced_accuracy"]):.3f} Accuracy={float(row["accuracy"]):.3f} '
                f'Kappa={float(row["cohen_kappa"]):.3f} '
                f'({row["n_completed_repeats"]} vollstaendige Wiederholung(en))', flush=True)
        print('Ausgabe:', args.out.resolve(), flush=True)
        print('Nur aggregierte Daten zum Teilen:', (args.out / 'timegan_v21_aggregate.zip').resolve(), flush=True)
    return int(code)


def main(argv=None):
    args = parse_args(argv)
    if args.plan_only:
        print(json.dumps(dict(version=VERSION, budget_hours=args.hours, device=args.device,
            nested_cv=[args.repeats, args.folds, args.inner_folds], seeds=args.training_seeds,
            schemes=[asdict(s) for s in build_schemes(args)]), indent=2))
        return 0
    helper = load_compare(args.compare_script)
    base = helper.load_base(args.base_script)
    args.overnight_sha256 = helper.sha256_file(__file__)
    args.compare_sha256 = helper.sha256_file(args.compare_script)
    if args.self_test:
        # base.main itself reduces its technical classifier and synthetic CSVs.
        args.repeats, args.folds, args.inner_folds, args.training_seeds = 1, 2, 2, 2
        args.hidden, args.noise_copies = '4', '0,1'
        args.ae_steps = args.supervisor_steps = 1
        args.classifier_steps = 2
        args.batch_size, args.diagnostic_every, args.checkpoint_every = 4, 1, 1
        args.bootstrap_samples = 0
        args.instance_anneal_cycles = 4
        args.hours = min(args.hours, 1.)
    schemes = build_schemes(args)
    return run(base, helper, args, schemes)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise SystemExit(str(exc)) from exc
