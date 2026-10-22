#!/usr/bin/env python3
r"""SI09 Mode Seeking versus Real+Noise, repeated Nested-CV, V2.0.

PowerShell in the existing data/venv directory:
  & ".\.venv\Scripts\python.exe" .\si09_modeseeking_noise_24h_v2\start_si09_modeseeking_noise_24h.py .\CSV_Export --device cuda --hours 24
Resume the same command with --resume. Inspect the full plan with --plan-only.

Exactly two arms: real+direct noise; real+direct noise+Mode-Seeking TimeGAN.
True one-channel SI09, all 101 curve points, frozen ConvNeXt backbone.
Default: THREE NEW CV repetitions (indices 2,3,4), each 3 outer x 3 inner
patient folds, six classifier seeds and three GAN seeds. The 24h target is
ADVISORY: training continues after 24h until all planned units are complete.

Stage 1 jointly compares 4000/5000/6000/7000/8000 GAN cycles, r=.25/.5/1,
q=.25/.5/.75, noise sigma=.01/.02: 90 GAN configurations. The control
compares all six q/sigma configurations with r=0 and NEVER trains a GAN.
Both arms use the SAME fixed reference head at 300 classifier updates.
Stage 2 locks the entire chosen augmentation setting and compares the SAME
12 classifier profiles, checkpoints 150/300/600 and inner thresholds .1-.9.
All selection is by patient-pooled inner OOF ensemble Cohen kappa.

A separate fixed augmentation recommendation aggregates ALL stage-1
configurations on the common COMPLETE repetitions. It never reads outer
test scores. This is a development recommendation, not a separately tested
fixed model: outer metrics evaluate each fold's adaptive HPO pipeline.
It becomes ready only after all planned repetitions AND at least three
complete paired repetitions (two in the artificial self-test). Earlier
rankings/recommendations are explicitly provisional.

New output folder only. No previous result directories are accessed.
GAN endpoints and frozen features are cached locally; --resume preserves
the exact plan. The aggregate ZIP excludes patient identifiers, individual
predictions, raw data, split manifests, models and local training caches.
"""


from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import time
import uuid
import zipfile
from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

VERSION = '2.0.0'
BASE_FILE = 'smn_basis_si7.py'
COMPARE_FILE = 'smn_training_utils.py'
PAPERS_FILE = 'smn_gan_verfahren.py'
SINGLE_FILE = 'smn_si09_modell.py'
METHOD = 'selected_augmentation'
FAMILIES = ('noise_only', 'mode_seeking')
MODELS = tuple('si09_' + family for family in FAMILIES)
SOURCES = tuple(f'mode_seeking_{cycle}' for cycle in (4000,5000,6000,7000,8000))


GRAY_MEAN = (.485 + .456 + .406) / 3
GRAY_STD = (.229 + .224 + .225) / 3

# Measured on the user's previous CUDA run (si09_gan_nacht_aggregate.zip).
# Phase rates: median positive time/step delta after deduplicating trajectories.
# Classifier rates: median fit_seconds / trained_steps for each head structure.
# The requested hardware is a GTX 1070. These are estimates, not a stopwatch.
GTX1070_TIMING = dict(
    source='user-provided si09_gan_nacht_aggregate.zip, complete 686.2854-minute CUDA session',
    hardware_assumption='NVIDIA GTX 1070 with the same data, CPU and storage conditions',
    gan_cycle_seconds=.03909586, ae_step_seconds=.00443574, supervisor_step_seconds=.003,
    classifier_step_seconds={'cnn_0':3.281/600,'cnn_32':3.438/600,
        'cnn_transformer_8':4.672/600,'cnn_transformer_16':4.890/600,
        'cnn_transformer_24':4.859/600},
    additional_overhead_fraction=.12)



class AdvisoryClock:
    """Display a time target, never stop scientific work when it is exceeded."""
    def __init__(self, hours, start=None):
        self.start = time.monotonic() if start is None else start
        self.target_hours = float(hours)
        self.end = float('inf')
        self.warned = False

    def minutes(self):
        return (time.monotonic()-self.start)/60

    def check(self):
        if not self.warned and self.minutes() >= self.target_hours*60:
            self.warned = True
            print(f'Zeitplanung von {self.target_hours:g} h erreicht. '
                'KEIN Zeitabbruch: alle geplanten Wiederholungen werden weitergerechnet.', flush=True)


def write_csv(path, rows):
    """Flush a complete table to disk before replacing the preceding version."""
    path=Path(path)
    if not rows:
        if path.is_file():
            path.unlink()
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.writing')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


@dataclass(frozen=True)
class Candidate:
    source: str
    gan_input_copies: int = 3
    gan_ratio: float = .5
    noise_probability: float = .75
    noise_sigma: float = .02
    learning_rate: float = .001
    weight_decay: float = .01
    dropout: float = .1
    transformer_width: int = 16
    architecture: str = 'cnn_transformer'
    feature_dropout: float = 0.
    head_hidden: int = 0
    gan_hidden: int = 16
    gan_learning_rate: float = .001

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def row(self):
        return dict(candidate_id=self.key, **asdict(self))



def family_for_model(model_id):
    if model_id not in MODELS:
        raise ValueError('Only SI09 Mode Seeking and SI09 Real+Noise are allowed.')
    return model_id.removeprefix('si09_')


def repeat_ids(args):
    return list(range(args.first_repeat, args.first_repeat + args.repeats))


def split_plan(base, gy, args):
    expanded = SimpleNamespace(**vars(args))
    expanded.repeats = args.first_repeat + args.repeats - 1
    return [u for u in base.build_split_plan(gy, expanded) if u['repeat'] in repeat_ids(args)]


def augmentation_key(candidate):
    row = asdict(candidate) if isinstance(candidate, Candidate) else candidate
    fields = ('source', 'gan_input_copies', 'gan_ratio', 'noise_probability',
              'noise_sigma', 'gan_hidden', 'gan_learning_rate')
    return tuple(str(row[k]) for k in fields)


def generator_plan(args, model_id):
    family = family_for_model(model_id)
    sources = SOURCES if family == 'mode_seeking' else ('noise_only',)
    if family == 'mode_seeking':
        sources = tuple(f'mode_seeking_{v}' for v in args.gan_cycles)
    ratios = args.gan_ratios if family == 'mode_seeking' else [0.]
    # One identical reference classifier for every augmentation configuration.
    return [Candidate(source, gan_input_copies=(1 if args.self_test else 3) if family=='mode_seeking' else 0,
        gan_ratio=r, noise_probability=q, noise_sigma=sigma,
        gan_hidden=4 if args.self_test else 16, gan_learning_rate=.001,
        architecture='cnn', transformer_width=0, head_hidden=32,
        feature_dropout=.1, learning_rate=.0003, weight_decay=.01, dropout=.2)
        for source in sources for r in ratios for q in args.noise_probabilities for sigma in args.noise_sigmas]


def workload_estimate(args, plans):
    """Dimension finite training operations using the previous measured run."""
    n_inner=args.repeats*args.folds*args.inner_folds*args.training_seeds
    n_outer=args.repeats*args.folds*args.training_seeds
    inner_gan=args.repeats*args.folds*args.inner_folds*args.gan_seeds
    outer_gan=args.repeats*args.folds*args.gan_seeds
    trajectories=inner_gan+outer_gan
    cycles=trajectories*max(args.gan_cycles)
    augmentation_fits=n_inner*sum(map(len,plans.values()))
    classifier_fits=n_inner*args.trials*len(MODELS)
    ref_updates=1 if args.self_test else 300
    head_updates=max(args.checkpoints)
    rates=GTX1070_TIMING['classifier_step_seconds']
    def rate(candidate):
        key=f'cnn_{candidate.head_hidden}' if candidate.architecture=='cnn' else f'cnn_transformer_{candidate.transformer_width}'
        return rates[key]
    augmentation_seconds=sum(n_inner*ref_updates*rate(c) for choices in plans.values() for c in choices)
    head_seconds=outer_seconds=0.
    for model_id,choices in plans.items():
        heads=head_plan(args.trials,args.seed,args.self_test,model_id,choices[0])
        head_seconds+=n_inner*head_updates*sum(map(rate,heads))
        # Outer head/checkpoint is unknown before HPO: average profile rate,
        # largest allowed checkpoint, and the largest GAN duration are budgeted.
        outer_seconds+=n_outer*head_updates*sum(map(rate,heads))/len(heads)
    gan_seconds=(cycles*GTX1070_TIMING['gan_cycle_seconds']+
        trajectories*args.ae_steps*GTX1070_TIMING['ae_step_seconds']+
        trajectories*args.supervisor_steps*GTX1070_TIMING['supervisor_step_seconds'])
    raw=gan_seconds+augmentation_seconds+head_seconds+outer_seconds
    return dict(calibration=GTX1070_TIMING,applicable_to_this_run=not args.self_test,
        gan_inner_trajectories=inner_gan,gan_outer_trajectories=outer_gan,
        gan_cycles_at_maximum_outer_duration=cycles,
        generator_supervisor_updates=2*cycles,encoder_decoder_updates=2*cycles,
        discriminator_updates_upper_bound=cycles,
        ae_pretrain_updates=trajectories*args.ae_steps,supervisor_pretrain_updates=trajectories*args.supervisor_steps,
        augmentation_classifier_fits=augmentation_fits,classifier_hpo_fits=classifier_fits,
        outer_classifier_fits=n_outer*len(MODELS),
        total_classifier_updates=augmentation_fits*ref_updates+classifier_fits*head_updates+n_outer*len(MODELS)*head_updates,
        estimated_gan_training_hours=gan_seconds/3600,
        estimated_augmentation_hpo_hours=augmentation_seconds/3600,
        estimated_classifier_hpo_hours=head_seconds/3600,
        estimated_outer_classifier_hours=outer_seconds/3600,
        estimated_training_hours=raw/3600,
        estimated_total_hours_including_reserve=raw*(1+GTX1070_TIMING['additional_overhead_fraction'])/3600,
        target_hours=args.hours,finish_condition='all finite planned units complete; no clock cutoff',
        assumptions='same cohort/architecture/backend; shared GAN prefixes; cached frozen CNN maps; largest outer GAN duration and checkpoint budgeted')


def head_plan(trials=12, seed=42, technical=False, model_id='si09_mode_seeking', generator=None):
    """Common classifier search; ALL augmentation fields are held fixed."""
    import random
    generator = generator or Candidate('mode_seeking_4000')
    definitions = [
        ('cnn',0,32,.0003,.01,.2), ('cnn_transformer',24,0,.0003,.001,.1),
        ('cnn',0,32,.001,.001,.1), ('cnn_transformer',16,0,.001,.001,.1),
        ('cnn',0,0,.0003,.001,.2), ('cnn_transformer',8,0,.001,.001,.2),
        ('cnn',0,0,.001,.01,.1), ('cnn_transformer',16,0,.0003,.01,.2),
        ('cnn',0,32,.001,.01,.2), ('cnn_transformer',8,0,.0003,.001,.1),
        ('cnn',0,32,.0003,.001,.1), ('cnn_transformer',24,0,.001,.01,.2)]
    rng, result, seen = random.Random(int(seed)+2202), [], set()
    while len(result) < trials:
        if len(result) < len(definitions):
            arch,width,hidden,lr,wd,drop = definitions[len(result)]
        else:
            arch = 'cnn' if len(result)%2==0 else 'cnn_transformer'
            width = rng.choice((8,16,24)) if arch!='cnn' else 0
            hidden = rng.choice((0,32)) if arch=='cnn' else 0
            lr,wd,drop = rng.choice((.0003,.001)),rng.choice((.001,.01)),rng.choice((.1,.2))
        candidate = replace(generator, architecture=arch, transformer_width=width,
            head_hidden=hidden, learning_rate=lr, weight_decay=wd, dropout=drop, feature_dropout=.1)
        if candidate.key not in seen:
            seen.add(candidate.key); result.append(candidate)
    assert all(augmentation_key(c)==augmentation_key(generator) for c in result)
    return result


def draw_probabilities(gan_ratio, noise_probability, n_patients, n_minority):
    if not (gan_ratio >= 0 and 0 < noise_probability < 1 and 0 < n_minority < n_patients):
        raise ValueError('Nonnegative GAN ratio, positive real/noise shares and both classes are required.')
    pg = gan_ratio*n_minority/(n_patients+gan_ratio*n_minority)
    return ((1-pg)*(1-noise_probability), (1-pg)*noise_probability, pg)


def si09_images(base, pairs, width):
    """Only SI09, full image height, exactly ONE image input channel."""
    if pairs.ndim != 3 or tuple(pairs.shape[1:]) != (1, 101):
        raise ValueError('SI09 erwartet genau ein Signal [N,1,101].')
    values = .5 + base.torch.atan(pairs) / math.pi
    values = base.F.interpolate(values, size=width, mode='linear', align_corners=True)
    return ((values.unsqueeze(2).expand(-1, 1, 96, -1) - GRAY_MEAN) / GRAY_STD).contiguous()


def si09_factory(base, rgb):
    factory = copy.copy(rgb)
    factory.features = copy.deepcopy(rgb.features)
    stem = factory.features[0][0]
    if not isinstance(stem, base.nn.Conv2d) or stem.in_channels != 3 or stem.groups != 1:
        raise ValueError('Ein ConvNeXt-Stem mit drei Eingangskanaelen wird erwartet.')
    new = base.nn.Conv2d(1, stem.out_channels, stem.kernel_size, stem.stride, stem.padding,
        stem.dilation, bias=stem.bias is not None, padding_mode=stem.padding_mode)
    with base.torch.no_grad():
        new.weight.copy_(stem.weight.sum(1, keepdim=True))
        if stem.bias is not None:
            new.bias.copy_(stem.bias)
    factory.features[0][0] = new
    factory.features.eval()
    for parameter in factory.features.parameters():
        parameter.requires_grad_(False)
    factory.provenance = rgb.provenance + '; SI09 full height, true one-channel summed RGB stem'
    factory.fingerprint = factory._fingerprint()
    return factory


def make_probe(base, factory, scaler, candidate, model_id, single=None):
    config = replace(base.probe_config(), learning_rate=candidate.learning_rate,
        weight_decay=candidate.weight_decay, head_dropout=candidate.dropout,
        transformer_dropout=candidate.dropout, transformer_width=candidate.transformer_width,
        feature_dropout=candidate.feature_dropout)
    if model_id in MODELS:
        if single is None:
            raise ValueError('Die hochgeladene SI09-Modell-Datei fehlt.')

        @dataclass(frozen=True)
        class BinarySI09Config(single.TransferConfig):
            task: str = 'classification'

            def __post_init__(self):
                if self.task != 'classification':
                    raise ValueError('Der SI09-Nachtlauf verwendet explizit binaere Klassifikation.')
                fields = asdict(self)
                fields['task'] = 'angle_regression'
                single.TransferConfig(**fields)  # Keep the source architecture validation.

        is_t = candidate.architecture == 'cnn_transformer'
        config = BinarySI09Config(architecture=candidate.architecture, channel_set='SI09',
            learning_rate=candidate.learning_rate, weight_decay=candidate.weight_decay,
            head_dropout=candidate.dropout, feature_dropout=candidate.feature_dropout,
            head_hidden=candidate.head_hidden, transformer_width=candidate.transformer_width if is_t else 0,
            attention_heads=2 if is_t else 0, transformer_dropout=candidate.dropout if is_t else 0.,
            feedforward_multiplier=2 if is_t else 0, tokens=4 if is_t else 0)

    class Probe(base.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = (single if model_id in MODELS else base).TransferModel(factory, config)
            self.register_buffer('gan_low', base.torch.as_tensor(scaler.low).reshape(1, 1, -1))
            self.register_buffer('gan_denom', base.torch.as_tensor(scaler.denom).reshape(1, 1, -1))

        def raw_maps(self, x):
            pairs = (x * self.gan_denom + self.gan_low).transpose(1, 2)
            if model_id == 'current_si7':
                return self.base.maps(pairs)
            return self.base.backbone(si09_images(base, pairs, config.image_width))

        def from_maps(self, maps):
            return self.base.from_maps(maps)

        def forward(self, x):
            return self.from_maps(self.raw_maps(x))

    return Probe()


def load_module(path, name):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Benoetigte unveraenderte Datei fehlt: {path.name}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_si09(base, single, args, gy, names):
    """Use the supplied SI09 importer; align by patient, never by row number."""
    output = io.StringIO()
    with redirect_stdout(output):
        records, dataset, audit, _ = single.load_standalone_si15_records(args.data_dir,
            force_threshold=args.force_threshold,
            contact_duration_tolerance=args.contact_duration_tolerance,
            contact_zero_tolerance=args.contact_zero_tolerance,
            min_si_pairs_per_subject=args.min_si_pairs_per_subject)
    if args.show_import_log:
        print(output.getvalue(), end='', flush=True)
    owners = {str(record.subject): record for record in records}
    if len(owners) != len(records) or set(owners) != {str(name) for name in names}:
        raise ValueError('SI7- und SI09-Import liefern unterschiedliche Patientenkohorten. '
                         'Kein stilles Weglassen von Patienten; Import/QC zuerst abstimmen.')
    ordered = [owners[str(name)] for name in names]
    labels = base.np.asarray([r.hard_reference for r in ordered], base.np.int64)
    if not base.np.array_equal(labels, gy):
        raise ValueError('SI7 und SI09 besitzen unterschiedliche Patientenlabels.')
    if single.SELECTED_SOURCE_CHANNEL_INDICES != (3, 4, 8):
        raise ValueError('Die hochgeladene SI09-Pipeline muss die Quellindices 3,4,8 verwenden.')
    x = base.np.concatenate([r.pairs[:, 2:3, :].transpose(0, 2, 1) for r in ordered]).astype(base.np.float32)
    groups = base.np.concatenate([base.np.full(r.n_pairs, i, base.np.int64) for i, r in enumerate(ordered)])
    y = gy[groups]
    if x.shape[1:] != (101, 1) or not base.np.array_equal(base.np.unique(groups), base.np.arange(len(gy))):
        raise ValueError('SI09-Import muss [N,101,1] fuer alle Vergleichspatienten liefern.')
    print(f'SI09-Import: {len(ordered)} identische Patienten; {len(x)} Paare [101,1]; '
        f'Importfehler={len(dataset["import_failures"])}.', flush=True)
    return x, y, groups


def fit_input_scaler(base, single, model_id, x, groups):
    if model_id == 'current_si7':
        return base.SI7Scaler().fit(x, groups)

    class SI09Scaler(base.Scaler):
        def fit(self, values, owners):
            records = [SimpleNamespace(measurements=[SimpleNamespace(
                pairs=values[owners == owner].transpose(0, 2, 1))]) for owner in base.np.unique(owners)]
            self.normalizer = single.FoldInputNormalizer(raw_indices=()).fit(records)
            normalized = self.normalizer.transform_pairs(values.transpose(0, 2, 1)).transpose(0, 2, 1)
            return super().fit(normalized)

        def transform(self, values):
            normalized = self.normalizer.transform_pairs(values.transpose(0, 2, 1)).transpose(0, 2, 1)
            return super().transform(normalized)

    return SI09Scaler().fit(x, groups)



def source_schemes(papers, args):
    # Build the existing PaperScheme directly: the paper module's technical
    # cycle lookup only covers 5000/6000/7000; production training is general.
    return {f'mode_seeking_{cycle}': papers.PaperScheme(
        name=f'mode_seeking_{cycle}', gan_cycles=index+2 if args.self_test else cycle,
        generator_updates_per_cycle=2, family='mode_seeking', production_cycles=cycle,
        fresh_discriminator_batch=True, instance_sigma_start=0., instance_sigma_end=0.,
        instance_anneal_cycles=8000, mode_seeking_lambda=1., packing=1,
        generator_lr=.001, discriminator_lr=.001)
        for index,cycle in enumerate(args.gan_cycles)}


def scheme_for_candidate(schemes, candidate):
    scheme = schemes[candidate.source]
    multiplier = candidate.gan_learning_rate / .001
    return replace(scheme, generator_lr=scheme.generator_lr * multiplier,
        discriminator_lr=scheme.discriminator_lr * multiplier)


def gan_identity(papers, scheme, x, groups, constant, hidden, seed, args):
    method = asdict(scheme)
    for field in ('name', 'gan_cycles', 'production_cycles'):
        method.pop(field)
    return dict(data=papers.array_digest(x, groups, constant), hidden=int(hidden),
        seed=int(seed), layers=args.layers, gan_lr=args.gan_lr, batch_size=args.batch_size,
        ae_steps=args.ae_steps, supervisor_steps=args.supervisor_steps,
        method=method, diagnostic_every=args.diagnostic_every)


def generate_checked(base, model, count, length, args, device, constant, seed, clock):
    model.eval()
    generator = base.torch.Generator(device=device).manual_seed(int(seed))
    pieces = []
    with base.torch.no_grad():
        for start in range(0, count, args.batch_size):
            clock.check()
            z = base.torch.rand(min(args.batch_size, count - start), length, model.channels,
                device=device, generator=generator)
            pieces.append(model.R(model.S(model.G(z))).cpu().numpy())
    fake = base.np.concatenate(pieces).astype(base.np.float32)
    fake[..., constant] = 0
    if not base.np.isfinite(fake).all():
        raise FloatingPointError('Generator erzeugte NaN/Inf.')
    return fake






class Sources:
    def __init__(self, base, helper, papers, args, device, clock, cache, schemes):
        self.b, self.h, self.p = base, helper, papers
        self.args, self.device, self.clock = args, device, clock
        self.cache, self.schemes = cache, schemes
        self.trained, self.hit = 0, 0
        self.session = uuid.uuid4().hex

    def fake(self, x, y, groups, minority, scaler, candidate, repeat, fold, inner, slot, model_id):
        self.clock.check()
        if model_id != 'si09_mode_seeking' or candidate.gan_ratio <= 0:
            raise AssertionError('The Real+Noise control must never call GAN training/generation.')
        b, a, p, c = self.b, SimpleNamespace(**vars(self.args)), self.p, candidate
        if x.shape[1:] != (101,1):
            raise ValueError('Der SI09-GAN darf nur [N,101,1] erhalten.')
        a.hidden, a.gan_lr = c.gan_hidden, c.gan_learning_rate
        constant = scaler.span <= 1e-7
        gan_slot = 1 + (slot - 1) % a.gan_seeds
        noise_seed = b.derived_seed(a.seed, repeat, fold, inner, gan_slot, 'si09 GAN input noise', c.gan_input_copies)
        ga = SimpleNamespace(**vars(a)); ga.noise_sigma = a.gan_noise_sigma
        _, gx, gg = b.make_training_sources(x, y, groups, minority, c.gan_input_copies, ga, noise_seed, constant)
        seed = b.derived_seed(a.seed, repeat, fold, inner, gan_slot, 'si09 generator', a.hidden, c.gan_input_copies)
        scheme = scheme_for_candidate(self.schemes, c)
        if scheme.family != family_for_model(model_id):
            raise ValueError('Only the Mode-Seeking arm may train its own one-channel GAN.')
        identity = gan_identity(p, scheme, gx, gg, constant, a.hidden, seed, a)
        key = self.cache.key('gan', identity)
        target = self.cache.root / 'gan' / key / f'GAN_{scheme.gan_cycles:06d}.pt'
        existed = target.is_file()
        context = dict(repeat=repeat, fold=fold, inner_fold=inner, seed_index=slot, gan_seed_slot=gan_slot,
            stage='inner' if inner else 'outer_train', noise_copies=c.gan_input_copies,
            source=c.source, gan_family=scheme.family, gan_input_channels=1, input_space='si09_convnext',
            gan_hidden=a.hidden, gan_learning_rate=a.gan_lr,
            actual_generator_lr=scheme.generator_lr, actual_discriminator_lr=scheme.discriminator_lr,
            n_gan_training_pairs=len(gx), n_gan_patients=len(b.np.unique(gg)))
        rows = []
        fit = p.checkpoint_fit(b, scheme, self.cache, self.session, a.out / 'gan_live_LOCAL_ONLY.jsonl')
        model = fit(gx, gg, a.hidden, a, self.device, self.clock, seed,
            f'{model_id} W{repeat}/F{fold}/I{inner}/S{slot}', history_rows=rows, context=context, constant=constant)
        self.hit += int(existed)
        self.trained += int(not existed)
        reference = x[y == minority]
        count = max(32, int(math.ceil(len(reference))))
        fake = generate_checked(b, model, count, x.shape[1], a, self.device, constant, seed + 1, self.clock)
        del model
        rng = b.np.random.default_rng(seed + 201)
        rg = groups[y == minority]
        subset = rng.choice(len(reference), min(128,len(reference)), replace=False, p=b.row_probabilities(rg))
        report = dict(context, hidden=a.hidden, gan_seed=seed, generated_sequences=len(fake),
            **b.diversity_report(fake, reference[subset]),local_cached_endpoint=existed)
        self.clock.check()
        return fake, rows, report


class Maps:
    """Frozen feature cache with model/scaler/data identity and a bounded RAM LRU."""
    def __init__(self, base, papers, cache, args, device, clock):
        self.b, self.p, self.cache = base, papers, cache
        self.args, self.device, self.clock = args, device, clock
        self.memory, self.bytes, self.hits = OrderedDict(), 0, 0

    def get(self, model, x, factory, model_id):
        b = self.b
        identity = dict(data=self.p.array_digest(x), backbone=factory.fingerprint, model='si09_convnext',
            scaler=self.p.array_digest(model.gan_low.cpu().numpy(), model.gan_denom.cpu().numpy()),
            image_width=model.base.config.image_width, image_height=96, version=VERSION)
        key = self.cache.key('maps', identity)
        self.clock.check()
        if key in self.memory:
            self.memory.move_to_end(key)
            self.hits += 1
            return self.memory[key]
        path = self.cache.path('maps', key)
        if path.is_file():
            saved = self.cache.load(path)
            if saved['identity'] != identity:
                raise ValueError('CNN-Featurecache hat eine andere Identitaet.')
            maps = saved['maps']
            self.hits += 1
        else:
            model.eval()
            chunks, start = [], 0
            print(f'{model_id}: CNN-Maps fuer {len(x)} Paare vorberechnen.', flush=True)
            with b.torch.no_grad():
                while start < len(x):
                    self.clock.check()
                    stop = min(len(x), start + self.args.contact_microbatch)
                    try:
                        value = model.raw_maps(b.torch.from_numpy(x[start:stop]).to(self.device)).cpu()
                    except b.torch.cuda.OutOfMemoryError:
                        gc.collect()
                        b.torch.cuda.empty_cache()
                        if self.args.contact_microbatch == 1:
                            raise
                        self.args.contact_microbatch = max(1, self.args.contact_microbatch // 2)
                        continue
                    chunks.append(value)
                    start = stop
            maps = b.torch.cat(chunks).contiguous()
            self.cache.save(path, dict(identity=identity, maps=maps))
        if maps.dtype != b.torch.float32 or maps.ndim != 4 or len(maps) != len(x) or not b.torch.isfinite(maps).all():
            raise ValueError('Ungueltige eingefrorene CNN-Maps.')
        size = maps.numel() * maps.element_size()
        limit = self.args.map_cache_mb * 1024**2
        if size <= limit:
            while self.memory and self.bytes + size > limit:
                _, old = self.memory.popitem(last=False)
                self.bytes -= old.numel() * old.element_size()
            self.memory[key] = maps
            self.bytes += size
        self.clock.check()
        return maps


def noisy_training(base, args, scaler, x, repeat, fold, inner, slot, sigma):
    seed = base.derived_seed(args.seed, repeat, fold, inner, slot, 'combined direct noise', sigma)
    return base.make_noisy_copies(x, args.direct_noise_copies, sigma, seed, scaler.span <= 1e-7)


def sample_mixed(base, rng, real_maps, noise_maps, fake_maps, y, groups, minority, candidate, batch_size, copies,
                 sampling=None):
    np = base.np
    if sampling is None:
        n_groups = len(np.unique(groups))
        n_min = len(np.unique(groups[y == minority]))
        sampling = (base.row_probabilities(groups),
            draw_probabilities(candidate.gan_ratio, candidate.noise_probability, n_groups, n_min))
    weights, (pr, pn, pg) = sampling
    original = rng.choice(len(y), batch_size, replace=True, p=weights)
    xb, yb = real_maps[original].clone(), y[original].copy()
    u = rng.random(batch_size)
    gan_mask, noise_mask = u < pg, (u >= pg) & (u < pg + pn)
    if noise_mask.any():
        count = int(noise_mask.sum())
        chosen = rng.choice(len(y), count, replace=True, p=weights)
        copy_number = rng.integers(0, copies, count)
        xb[noise_mask] = noise_maps[copy_number * len(y) + chosen]
        yb[noise_mask] = y[chosen]
    if gan_mask.any():
        chosen = rng.integers(0, len(fake_maps), int(gan_mask.sum()))
        xb[gan_mask] = fake_maps[chosen]
        yb[gan_mask] = minority
    return xb, yb, dict(real=int((~(gan_mask | noise_mask)).sum()),
        noise=int(noise_mask.sum()), gan=int(gan_mask.sum()))


def predict_from_maps(base, model, maps, y, groups, args, device, clock):
    model.eval()
    probabilities = []
    with base.torch.no_grad():
        for start in range(0, len(maps), args.batch_size):
            clock.check()
            probabilities.append(model.from_maps(maps[start:start + args.batch_size].to(device)).sigmoid().cpu().numpy())
    p = base.np.concatenate(probabilities)
    unique, inverse, counts = base.np.unique(groups, return_inverse=True, return_counts=True)
    pooled = base.np.bincount(inverse, weights=p) / counts
    truth = base.np.array([y[groups == g][0] for g in unique])
    return truth, pooled


def fit_combined(base, papers, cache, maps, factory, scaler, model_id, candidate, x, y, groups,
                 minority, noise, fake, args, device, clock, seed, tag, valid=None, steps=None):
    np, torch = base.np, base.torch
    began = time.monotonic()
    nsteps = max(args.checkpoints) if steps is None else int(steps)
    identity = dict(model=model_id, backbone=factory.fingerprint, candidate=asdict(candidate),
        seed=int(seed), data=papers.array_digest(x, y, groups),
        scaler=papers.array_digest(scaler.low, scaler.denom), noise=papers.array_digest(noise),
        fake=papers.array_digest(fake), validation=papers.array_digest(*valid) if valid is not None else None,
        steps=nsteps, checkpoints=args.checkpoints if valid is not None else [],
        batch_size=args.batch_size, minority=int(minority))
    path = cache.path('classifier', cache.key('classifier', identity))
    clock.check()
    base.seed_all(seed)
    model = make_probe(base, factory, scaler, candidate, model_id, args._single).to(device)
    if path.is_file():
        saved = cache.load(path)
        if saved['identity'] != identity:
            raise ValueError('Klassifikatorcache hat eine andere Identitaet.')
        model.base.restore_learned(saved['state'])
        cache.hits['classifier'] += 1
        cache.last_fit = dict(cached=True, fit_seconds=saved.get('fit_seconds',0.), seconds_this_session=time.monotonic()-began, trained_steps=nsteps)
        return model.eval(), saved['history'], saved['draw_counts']
    real_maps = maps.get(model, x, factory, model_id)
    noise_maps = maps.get(model, noise, factory, model_id)
    if candidate.gan_ratio == 0 and len(fake):
        raise AssertionError('Real+Noise must never receive synthetic GAN data.')
    fake_maps = maps.get(model, fake, factory, model_id) if candidate.gan_ratio > 0 else real_maps[:0]
    valid_maps = maps.get(model, valid[0], factory, model_id) if valid is not None else None
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=candidate.learning_rate, weight_decay=candidate.weight_decay)
    base.seed_all(seed + 73)
    rng = np.random.default_rng(seed)
    sampling = (base.row_probabilities(groups), draw_probabilities(candidate.gan_ratio,
        candidate.noise_probability, len(np.unique(groups)), len(np.unique(groups[y == minority]))))
    history, counts = [], dict(real=0, noise=0, gan=0)
    for step in range(1, nsteps + 1):
        clock.check()
        model.train()
        xb, yb, drawn = sample_mixed(base, rng, real_maps, noise_maps, fake_maps, y, groups,
            minority, candidate, args.batch_size, args.direct_noise_copies, sampling)
        for kind, number in drawn.items():
            counts[kind] += number
        logits = model.from_maps(xb.to(device)).reshape(-1)
        loss = base.F.binary_cross_entropy_with_logits(logits, torch.as_tensor(yb, dtype=torch.float32, device=device))
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Nichtendlicher Klassifikator-Loss.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        if valid is not None and step in args.checkpoints:
            truth, probability = predict_from_maps(base, model, valid_maps, valid[1], valid[2], args, device, clock)
            history.append(dict(step=step, y=truth, p=probability))
            score = base.metrics(truth, probability, .5)
            print(f'{tag} {step}/{nsteps} Einzelmodell t=0.5: BA={score["balanced_accuracy"]:.3f} '
                f'Accuracy={score["accuracy"]:.3f} Kappa={score["cohen_kappa"]:.3f} '
                f'| {clock.minutes():.1f} min', flush=True)
        elif valid is None and (step == 1 or step % args.log_every == 0 or step == nsteps):
            print(f'{tag} {step}/{nsteps} Loss={loss.item():.4f} | {clock.minutes():.1f} min', flush=True)
    cache.last_fit = dict(cached=False, fit_seconds=time.monotonic()-began, seconds_this_session=time.monotonic()-began, trained_steps=nsteps)
    cache.trained_classifier += 1
    cache.save(path, dict(identity=identity, state=model.base.learned_state(), history=history, draw_counts=counts, fit_seconds=cache.last_fit['fit_seconds']))
    return model.eval(), history, counts


def choose_candidate(base, candidates, histories, args):
    rows, curves = [], []
    for candidate in candidates:
        best, points = base.select_inner_candidate(histories[candidate.key], args.inner_folds,
            args.training_seeds, args.thresholds)
        rows.append(dict(candidate.row(), **best))
        curves.extend(dict(candidate.row(), **point) for point in points)
    # Entire tie break is predeclared and uses no outer outcome.
    def key(row):
        return (row['cohen_kappa'], row['balanced_accuracy'], row['seed_kappa_min'],
            -row['transformer_width'], -row['gan_ratio'], -row['noise_probability'],
            -abs(row['threshold'] - .5), -row['steps'], row['candidate_id'])
    selected = max(rows, key=key)
    candidate = next(c for c in candidates if c.key == selected['candidate_id'])
    return candidate, selected, rows, curves


def train_unit(base, helper, papers, args, device, clock, cache, maps, sources, factory,
               generator_candidates, model_id, unit, x, y, groups, gy, model_fingerprint):
    np = base.np
    repeat, fold = unit['repeat'], unit['fold']
    folder = args.out / model_id / 'units' / f'repeat_{repeat:02d}_fold_{fold:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    minority = int(np.argmin(np.bincount(gy[unit['train']], minlength=2)))
    tables = {name: [] for name in ('inner_candidates', 'inner_checkpoint_metrics',
        'inner_split_sizes', 'generation_diagnostics', 'timegan_training_history', 'classifier_fit_diagnostics')}
    meta = dict(repeat=repeat, fold=fold, model=model_id, method=METHOD, gan_family=family_for_model(model_id))

    def flush():
        for name, rows in tables.items():
            helper.write_csv(folder / (name + '.csv'), rows)
        # Inner scores determine both the outer choice and final recommendation.
        # Verify their persisted row count before proceeding to a later stage.
        for name in ('inner_candidates','classifier_fit_diagnostics'):
            if len(helper.read_csv(folder/(name+'.csv')))!=len(tables[name]):
                raise IOError('Incomplete persisted HPO table: '+name)

    def get_sources(tx, ty, tg, scaler, inner, slot):
        fake_by_key, noise_by_sigma = {}, {}
        for candidate in candidates:
            key = (candidate.source, candidate.gan_input_copies,
                   candidate.gan_hidden, candidate.gan_learning_rate)
            if key not in fake_by_key:
                if candidate.gan_ratio == 0:
                    fake_by_key[key] = np.empty((0,101,1),dtype=np.float32)
                else:
                    fake_by_key[key], rows, report = sources.fake(tx, ty, tg, minority,
                        scaler, candidate, repeat, fold, inner, slot, model_id)
                    tables['timegan_training_history'].extend(dict(row, model=model_id, hpo_stage=meta['hpo_stage']) for row in rows)
                    tables['generation_diagnostics'].append(dict(report, model=model_id, hpo_stage=meta['hpo_stage']))
            if candidate.noise_sigma not in noise_by_sigma:
                noise_by_sigma[candidate.noise_sigma] = noisy_training(base, args, scaler, tx,
                    repeat, fold, inner, slot, candidate.noise_sigma)
        return fake_by_key, noise_by_sigma

    generator_choice = None
    for stage in ('augmentation', 'classifier'):
        if stage == 'augmentation':
            candidates = generator_candidates
            fit_args = SimpleNamespace(**vars(args))
            fit_args.checkpoints = [1 if args.self_test else 300]
        else:
            candidates = head_plan(args.trials, args.seed, args.self_test, model_id, generator_choice)
            fit_args = args
            locked = {augmentation_key(c) for c in candidates}
            if len(locked) != 1:
                raise AssertionError('Stage 2 must lock cycles AND the full augmentation mixture.')
        histories = {c.key: {} for c in candidates}
        meta['hpo_stage'] = stage
        print(f'\n{model_id} W{repeat}/F{fold}: HPO-{stage}, {len(candidates)} Kandidaten; '
            f'{args.inner_folds} innere Folds x {args.training_seeds} Seeds.', flush=True)
        for inner in unit['inner']:
            index = inner['inner_fold']
            if set(inner['train']) & (set(inner['valid']) | set(unit['test'])) or set(inner['valid']) & set(unit['test']):
                raise AssertionError('Ungueltige Patiententrennung.')
            tr, va = np.isin(groups, inner['train']), np.isin(groups, inner['valid'])
            scaler = fit_input_scaler(base, args._single, model_id, x[tr], groups[tr])
            tx, vx = scaler.transform(x[tr]), scaler.transform(x[va])
            valid = (vx, y[va], groups[va])
            tables['inner_split_sizes'].append(dict(meta, inner_fold=index,
                train_patients=len(inner['train']), validation_patients=len(inner['valid']),
                outer_test_patients=len(unit['test'])))
            for slot in range(1, args.training_seeds + 1):
                fake_by_key, noise_by_sigma = get_sources(tx, y[tr], groups[tr], scaler, index, slot)
                seed = base.derived_seed(args.seed, repeat, fold, index, slot, 'classifier')
                for number, candidate in enumerate(candidates, 1):
                    fake = fake_by_key[candidate.source, candidate.gan_input_copies,
                                       candidate.gan_hidden, candidate.gan_learning_rate]
                    noise = noise_by_sigma[candidate.noise_sigma]
                    model, history, _ = fit_combined(base, papers, cache, maps, factory, scaler, model_id,
                        candidate, tx, y[tr], groups[tr], minority, noise, fake, fit_args, device, clock, seed,
                        f'{model_id} W{repeat}/F{fold}/I{index}/S{slot} {stage} {number}/{len(candidates)}', valid=valid)
                    tables['classifier_fit_diagnostics'].append(dict(meta, inner_fold=index, seed_index=slot, **candidate.row(), **cache.last_fit))
                    histories[candidate.key][index - 1, slot - 1] = history
                    del model
                del fake_by_key, noise_by_sigma
            flush()
        candidate, choice, rows, curves = choose_candidate(base, candidates, histories, fit_args)
        tables['inner_candidates'].extend(dict(meta, **row) for row in rows)
        tables['inner_checkpoint_metrics'].extend(dict(meta, **row) for row in curves)
        if stage == 'augmentation':
            generator_choice = candidate
            helper.atomic_json(folder / 'augmentation_selection.json', dict(candidate=asdict(candidate),
                inner_metrics=choice, reference_classifier='fixed CNN-32, lr=.0003, WD=.01, dropout=.2, 300 updates; joint cycles+mixture search',
                selection_complete_before_classifier_hpo=True, no_outer_test_used=True))
            print(f'{model_id} W{repeat}/F{fold} Augmentations-HPO gewaehlt: {candidate.source}, '
                f'hidden={candidate.gan_hidden}, lr={candidate.gan_learning_rate:g}, Kopien={candidate.gan_input_copies}; '
                f'INNER Kappa={choice["cohen_kappa"]:.3f}. Starte Klassifikator-HPO.', flush=True)
        flush()
        del histories
    selection = dict(choice, **meta, mode='real_noise' if candidate.gan_ratio==0 else 'real_noise_mode_seeking', hidden=candidate.gan_hidden,
        ratio=candidate.gan_ratio, noise_copies=args.direct_noise_copies)
    # Commit the choice BEFORE touching or transforming any outer-test sequence.
    helper.atomic_json(folder / 'selection.json', dict(selected={METHOD: selection},
        selection_complete_before_outer_prediction=True, criterion='pooled inner OOF six/all-seed ensemble Cohen kappa',
        candidate=asdict(candidate), augmentation_hpo=asdict(generator_choice), minority_class=minority,
        two_stage_inner_hpo=True, augmentation_locked_before_classifier_hpo=True))
    flush()
    print(f'{model_id} W{repeat}/F{fold} INNER AUSWAHL {candidate.source} '
        f'GAN-Kopien={candidate.gan_input_copies}; Updates={choice["steps"]}; t={choice["threshold"]:.2f}; '
        f'BA={choice["balanced_accuracy"]:.3f} Accuracy={choice["accuracy"]:.3f} Kappa={choice["cohen_kappa"]:.3f}', flush=True)
    tr, te = np.isin(groups, unit['train']), np.isin(groups, unit['test'])
    scaler = fit_input_scaler(base, args._single, model_id, x[tr], groups[tr])
    tx = scaler.transform(x[tr])
    test_x = scaler.transform(x[te])
    truth_expected = gy[np.asarray(unit['test'])]
    predictions, seed_rows = [], []
    for slot in range(1, args.training_seeds + 1):
        if candidate.gan_ratio == 0:
            fake = np.empty((0,101,1),dtype=np.float32)
        else:
            fake, rows, report = sources.fake(tx, y[tr], groups[tr], minority, scaler,
                candidate, repeat, fold, 0, slot, model_id)
            tables['timegan_training_history'].extend(dict(row, model=model_id, hpo_stage='outer_refit') for row in rows)
            tables['generation_diagnostics'].append(dict(report, model=model_id, hpo_stage='outer_refit'))
        noise = noisy_training(base, args, scaler, tx, repeat, fold, 0, slot, candidate.noise_sigma)
        seed = base.derived_seed(args.seed, repeat, fold, 0, slot, 'classifier')
        model, _, counts = fit_combined(base, papers, cache, maps, factory, scaler, model_id, candidate,
            tx, y[tr], groups[tr], minority, noise, fake, args, device, clock, seed,
            f'{model_id} W{repeat}/F{fold}/S{slot} REFIT', steps=choice['steps'])
        test_maps = maps.get(model, test_x, factory, model_id)
        truth, probability = predict_from_maps(base, model, test_maps, y[te], groups[te], args, device, clock)
        if not np.array_equal(truth, truth_expected):
            raise AssertionError('Patientenreihenfolge der aeusseren Prognosen hat sich geaendert.')
        predictions.append(probability)
        seed_rows.append(dict(meta, **candidate.row(), seed_index=slot, classifier_seed=seed,
            steps=choice['steps'], threshold=choice['threshold'],
            **{'drawn_' + k: v for k, v in counts.items()}, **base.metric_record(truth, probability, choice['threshold'])))
        tables['classifier_fit_diagnostics'].append(dict(meta, hpo_stage='outer_refit', inner_fold=0, seed_index=slot, **candidate.row(), **cache.last_fit))
        del model, test_maps, fake, noise
    seeds = np.stack(predictions)
    probability = seeds.mean(axis=0)
    score = base.metric_record(truth_expected, probability, choice['threshold'])
    mix = draw_probabilities(candidate.gan_ratio, candidate.noise_probability,
        len(unit['train']), int(np.sum(gy[unit['train']] == minority)))
    fold_row = dict(meta, **candidate.row(), steps=choice['steps'], threshold=choice['threshold'],
        n_seeds=args.training_seeds, draw_probability_real=mix[0], draw_probability_noise=mix[1],
        draw_probability_gan=mix[2], **score)
    arrays = dict(test_indices=np.asarray(unit['test']), y=truth_expected)
    arrays.update({METHOD + '_seed_p': seeds, METHOD + '_p': probability,
        METHOD + '_binary': (probability >= choice['threshold']).astype(np.int64),
        METHOD + '_threshold': np.asarray(choice['threshold'])})
    path = folder / 'predictions_LOCAL_ONLY.npz'
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    helper.write_csv(folder / 'outer_fold_metrics.csv', [fold_row])
    helper.write_csv(folder / 'outer_seed_metrics.csv', seed_rows)
    flush()
    expected=len(generator_candidates)+args.trials
    if len(helper.read_csv(folder/'inner_candidates.csv'))!=expected:
        raise IOError('A complete outer unit must preserve both complete inner HPO stages.')
    table_hashes={name:helper.sha256_file(folder/(name+'.csv')) for name,rows in tables.items() if rows}
    helper.atomic_json(folder / 'completed.json', dict(repeat=repeat, fold=fold,
        run_fingerprint=model_fingerprint, n_methods=1, n_seeds=args.training_seeds,
        n_test=len(truth_expected), prediction_sha256=helper.sha256_file(path),
        training_table_sha256=table_hashes,completed=True))
    print(f'{model_id} W{repeat}/F{fold} AUSSEN {METHOD}: BA={score["balanced_accuracy"]:.3f} '
        f'Accuracy={score["accuracy"]:.3f} Kappa={score["cohen_kappa"]:.3f}', flush=True)



AGG_TABLES = ('outer_fold_metrics','outer_seed_metrics','inner_candidates',
    'inner_checkpoint_metrics','inner_split_sizes','generation_diagnostics',
    'timegan_training_history','classifier_fit_diagnostics')
ROOT_TABLES = (*AGG_TABLES,'repeat_patient_pooled','repeat_seed_patient_pooled',
    'summary_patient_pooled','common_repeat_summary','selected_configurations','selection_frequency',
    'paired_outer_fold_differences','paired_repeat_differences','hpo_candidate_plan','gan_source_plan',
    'fixed_augmentation_ranking','fixed_augmentation_by_repeat','fixed_cycle_ranking')


def aggregate(base, helper, args, plan, gy, identities):
    tables = {key:[] for key in ROOT_TABLES}
    completed_by_model, repeats_by_model = {}, {}
    for model_id in MODELS:
        fake_args = SimpleNamespace(_run_fingerprint=identities[model_id], training_seeds=args.training_seeds)
        completed = base.completed_units(plan, fake_args, args.out/model_id, gy)
        completed_by_model[model_id] = len(completed)
        for unit,folder,arrays in completed:
            marker=helper.read_json(folder/'completed.json')
            for name,digest in marker.get('training_table_sha256',{}).items():
                path=folder/(name+'.csv')
                if not path.is_file() or helper.sha256_file(path)!=digest:
                    raise ValueError('Committed HPO table changed: '+folder.name+'/'+name)
            for name in AGG_TABLES:
                tables[name].extend(helper.read_csv(folder/(name+'.csv')))
            tables['selected_configurations'].append(helper.read_json(folder/'selection.json')['selected'][METHOD])
        full_repeats = []
        for repeat in repeat_ids(args):
            units = [(u,a) for u,_,a in completed if u['repeat']==repeat]
            if not units:
                continue
            indices = base.np.concatenate([a['test_indices'] for _,a in units])
            if len(base.np.unique(indices)) != len(indices):
                raise AssertionError('Doppelte Patienten im aeusseren OOF.')
            is_complete = len(units)==args.folds
            if is_complete:
                if not base.np.array_equal(base.np.sort(indices),base.np.arange(len(gy))):
                    raise AssertionError('Vollstaendige CV deckt nicht alle Patienten ab.')
                full_repeats.append(repeat)
            truth = base.np.concatenate([a['y'] for _,a in units])
            probability = base.np.concatenate([a[METHOD+'_p'] for _,a in units])
            binary = base.np.concatenate([a[METHOD+'_binary'] for _,a in units])
            info = dict(model=model_id,gan_family=family_for_model(model_id),method=METHOD,repeat=repeat,
                repeat_complete=is_complete,n_completed_folds=len(units),n_unique_test_patients=len(truth))
            tables['repeat_patient_pooled'].append(dict(info,**base.pooled_metrics(truth,probability,binary)))
            for slot in range(args.training_seeds):
                sp = base.np.concatenate([a[METHOD+'_seed_p'][slot] for _,a in units])
                sb = base.np.concatenate([(a[METHOD+'_seed_p'][slot]>=a[METHOD+'_threshold']).astype(int) for _,a in units])
                tables['repeat_seed_patient_pooled'].append(dict(info,seed_index=slot+1,**base.pooled_metrics(truth,sp,sb)))
        repeats_by_model[model_id] = full_repeats
        full = [r for r in tables['repeat_patient_pooled'] if r['model']==model_id and r['repeat_complete']]
        if full:
            tables['summary_patient_pooled'].append(summary_row(base,args,model_id,full,len(completed),len(gy)))
    common = sorted(set.intersection(*(set(repeats_by_model[name]) for name in MODELS)))
    if common:
        for model_id in MODELS:
            full = [r for r in tables['repeat_patient_pooled'] if r['model']==model_id and r['repeat'] in common]
            tables['common_repeat_summary'].append(summary_row(base,args,model_id,full,len(common)*args.folds,len(gy)))
    # Each comparison uses only matching, completely committed units of that PAIR.
    for name,output,keys in (('outer_fold_metrics','paired_outer_fold_differences',('repeat','fold')),
                            ('repeat_patient_pooled','paired_repeat_differences',('repeat',))):
        by_key = {}
        for row in tables[name]:
            if name.startswith('repeat') and not row['repeat_complete']:
                continue
            by_key.setdefault(tuple(int(row[k]) for k in keys),{})[row['model']]=row
        for key,pair in sorted(by_key.items()):
            for reference,target in combinations(MODELS,2):
                if reference in pair and target in pair:
                    tables[output].append(dict(zip(keys,key),target=target,reference=reference,
                        **{'delta_'+metric:float(pair[target][metric])-float(pair[reference][metric]) for metric in base.METRICS}))
    for model_id in MODELS:
        selected = [r for r in tables['selected_configurations'] if r['model']==model_id]
        for cid in sorted({r['candidate_id'] for r in selected}):
            chosen = [r for r in selected if r['candidate_id']==cid]
            tables['selection_frequency'].append(dict(model=model_id,gan_family=family_for_model(model_id),
                candidate_id=cid,source=chosen[0]['source'],gan_input_copies=chosen[0]['gan_input_copies'],
                selected_outer_folds=len(chosen),total_completed_outer_folds=len(selected)))
    recommendation = fixed_recommendation(base, helper, args, tables, common, gy)
    for name in ROOT_TABLES:
        if name not in ('hpo_candidate_plan','gan_source_plan'):
            helper.write_csv(args.out/(name+'.csv'),tables[name])
    return dict(completed_by_model=completed_by_model,completed_repeats=repeats_by_model,
        common_completed_repeats=common,summary=tables['summary_patient_pooled'],
        fixed_recommendation_status=recommendation['status'])


def fixed_recommendation(base, helper, args, tables, common, gy):
    """Rank complete, matched INNER searches; outer scores never enter here.

    Each repetition has equal weight after averaging its outer training
    contexts. SD describes dependent repetitions, not independent evidence.
    Stage-1 candidates share a fixed reference head. The recommendation is
    a development setting, not an independently evaluated fixed model.
    """
    np = base.np
    ranking, per_repeat, cycle_ranking = [], [], []
    recommendation = dict(version=VERSION, status='waiting_for_complete_paired_repetitions',
        complete_paired_repeat_indices=list(common), planned_repeat_indices=repeat_ids(args),
        required_complete_repetitions=2 if args.self_test else 3,
        selection_uses_outer_test_scores=False, reference_head_conditioned=True,
        selection_rule='mean inner ensemble Kappa, SD across repetition means, BA, minimum Kappa, cycles, r/q/sigma',
        interpretation='development recommendation; outer CV metrics evaluate the per-fold adaptive HPO pipeline',
        independent_validation_of_fixed_configuration=False,
        fixed_fields=['GAN production cycles','GAN minority sampling ratio r','direct noise probability q','direct noise sigma'],
        classifier_and_threshold='remain training-data HPO choices; not globally frozen here',
        configurations={})
    for model_id in MODELS:
        candidates = generator_plan(args, model_id)
        rows = [r for r in tables['inner_candidates'] if r['model']==model_id
            and r['hpo_stage']=='augmentation' and int(r['repeat']) in common]
        if not common:
            continue
        expected = {(rep,fold,c.key) for rep in common for fold in range(1,args.folds+1) for c in candidates}
        lookup = {}
        for row in rows:
            key = int(row['repeat']),int(row['fold']),row['candidate_id']
            if key in lookup:
                raise ValueError('Duplicate inner augmentation candidate in a committed unit.')
            lookup[key] = row
        if set(lookup) != expected:
            raise ValueError('Fixed recommendation requires ALL stage-1 candidates on identical complete contexts.')
        for candidate in candidates:
            context = [lookup[rep,fold,candidate.key] for rep in common for fold in range(1,args.folds+1)]
            means, ba_means = [], []
            for rep in common:
                block = [lookup[rep,fold,candidate.key] for fold in range(1,args.folds+1)]
                mean = float(np.mean([float(r['cohen_kappa']) for r in block]))
                ba = float(np.mean([float(r['balanced_accuracy']) for r in block]))
                means.append(mean); ba_means.append(ba)
                per_repeat.append(dict(model=model_id, gan_family=family_for_model(model_id),
                    repeat=rep, candidate_id=candidate.key, source=candidate.source,
                    gan_cycles=0 if candidate.gan_ratio==0 else int(candidate.source.rsplit('_',1)[1]),
                    gan_ratio=candidate.gan_ratio,noise_probability=candidate.noise_probability,
                    noise_sigma=candidate.noise_sigma,n_outer_training_contexts=len(block),
                    mean_inner_cohen_kappa=mean,mean_inner_balanced_accuracy=ba))
            chosen = [r for r in tables['selected_configurations'] if r['model']==model_id
                and int(r['repeat']) in common and augmentation_key(r)==augmentation_key(candidate)]
            ranking.append(dict(model=model_id,gan_family=family_for_model(model_id),**candidate.row(),
                gan_cycles=0 if candidate.gan_ratio==0 else int(candidate.source.rsplit('_',1)[1]),
                n_complete_paired_repetitions=len(common),n_outer_training_contexts=len(context),
                mean_inner_cohen_kappa=float(np.mean(means)),
                sd_repeat_mean_inner_cohen_kappa=float(np.std(means,ddof=1)) if len(means)>1 else 0.,
                minimum_context_inner_cohen_kappa=min(float(r['cohen_kappa']) for r in context),
                mean_inner_balanced_accuracy=float(np.mean(ba_means)),
                selected_augmentation_outer_contexts=len(chosen)))
        arm = [r for r in ranking if r['model']==model_id]
        arm.sort(key=lambda r:(-r['mean_inner_cohen_kappa'],r['sd_repeat_mean_inner_cohen_kappa'],
            -r['mean_inner_balanced_accuracy'],-r['minimum_context_inner_cohen_kappa'],
            r['gan_cycles'],r['gan_ratio'],r['noise_probability'],r['noise_sigma'],r['candidate_id']))
        for rank,row in enumerate(arm,1):
            row['rank_within_arm'] = rank
        best = arm[0]
        pr,pn,pg = draw_probabilities(best['gan_ratio'],best['noise_probability'],len(gy),int(np.bincount(gy).min()))
        recommendation['configurations'][model_id] = dict(
            source=best['source'],gan_cycles=best['gan_cycles'],gan_ratio=best['gan_ratio'],
            noise_probability=best['noise_probability'],noise_sigma=best['noise_sigma'],
            direct_noise_copies=args.direct_noise_copies,gan_training_noise_copies=best['gan_input_copies'],
            gan_training_noise_sigma=.02 if best['gan_ratio'] else None,
            gan_hidden=best['gan_hidden'] if best['gan_ratio'] else None,
            gan_layers=args.layers if best['gan_ratio'] else None,
            gan_learning_rate=best['gan_learning_rate'] if best['gan_ratio'] else None,
            mode_seeking_lambda=1. if best['gan_ratio'] else None,
            input_channels=1,curve_points=101,
            selection_mean_inner_kappa=best['mean_inner_cohen_kappa'],
            sd_repeat_mean_inner_kappa=best['sd_repeat_mean_inner_cohen_kappa'],
            mean_kappa_margin_to_runner_up=(best['mean_inner_cohen_kappa']-arm[1]['mean_inner_cohen_kappa']) if len(arm)>1 else None,
            expected_sampling_on_full_cohort=dict(real=pr,noise=pn,gan=pg),
            candidate_id=best['candidate_id'])
    if common:
        complete_all=set(common)==set(repeat_ids(args))
        enough=len(common)>=recommendation['required_complete_repetitions']
        recommendation['status']='ready_development_recommendation' if complete_all and enough else 'provisional'
        for cycle in args.gan_cycles:
            subset=[r for r in per_repeat if r['model']=='si09_mode_seeking' and r['gan_cycles']==cycle]
            if not subset:
                continue
            means=[float(np.mean([r['mean_inner_cohen_kappa'] for r in subset if r['repeat']==rep])) for rep in common]
            cycle_ranking.append(dict(model='si09_mode_seeking',gan_cycles=cycle,
                n_complete_paired_repetitions=len(common),n_mixtures_per_cycle=len(subset)//len(common),
                mean_inner_kappa_over_all_mixtures=float(np.mean(means)),
                sd_repeat_mean_inner_kappa=float(np.std(means,ddof=1)) if len(means)>1 else 0.,
                interpretation='marginal cycle comparison; final recommendation ranks joint cycles+mixture configurations'))
    tables['fixed_augmentation_ranking']=sorted(ranking,key=lambda r:(r['model'],r['rank_within_arm']))
    tables['fixed_augmentation_by_repeat']=per_repeat
    tables['fixed_cycle_ranking']=cycle_ranking
    helper.atomic_json(args.out/'fixed_augmentation_recommendation.json',recommendation)
    return recommendation


def summary_row(base,args,model_id,full,n_folds,n_patients):
    row = dict(model=model_id,gan_family=family_for_model(model_id),method=METHOD,
        n_completed_repeats=len(full),n_completed_outer_folds=n_folds,n_unique_patients=n_patients,
        n_seeds_per_ensemble=args.training_seeds,
        interpretation='exploratory development on previously examined cohort; dependent folds and seeds')
    for metric in base.METRICS:
        values = [r[metric] for r in full]
        row[metric] = float(base.np.mean(values))
        row['sd_repeat_'+metric] = float(base.np.std(values,ddof=1)) if len(values)>1 else None
    return row


def aggregate_zip(args):
    target = args.out / 'si09_modeseeking_noise_aggregate.zip'
    temporary = target.with_name(target.name + '.writing')
    allowed = [name + '.csv' for name in ROOT_TABLES] + ['protocol.json', 'run_status.json', 'fixed_augmentation_recommendation.json']
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in allowed:
            path = args.out / name
            if path.is_file():
                archive.write(path, name)
    os.replace(temporary, target)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', nargs='?', type=Path, default=Path('CSV_Export'))
    parser.add_argument('--base-script', type=Path, default=Path(__file__).with_name(BASE_FILE))
    parser.add_argument('--compare-script', type=Path, default=Path(__file__).with_name(COMPARE_FILE))
    parser.add_argument('--papers-script', type=Path, default=Path(__file__).with_name(PAPERS_FILE))
    parser.add_argument('--si09-script', type=Path, default=Path(__file__).with_name(SINGLE_FILE))



    parser.add_argument('--output-dir', '--out', dest='out', type=Path, default=Path('results_si09_modeseeking_noise_24h_v2'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--weights-dir', type=Path, default=Path('pretrained_weights_v36'))
    parser.add_argument('--hours', type=float, default=24.,
        help='Advisory runtime target only. NEVER abort after this number of hours.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--first-repeat', type=int, default=2, help='Default 2 starts NEW splits/seeds; 3 repetitions use indices 2,3,4.')
    parser.add_argument('--folds', '--outer-folds', type=int, default=3)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--training-seeds', type=int, default=6)
    parser.add_argument('--gan-seeds', type=int, default=3,
        help='GAN-Seeds je Trainingsfold; Kopf-Seeds nutzen zyklisch diese Pools (Standard 3 GAN / 6 Koepfe).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=12, help='4..24 common classifier profiles AFTER joint augmentation HPO.')
    parser.add_argument('--gan-cycles', default='4000,5000,6000,7000,8000', help='Selection from 4000,5000,6000,7000,8000; training prefixes continue.')
    parser.add_argument('--gan-ratios', default='0.25,0.5,1')
    parser.add_argument('--noise-probabilities', default='0.25,0.5,0.75')
    parser.add_argument('--noise-sigmas', default='0.01,0.02')
    parser.add_argument('--checkpoints', default='150,300,600')
    parser.add_argument('--thresholds', default=','.join(f'{i/100:.2f}' for i in range(10, 91, 5)))
    parser.add_argument('--contact-microbatch', type=int, default=16)
    parser.add_argument('--map-cache-mb', type=int, default=512)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--log-every', type=int, default=250)
    parser.add_argument('--force-threshold', type=float, default=0.)
    parser.add_argument('--contact-duration-tolerance', type=float, default=.25)
    parser.add_argument('--contact-zero-tolerance', type=float, default=0.)
    parser.add_argument('--min-si-pairs-per-subject', type=int, default=3)
    parser.add_argument('--show-import-log', action='store_true')
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    try:
        args.gan_cycles = sorted(set(int(v) for v in args.gan_cycles.split(',')))
        args.gan_ratios = sorted(set(float(v) for v in args.gan_ratios.split(',')))
        args.noise_probabilities = sorted(set(float(v) for v in args.noise_probabilities.split(',')))
        args.noise_sigmas = sorted(set(float(v) for v in args.noise_sigmas.split(',')))
        args.checkpoints = sorted(set(int(v) for v in args.checkpoints.split(',')))
        args.thresholds = sorted(set(float(v) for v in args.thresholds.split(',')))
    except ValueError:
        parser.error('Checkpoints und Schwellen als kommagetrennte Zahlenlisten angeben.')
    if not args.gan_cycles or not set(args.gan_cycles) <= {4000,5000,6000,7000,8000}:
        parser.error('--gan-cycles must select from 4000,5000,6000,7000,8000.')
    if args.trials < 4 or args.trials > 24:
        parser.error('--trials must be between 4 and 24.')
    if not args.checkpoints or min(args.checkpoints) <= 0:
        parser.error('--checkpoints muss positive Update-Zahlen enthalten.')
    if not args.thresholds or any(not math.isfinite(v) or not 0 < v < 1 for v in args.thresholds):
        parser.error('--thresholds muss endliche Werte innerhalb (0,1) enthalten.')
    positive = (args.hours, args.repeats, args.training_seeds, args.contact_microbatch,
        args.map_cache_mb, args.threads, args.checkpoint_every, args.log_every, args.min_si_pairs_per_subject)
    if not all(math.isfinite(v) and v > 0 for v in positive) or min(args.folds, args.inner_folds) < 2:
        parser.error('Budgets/Anzahlen muessen positiv und endlich sein; beide Foldzahlen >=2.')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed muss in 0..2**32-1 liegen.')
    if any(not math.isfinite(v) or v < 0 for v in (args.force_threshold,
            args.contact_duration_tolerance, args.contact_zero_tolerance)):
        parser.error('Kontakt-QC-Parameter muessen endlich und nichtnegativ sein.')
    for values,allowed,label in ((args.gan_ratios,{.25,.5,1.},'gan-ratios'),
            (args.noise_probabilities,{.25,.5,.75},'noise-probabilities'),
            (args.noise_sigmas,{.01,.02},'noise-sigmas')):
        if not values or not set(values)<=allowed:
            parser.error('--'+label+' contains values outside the predeclared grid.')
    if args.first_repeat<1:
        parser.error('--first-repeat must be positive.')
    # Shared training settings preserve the previous one-dimensional SI09 conditions.
    args.hidden, args.layers, args.batch_size = 16, 2, 16
    args.ae_steps = args.supervisor_steps = 500
    args.gan_lr, args.gan_noise_sigma, args.diagnostic_every = .001, .02, 250
    args.direct_noise_copies = 3
    args._checkpoint_every = args.checkpoint_every
    if args.self_test:
        args.folds = args.inner_folds = 2
        args.repeats, args.training_seeds, args.trials = 2, 2, 4
        args.gan_cycles = [4000,8000]
        args.gan_ratios, args.noise_probabilities, args.noise_sigmas = [.25,1.],[.5,.75],[.02]
        args.gan_seeds = 1
        args.hidden, args.batch_size = 4, 4
        args.ae_steps = args.supervisor_steps = 1
        args.diagnostic_every = args._checkpoint_every = 1
        args.checkpoints, args.thresholds = [1, 2], [.25, .5, .75]
    if not 1 <= args.gan_seeds <= args.training_seeds:
        parser.error('--gan-seeds muss zwischen 1 und --training-seeds liegen.')
    if args.training_seeds % args.gan_seeds:
        parser.error('--training-seeds muss fuer gleich gewichtete GAN-Pools durch --gan-seeds teilbar sein.')
    return args



def public_protocol(args,candidates,schemes):
    n_inner=args.repeats*args.folds*args.inner_folds*args.training_seeds
    n_outer=args.repeats*args.folds*args.training_seeds
    return dict(version=VERSION,category=METHOD,models=list(MODELS),gan_families=['mode_seeking'],
        arms={'si09_noise_only':'real + direct noise; NO GAN trained or sampled',
              'si09_mode_seeking':'real + direct noise + Mode-Seeking GAN minority sequences'},
        input='SI09 only; force_si_max_dom_minus_other; SI15 zero-based channel index 8',
        sequence_shape=[101,1],image_shape=[1,96,160],frozen_backbone=True,
        si09_stem_adaptation='sum pretrained RGB kernels; retain bias; fixed grayscale normalization',
        source_model_original_task='angle_regression; adapted to binary BCE classification',
        repeats=args.repeats,repeat_indices=repeat_ids(args),outer_folds=args.folds,inner_folds=args.inner_folds,
        training_seeds=args.training_seeds,gan_seeds=args.gan_seeds,
        gan_seed_mapping='gan_slot=1+(classifier_slot-1)%gan_seeds; independent classifier/direct-noise seed per head',
        planned_unique_inner_gan_trajectories=args.repeats*args.folds*args.inner_folds*args.gan_seeds,
        planned_unique_outer_gan_trajectories=args.repeats*args.folds*args.gan_seeds,
        gan_production_cycles=args.gan_cycles,gan_training_profiles=[asdict(s) for s in schemes.values()],
        augmentation_candidate_counts={name:len(c) for name,c in candidates.items()},
        augmentation_candidate_plan={name:[c.row() for c in choices] for name,choices in candidates.items()},
        classifier_candidate_count_per_arm=args.trials,
        classifier_profiles={name:[dict(c.row(),source='inner-selected augmentation; all augmentation fields locked')
            for c in head_plan(args.trials,args.seed,args.self_test,name,choices[0])] for name,choices in candidates.items()},
        classifier_checkpoints=args.checkpoints,thresholds=args.thresholds,
        hpo_stage_1='JOINT cycles/r/q/noise-sigma grid, common fixed CNN-32 reference at 300 updates; inner OOF ensemble Kappa',
        hpo_stage_2='lock complete augmentation; common classifier/checkpoint/threshold HPO, inner ensemble Kappa',
        fixed_gan_parameters=dict(hidden=4 if args.self_test else 16,layers=2,training_noise_copies=1 if args.self_test else 3,
            training_noise_sigma=.02,base_lr=.001,ae_steps=args.ae_steps,supervisor_steps=args.supervisor_steps),
        fixed_paper_parameters=dict(mode_seeking_lambda=1.,generator_updates_per_cycle=2,
            embedder_updates_per_cycle=2,packing=1,generator_lr=.001,discriminator_lr=.001),
        augmentation='both arms: real+noise; GAN minority examples only in Mode-Seeking arm',
        sampling='p_GAN=r*n_min/(n+r*n_min); p_noise=(1-p_GAN)*q; p_real=(1-p_GAN)*(1-q); control r=0',
        direct_noise_copies=args.direct_noise_copies,gan_ratios=args.gan_ratios,
        noise_probabilities=args.noise_probabilities,noise_sigmas=args.noise_sigmas,
        objective='patient-pooled inner OOF ensemble Cohen kappa',
        fixed_recommendation=dict(source='ALL stage-1 inner OOF scores from common COMPLETE repetitions only',
            ranking='mean inner Kappa; then lower SD across repetition means, higher BA, higher minimum, fewer cycles, smaller r/q/sigma',
            reads_outer_scores=False,ready_after_all_planned_and_at_least=2 if args.self_test else 3,
            meaning='development augmentation recommendation; outer results evaluate per-fold adaptive HPO, NOT this globally chosen setting'),
        workload=workload_estimate(args,candidates),
        planned_inner_classifier_fits=n_inner*sum(len(c)+args.trials for c in candidates.values()),
        planned_outer_classifier_fits=n_outer*len(MODELS),advisory_runtime_hours=args.hours,
        abort_at_time_target=False,continue_until_all_planned_units_complete=True,
        complete_within_budget_guaranteed=False,previous_results_read=False,
        split_seed=args.seed,same_splits_and_classifier_seeds_for_both_arms=True,
        family_order='alternate by planned outer unit; independent of outcomes',
        precision='float32',exploratory=True,independent_confirmation=False,technical_self_test=args.self_test,
        caveat='same previously examined cohort; folds/repeats/seeds are dependent; no independent confirmation',
        source_paper='https://arxiv.org/abs/1903.05628')


def main(argv=None):
    began=time.monotonic()
    args=parse_args(argv)
    candidates={name:generator_plan(args,name) for name in MODELS}
    papers=load_module(args.papers_script,'_smn_papers')
    schemes=source_schemes(papers,args)
    protocol=public_protocol(args,candidates,schemes)
    if args.plan_only:
        print(json.dumps(protocol,indent=2,ensure_ascii=False)); return 0
    helper=load_module(args.compare_script,'_smn_utils')
    helper.write_csv=write_csv
    base=helper.load_base(args.base_script)
    args._single=load_module(args.si09_script,'_smn_single')
    base.METHODS=(METHOD,)
    clock=AdvisoryClock(args.hours,start=began)
    base.torch.set_num_threads(args.threads)
    if args.device=='cuda' and not args.check_data:
        if not base.torch.cuda.is_available():
            raise ValueError('CUDA-PyTorch fehlt in diesem Python; vorhandene CUDA-venv verwenden.')
        device=base.torch.device('cuda')
        base.torch.backends.cudnn.benchmark=False
        base.torch.backends.cudnn.deterministic=True
        base.torch.backends.cuda.matmul.allow_tf32=False
        base.torch.backends.cudnn.allow_tf32=False
        probe=base.nn.GRU(1,4,batch_first=True).to(device)
        probe(base.torch.ones(2,8,1,device=device))[0].sum().backward()
        base.torch.cuda.synchronize(); del probe
        print('GPU:',base.torch.cuda.get_device_name(0),'| float32',flush=True)
    else:
        device=base.torch.device('cpu')
    temporary=tempfile.TemporaryDirectory(prefix='smn_synthetic_',dir=Path.cwd()) if args.self_test else None
    try:
        if temporary is not None:
            args.data_dir=Path(temporary.name); base.create_test_csvs(args.data_dir)
            print('TECHNISCHER SELBSTTEST: synthetische CSVs, kleines Test-CNN, kurze Fits; keine Leistungsbewertung.',flush=True)
        x,y,groups,gy,names,audit=base.load_si7_csv(args)
        base.validate_dataset(x,y,groups,gy,names)
        sx,sy,sg=load_si09(base,args._single,args,gy,names)
        plan=split_plan(base,gy,args)
        data_fingerprint=base.dataset_fingerprint(x,y,groups,gy,names)
        si09_fingerprint=base.dataset_fingerprint(sx,sy,sg,gy,names)
        protocol.update(n_unique_patients=len(gy),patient_class_counts=base.np.bincount(gy,minlength=2).tolist())
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if not k.startswith('_')}
        operational={'data_dir','base_script','compare_script','papers_script','si09_script',
            'out','weights_dir','hours','resume','threads','contact_microbatch','map_cache_mb','log_every',
            'show_import_log','check_data','plan_only'}
        identity=dict(version=VERSION,data=data_fingerprint,si09_data=si09_fingerprint,
            code={name:helper.sha256_file(path) for name,path in
                (('runner',Path(__file__)),('base',args.base_script),('compare',args.compare_script),
                 ('papers',args.papers_script),('si09_model',args.si09_script))},
            settings={k:v for k,v in settings.items() if k not in operational},
            candidates={name:[c.row() for c in choices] for name,choices in candidates.items()},
            schemes=[asdict(s) for s in schemes.values()],torch_version=str(base.torch.__version__),device=args.device)
        fingerprint=helper.digest(identity)
        manifest_path=args.out/'run_manifest_LOCAL_ONLY.json'
        if args.out.exists() and not args.out.is_dir():
            raise ValueError('--output-dir muss ein Ordner sein.')
        if args.resume:
            if not manifest_path.is_file() or helper.read_json(manifest_path).get('fingerprint')!=fingerprint:
                raise ValueError('Wiederaufnahme abgelehnt: Daten, Code, Suchraum, Folds oder Seeds unterscheiden sich.')
        elif args.out.is_dir() and any(args.out.iterdir()):
            raise ValueError('Ausgabeordner ist belegt; --resume oder neuen --output-dir verwenden.')
        args.out.mkdir(parents=True,exist_ok=True)
        if not args.resume:
            helper.atomic_json(manifest_path,dict(fingerprint=fingerprint,identity=identity))
        helper.atomic_json(args.out/'protocol.json',protocol)
        helper.write_csv(args.out/'hpo_candidate_plan.csv',
            [dict(c.row(),model=name,gan_family=family_for_model(name),hpo_stage='augmentation')
             for name,choices in candidates.items() for c in choices]+
            [dict(c.row(),model=name,gan_family=family_for_model(name),hpo_stage='classifier_template',source='inner_selected_augmentation')
             for name,choices in candidates.items() for c in head_plan(args.trials,args.seed,args.self_test,name,choices[0])])
        helper.write_csv(args.out/'gan_source_plan.csv',[asdict(s) for s in schemes.values()])
        helper.write_csv(args.out/'import_audit_aggregate.csv',audit)
        base.export_manifest(plan,gy,names,data_fingerprint,args.out)
        if args.check_data:
            print('Import, Kohorte und Folds geprueft. Kein Training. Zum Start --resume ohne --check-data verwenden.',flush=True)
            return 0
        cache=papers.FitCache(base,helper,args.out/'fit_cache_LOCAL_ONLY',fingerprint)
        cache.trained_classifier=0
        sources=Sources(base,helper,papers,args,device,clock,cache,schemes)
        maps=Maps(base,papers,cache,args,device,clock)
        identities={name:helper.digest([fingerprint,name]) for name in MODELS}
        status,code,state='running',0,dict(completed_by_model={},completed_repeats={},summary=[])
        def save():
            nonlocal state
            state=aggregate(base,helper,args,plan,gy,identities)
            helper.atomic_json(args.out/'run_status.json',dict(version=VERSION,status=status,category=METHOD,
                planned_outer_folds_per_family=len(plan),completed_outer_folds=state['completed_by_model'],
                completed_repeats=state['completed_repeats'],common_completed_repeats=state['common_completed_repeats'],
                advisory_runtime_hours=args.hours,abort_at_time_target=False,
                runtime_minutes_this_session=clock.minutes(),
                training_seeds=args.training_seeds,gan_seeds=args.gan_seeds,trials_per_arm=args.trials,fixed_recommendation_status=state['fixed_recommendation_status'],
                reused_local_gan_endpoints=sources.hit,
                trained_or_resumed_gan_endpoints=sources.trained,reused_classifier_fits=cache.hits['classifier'],
                trained_classifier_fits=cache.trained_classifier,reused_map_requests=maps.hits,
                n_unique_patients=len(gy),exploratory=True))
            aggregate_zip(args)
        try:
            clock.check()
            rgb=base.BackboneFactory(args,synthetic=args.self_test)
            factory=si09_factory(base,rgb)
            for model_id in MODELS:
                path=args.out/model_id/'backbone.json'; path.parent.mkdir(parents=True,exist_ok=True)
                info=dict(provenance=factory.provenance,fingerprint=factory.fingerprint,cnn_input_channels=1)
                if path.is_file() and helper.read_json(path)!=info:
                    raise ValueError('Backbone unterscheidet sich vom begonnenen Lauf.')
                helper.atomic_json(path,info)
            save()
            if not args.self_test:
                estimate=protocol['workload']
                print(f'GTX 1070 Arbeitsmenge: {estimate["gan_cycles_at_maximum_outer_duration"]:,} GAN-Zyklen '
                    f'und {estimate["total_classifier_updates"]:,} Klassifikator-Updates; '
                    f'geschaetzt {estimate["estimated_total_hours_including_reserve"]:.1f} h inklusive 12% Reserve. '
                    'Beenden nach vollstaendiger Arbeit, OHNE Zeitabbruch.',flush=True)
            print(f'SI09 Mode Seeking vs Real+Noise V{VERSION}: Wiederholungen={repeat_ids(args)}; GAN-Zyklen={args.gan_cycles}; '
                f'{args.trials} Kopfprofile je Arm nach gemeinsamer Zyklen-/Mischsuche; {args.folds}x{args.inner_folds} Nested, '
                f'{args.training_seeds} Kopf-Seeds / {args.gan_seeds} GAN-Seeds; '
                f'Zeitplanung={args.hours:g} h OHNE Zeitabbruch.',flush=True)
            for unit in plan:
                offset=((unit['repeat']-args.first_repeat)*args.folds+unit['fold']-1)%len(MODELS)
                order=MODELS[offset:]+MODELS[:offset]
                for model_id in order:
                    clock.check()
                    finished=base.completed_units(plan,SimpleNamespace(_run_fingerprint=identities[model_id],
                        training_seeds=args.training_seeds),args.out/model_id,gy)
                    if any((u['repeat'],u['fold'])==(unit['repeat'],unit['fold']) for u,_,_ in finished):
                        print(f'{model_id} W{unit["repeat"]}/F{unit["fold"]}: fertig gespeichert.',flush=True); continue
                    train_unit(base,helper,papers,args,device,clock,cache,maps,sources,factory,candidates[model_id],
                        model_id,unit,sx,sy,sg,gy,identities[model_id])
                    gc.collect()
                    if device.type=='cuda': base.torch.cuda.empty_cache()
                    save()
            status='complete'
        except KeyboardInterrupt as error:
            status,code='interrupted',2
            print(str(error) or 'Unterbrochen; gespeicherte Fortschritte bleiben erhalten.',flush=True)
        except BaseException:
            status,code='failed',2
            raise
        finally:
            save()
            print(f'\nGesamtstatus={status}; {clock.minutes():.1f} min in dieser Sitzung.',flush=True)
            for row in state['summary']:
                print(f'{row["model"]:23s} BA={row["balanced_accuracy"]:.3f} Accuracy={row["accuracy"]:.3f} '
                    f'Kappa={row["cohen_kappa"]:.3f} ({row["n_completed_repeats"]} vollstaendige Wiederholung(en))',flush=True)
            if not state['summary']:
                print('Noch keine vollstaendige CV-Wiederholung. Fertige Teil-Folds stehen separat in outer_fold_metrics.csv.',flush=True)
            print('Gemeinsame vollstaendige Wiederholungen beider Ansaetze:',state['common_completed_repeats'],flush=True)
            print('Feste Augmentations-Empfehlung:',state['fixed_recommendation_status'],flush=True)
            fixed=helper.read_json(args.out/'fixed_augmentation_recommendation.json')
            for name,config in fixed['configurations'].items():
                print(f'{name}: feste Entwicklungsempfehlung GAN-Zyklen={config["gan_cycles"]}; '
                    f'r={config["gan_ratio"]:g}, q={config["noise_probability"]:g}, '
                    f'Noise-Sigma={config["noise_sigma"]:g}.',flush=True)
            print('Ausgabe:',args.out.resolve(),flush=True)
            print('Aggregierter ZIP zum Teilen:',(args.out/'si09_modeseeking_noise_aggregate.zip').resolve(),flush=True)
        return code
    finally:
        if temporary is not None: temporary.cleanup()


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,FileNotFoundError,PermissionError) as error:
        raise SystemExit(str(error)) from error
