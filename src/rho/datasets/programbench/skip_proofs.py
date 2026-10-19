"""Verifier-only validation of explicit, immutable paired terminal-skip reviews.

Never infer causes from test names, award passes, remove denominator entries,
or accept missing test executions. No path from this module to Solver inputs.
"""
import json
from pathlib import Path

from rho.datasets.programbench.grading import sha256


def validate_terminal_skip_review(context, raw):
    def require(value, reason):
        if not value:
            raise ValueError(reason)

    path = Path(context['path'])
    require(sha256(path) == context['sha256'], 'terminal skip review digest changed')
    proof = json.loads(path.read_text())
    require(proof.get('schema') == 1, 'unsupported terminal skip review')
    task = context['task_id']
    require(proof['task_id'] == task, 'terminal skip review task mismatch')
    if raw.get('instance_id') is not None:
        require(raw['instance_id'] == task, 'evaluation task mismatch')
    require(proof['submission_sha256'] == sha256(Path(context['submission_path'])), 'review submission mismatch')
    require(proof['executable_sha256'] == raw.get('executable_hash'), 'review executable mismatch')
    archives = proof['official_archive_sha256']
    require(bool(archives), 'review has no immutable official archives')
    for key, digest in archives.items():
        require(key.startswith(task+'/tests/') and context['official_archive_sha256'].get(key) == digest,
                'review official archive mismatch')
    evidence = proof['evidence_files']
    require(bool(evidence), 'review is missing causal evidence')
    for filename, digest in evidence.items():
        require(sha256(Path(filename)) == digest, 'paired evidence changed: '+Path(filename).name)

    def artifact(name):
        matches = [Path(p) for p in evidence if Path(p).name == name]
        require(len(matches) == 1, 'missing or ambiguous causal artifact: '+name)
        return json.loads(matches[0].read_text())

    reviewed = {}
    for row in proof['reviewed_skips']:
        key = (row['branch'], row['name'])
        require(key not in reviewed, 'duplicate terminal skip review')
        require(row['status_kept'] == 'skipped' and row['native_status'] == 'passed', 'review changes official outcomes')
        require(task+'/tests/'+key[0]+'.tar.gz' in archives, 'reviewed branch not bound to official archive')
        reviewed[key] = row['message']
    require(bool(reviewed), 'empty skip review')
    manifest = artifact('manifest.json')
    require(manifest['task_id'] == task and manifest['submission_sha256'] == proof['submission_sha256']
            and manifest['original_executable_hash'] == proof['executable_sha256'], 'paired manifest identity mismatch')
    require(set(reviewed) == {(manifest['branch'], n) for n in manifest['selected_tests']}, 'paired selection mismatch')
    for variant, status in [('native', 'passed'), ('candidate', 'skipped')]:
        results = artifact(variant+'-results.json')
        require(results.get('diagnostic_only') is True, 'causal artifact is not a paired diagnostic')
        rows = {(r['branch'], r['name']):r for r in results['results']}
        require(len(rows) == len(results['results']) and set(rows) == set(reviewed), 'paired test identities do not reconcile')
        require(all(r['status'] == status for r in rows.values()), 'paired causal outcome does not hold')
        observed = artifact(variant+'-observations.json')
        require(observed['exitstatus'] == 0 and set(observed['tests']) == {k[1] for k in reviewed},
                'paired observation is incomplete')
        require(all(any(r['when'] == 'call' and r['outcome'] == status for r in t['reports'])
                    for t in observed['tests'].values()), 'paired tests did not execute the expected outcome')
        if variant == 'candidate':
            require(results['executable_hash'] == proof['executable_sha256'], 'paired candidate executable changed')
            require(all(r.get('extra', {}).get('message') == reviewed[k] for k, r in rows.items()),
                    'paired candidate skip cause changed')
    return reviewed
