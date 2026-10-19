"""Opt-in persistent black-box oracle for the isolated Medium V19 candidate."""
from pier.environments.base import BaseEnvironment
from rho.datasets.programbench.agents import MountedProgramBenchCodex


class ReferenceProgramBenchCodex(MountedProgramBenchCodex):
    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        result = await environment.exec(
            command='python3 /harness/tools/preserve_reference.py',user='root',timeout_sec=60)
        if result.return_code:
            raise RuntimeError('protected reference setup failed: '+str(result.stderr or result.stdout))
        result = await environment.exec(
            command='test -x /opt/rho-reference/executable && '
                    '! test -r /opt/rho-reference/executable && '
                    '! test -w /opt/rho-reference/executable',user='agent',timeout_sec=10)
        if result.return_code:
            raise RuntimeError('reference execute-only permission boundary did not hold')
