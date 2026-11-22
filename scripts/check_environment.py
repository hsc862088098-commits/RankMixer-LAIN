"""Print environment information without installing packages or contacting services."""
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys


def main():
    report = {'system': platform.platform(), 'python': sys.version,
              'executable': sys.executable, 'packages': {}}
    for name in ('torch', 'numpy', 'pandas', 'pyarrow', 'polars', 'heavyball',
                 'scikit-learn', 'PyYAML', 'h5py', 'keras-preprocessing', 'huggingface-hub'):
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    if report['packages']['torch']:
        import torch
        report.update(cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
                      gpu_count=torch.cuda.device_count())
    for command in ('git', 'code', 'codex', 'nvidia-smi'):
        report[command] = shutil.which(command)
    if report['nvidia-smi']:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        report['nvidia_smi_output'] = result.stdout.strip() or result.stderr.strip()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
