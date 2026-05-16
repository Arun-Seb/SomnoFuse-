from setuptools import setup, find_packages

setup(
    name="cps-sleep-analysis",
    version="0.1.0",
    description="Multimodal snoring and sleep arousal analysis on the CPS PhysioNet dataset",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24", "scipy>=1.11", "pandas>=2.0", "scikit-learn>=1.3",
        "wfdb>=4.1.0", "librosa>=0.10", "mne>=1.6",
        "torch>=2.1", "torchaudio>=2.1",
        "pyyaml>=6.0", "tqdm>=4.66", "rich>=13.0",
    ],
)
