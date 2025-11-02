from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="bucket-brigade-core",
    version="0.1.0",
    packages=find_packages(),
    rust_extensions=[
        RustExtension(
            "bucket_brigade_core.bucket_brigade_core",
            path="Cargo.toml",
            binding=Binding.PyO3,
            features=["python"],
        )
    ],
    setup_requires=["setuptools-rust>=1.9.0"],
    zip_safe=False,
    python_requires=">=3.8",
    description="Rust core for Bucket Brigade multi-agent cooperation simulation",
    long_description=open("README.md").read() if "README.md" in ["README.md"] else "",
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
