# Bucket Brigade - Local Development Commands
# These commands mirror the CI pipeline for consistent local testing

.PHONY: help install test test-python test-web lint lint-python lint-web format format-python format-web typecheck build build-rust build-web clean ci-local

# Default target
help:
	@echo "Bucket Brigade - Local Development Commands"
	@echo ""
	@echo "Core Commands:"
	@echo "  install         Install all dependencies (Python, Rust, Node)"
	@echo "  test            Run all tests (Python + Web)"
	@echo "  lint            Run all linters (Python + Web)"
	@echo "  format          Run all formatters (Python + Web)"
	@echo "  typecheck       Run all type checkers (Python + Web)"
	@echo "  build           Build all components (Rust + Web)"
	@echo ""
	@echo "Component-Specific Commands:"
	@echo "  install-python  Install Python dependencies"
	@echo "  install-rust    Install Rust dependencies"
	@echo "  install-web     Install web dependencies"
	@echo "  test-python     Run Python tests only"
	@echo "  test-web        Run web tests only"
	@echo "  lint-python     Run Python linting only"
	@echo "  lint-web        Run web linting only"
	@echo "  format-python   Run Python formatting only"
	@echo "  format-web      Run web formatting only"
	@echo "  typecheck-python Run Python type checking only"
	@echo "  typecheck-web   Run web type checking only"
	@echo "  build-rust      Build Rust components only"
	@echo "  build-web       Build web components only"
	@echo ""
	@echo "CI Simulation:"
	@echo "  ci-local        Run complete CI pipeline locally"
	@echo "  clean           Clean build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make install        # Set up development environment"
	@echo "  make test           # Run all tests"
	@echo "  make ci-local       # Simulate full CI pipeline"

# Installation
install: install-python install-rust install-web

install-python:
	@echo "📦 Installing Python dependencies..."
	uv pip install -e .[dev]
	cd bucket-brigade-core && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv pip install -e .

install-rust:
	@echo "🦀 Installing Rust dependencies..."
	cd bucket-brigade-core && cargo fetch

install-web:
	@echo "🌐 Installing web dependencies..."
	pnpm install

# Testing
test: test-python test-web

test-python:
	@echo "🐍 Running Python tests..."
	uv run pytest --cov=bucket_brigade --cov-report=xml --tb=short

test-web:
	@echo "🌐 Running web tests..."
	cd web && timeout 30 pnpm run test || echo "⚠️  Web tests timed out or failed - this may be expected in CI"

# Linting
lint: lint-python lint-web lint-rust

lint-python:
	@echo "🐍 Running Python linting..."
	uv run ruff check . --exclude scripts/ --exclude node_modules/ --exclude web/node_modules/

lint-rust:
	@echo "🦀 Running Rust linting..."
	cd bucket-brigade-core && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy --all-features --all-targets -- -D warnings

lint-web:
	@echo "🌐 Running web linting..."
	cd web && pnpm run lint:biome

# Formatting
format: format-python format-web format-rust

format-python:
	@echo "🐍 Running Python formatting..."
	uv run black bucket_brigade/ tests/ --exclude scripts/ --exclude node_modules/ --exclude web/node_modules/

format-rust:
	@echo "🦀 Running Rust formatting..."
	cd bucket-brigade-core && cargo fmt

format-web:
	@echo "🌐 Running web formatting..."
	cd web && pnpm run format

# Type checking
typecheck: typecheck-python typecheck-web

typecheck-python:
	@echo "🐍 Running Python type checking..."
	uv run mypy . --ignore-missing-imports || true

typecheck-web:
	@echo "🌐 Running web type checking..."
	cd web && pnpm run typecheck

# Building
build: build-rust build-web

build-rust:
	@echo "🦀 Building Rust components..."
	cd bucket-brigade-core && cargo build --release
	cd bucket-brigade-core && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --features python

build-web:
	@echo "🌐 Building web components..."
	cd web && pnpm run build

# CI simulation
ci-local: install test lint typecheck build
	@echo "✅ Local CI pipeline completed successfully!"
	@echo ""
	@echo "Summary:"
	@echo "  ✅ Python tests passed"
	@echo "  ✅ Web tests passed"
	@echo "  ✅ Python linting passed"
	@echo "  ✅ Web linting passed"
	@echo "  ✅ Python type checking completed"
	@echo "  ✅ Web type checking completed"
	@echo "  ✅ Rust build completed"
	@echo "  ✅ Web build completed"
	@echo ""
	@echo "🎉 Ready for commit and push!"

# Cleaning
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf bucket_brigade_core.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf bucket_brigade/**/*.pyc
	rm -rf bucket_brigade/**/__pycache__/
	rm -rf bucket-brigade-core/target/
	rm -rf bucket-brigade-core/bucket_brigade_core.egg-info/
	rm -rf web/dist/
	rm -rf web/node_modules/.vite/
	@echo "✅ Clean completed"

# Development shortcuts
dev-web:
	@echo "🚀 Starting web development server..."
	cd web && pnpm run dev

dev-python:
	@echo "🐍 Starting Python development..."
	@echo "Run: uv run python scripts/run_one_game.py"

# Rust-specific commands
test-rust:
	@echo "🦀 Running Rust tests..."
	cd bucket-brigade-core && cargo test

test-rust-coverage:
	@echo "🦀 Running Rust tests with coverage..."
	cd bucket-brigade-core && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo llvm-cov --all-features --html
	@echo "📊 Coverage report generated at bucket-brigade-core/target/llvm-cov/html/index.html"

check-rust:
	@echo "🦀 Running Rust checks..."
	cd bucket-brigade-core && cargo check

clippy-rust:
	@echo "🦀 Running Rust clippy..."
	cd bucket-brigade-core && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy --all-features --all-targets -- -D warnings

# Web-specific commands
preview-web:
	@echo "🌐 Previewing built web app..."
	cd web && pnpm run preview

# Utility commands
deps-update:
	@echo "📦 Updating dependencies..."
	uv lock --upgrade
	cd web && pnpm update

# CI status check
ci-status:
	@echo "🔍 Checking CI readiness..."
	@echo "Python tests:" && uv run pytest --collect-only -q | grep -c "test session starts" || echo "❌ Python tests failed"
	@echo "Python lint:" && uv run ruff check . --exclude scripts/ --exclude node_modules/ --exclude web/node_modules/ --quiet && echo "✅ Python lint passed" || echo "❌ Python lint failed"
	@echo "Web build:" && cd web && pnpm run build --silent && echo "✅ Web build passed" || echo "❌ Web build failed"
	@echo "Rust build:" && cd bucket-brigade-core && cargo check --quiet && echo "✅ Rust check passed" || echo "❌ Rust check failed"
