#!/usr/bin/env python3
"""
Query and display experiment tracking data.

This script provides a simple CLI for viewing training runs, metrics,
and evaluation results from the experiments database.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime
from bucket_brigade.db.experiments import init_experiments_db
from bucket_brigade.db.models import ExperimentRun, TrainingMetric, EvaluationResult


def list_all_runs(session):
    """List all experiment runs with basic info."""
    runs = session.query(ExperimentRun).order_by(ExperimentRun.started_at.desc()).all()

    if not runs:
        print("No experiment runs found.")
        return

    print(
        f"\n{'ID':<6} {'Run Name':<40} {'Scenario':<20} {'Started':<20} {'Status':<10}"
    )
    print("=" * 100)

    for run in runs:
        status = "Complete" if run.completed_at else "Running"
        started = run.started_at.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{run.id:<6} {run.run_name:<40} {run.scenario:<20} {started:<20} {status:<10}"
        )

    print(f"\nTotal runs: {len(runs)}")


def show_run_details(session, run_name):
    """Show detailed information about a specific run."""
    run = (
        session.query(ExperimentRun)
        .filter(ExperimentRun.run_name.like(f"%{run_name}%"))
        .first()
    )

    if not run:
        print(f"No run found matching: {run_name}")
        return

    print(f"\n🔬 Experiment Run: {run.run_name} (ID: {run.id})")
    print("=" * 80)

    print(f"\n📋 Basic Info:")
    print(f"   Scenario: {run.scenario}")
    print(f"   Model Path: {run.model_path}")
    print(f"   Started: {run.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if run.completed_at:
        print(f"   Completed: {run.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        duration = (run.completed_at - run.started_at).total_seconds() / 60
        print(f"   Duration: {duration:.1f} minutes")
    else:
        print(f"   Status: Running")

    if run.hyperparameters:
        print(f"\n⚙️  Hyperparameters:")
        for key, value in run.hyperparameters.items():
            print(f"   {key}: {value}")

    if run.final_stats:
        print(f"\n📊 Final Statistics:")
        for key, value in run.final_stats.items():
            print(f"   {key}: {value}")

    # Show training metrics summary
    metrics = (
        session.query(TrainingMetric)
        .filter(TrainingMetric.run_id == run.id)
        .order_by(TrainingMetric.step)
        .all()
    )

    if metrics:
        print(f"\n📈 Training Metrics: {len(metrics)} recorded")
        print(f"   First step: {metrics[0].step}")
        print(f"   Last step: {metrics[-1].step}")
        if metrics[-1].avg_reward is not None:
            print(f"   Final avg reward: {metrics[-1].avg_reward:.2f}")

    # Show evaluation results
    evaluations = (
        session.query(EvaluationResult)
        .filter(EvaluationResult.run_id == run.id)
        .order_by(EvaluationResult.evaluated_at)
        .all()
    )

    if evaluations:
        print(f"\n🎯 Evaluation Results:")
        print(f"   {'Scenario':<25} {'Mean Reward':<15} {'Episodes':<10} {'Date':<20}")
        print("   " + "-" * 70)
        for eval in evaluations:
            date = eval.evaluated_at.strftime("%Y-%m-%d %H:%M:%S")
            mean = float(eval.mean_reward) if eval.mean_reward is not None else 0.0
            std = float(eval.std_reward) if eval.std_reward is not None else 0.0
            print(
                f"   {eval.eval_scenario:<25} {mean:>6.2f} ± {std:<5.2f} {eval.num_episodes:<10} {date:<20}"
            )


def list_by_scenario(session, scenario):
    """List all runs for a specific scenario."""
    runs = (
        session.query(ExperimentRun)
        .filter(ExperimentRun.scenario == scenario)
        .order_by(ExperimentRun.started_at.desc())
        .all()
    )

    if not runs:
        print(f"No runs found for scenario: {scenario}")
        return

    print(f"\n🎮 Runs for scenario: {scenario}")
    print(f"\n{'ID':<6} {'Run Name':<40} {'Started':<20} {'Status':<10}")
    print("=" * 80)

    for run in runs:
        status = "Complete" if run.completed_at else "Running"
        started = run.started_at.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{run.id:<6} {run.run_name:<40} {started:<20} {status:<10}")

    print(f"\nTotal runs: {len(runs)}")


def compare_runs(session, scenario):
    """Compare all completed runs for a scenario."""
    runs = (
        session.query(ExperimentRun)
        .filter(
            ExperimentRun.scenario == scenario,
            ExperimentRun.completed_at.isnot(None),
        )
        .order_by(ExperimentRun.started_at.desc())
        .all()
    )

    if not runs:
        print(f"No completed runs found for scenario: {scenario}")
        return

    print(f"\n📊 Comparing runs for scenario: {scenario}")
    print(
        f"\n{'Run Name':<40} {'Final Reward':<15} {'Steps':<10} {'LR':<8} {'Batch':<8}"
    )
    print("=" * 85)

    for run in runs:
        # Get final training metric
        final_metric = (
            session.query(TrainingMetric)
            .filter(TrainingMetric.run_id == run.id)
            .order_by(TrainingMetric.step.desc())
            .first()
        )

        final_reward = ""
        if final_metric and final_metric.avg_reward is not None:
            final_reward = f"{final_metric.avg_reward:.2f}"

        steps = (
            run.hyperparameters.get("num_steps", "N/A")
            if run.hyperparameters
            else "N/A"
        )
        lr = (
            run.hyperparameters.get("learning_rate", "N/A")
            if run.hyperparameters
            else "N/A"
        )
        batch = (
            run.hyperparameters.get("batch_size", "N/A")
            if run.hyperparameters
            else "N/A"
        )

        print(f"{run.run_name:<40} {final_reward:<15} {steps:<10} {lr:<8} {batch:<8}")


def main():
    parser = argparse.ArgumentParser(description="Query experiment tracking database")
    parser.add_argument(
        "--experiments-db",
        type=str,
        default=None,
        help="Path to experiments database (default: data/experiments.db)",
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Show details for a specific run (matches run name)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="List runs for a specific scenario",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare all runs for a scenario",
    )

    args = parser.parse_args()

    # Initialize database
    session = init_experiments_db(args.experiments_db)

    try:
        if args.run:
            show_run_details(session, args.run)
        elif args.scenario:
            list_by_scenario(session, args.scenario)
        elif args.compare:
            compare_runs(session, args.compare)
        else:
            list_all_runs(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()
