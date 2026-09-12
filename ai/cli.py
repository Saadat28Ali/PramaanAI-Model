"""CLI for document verification, ELA analysis, synthetic generation, and serving."""

import click
import cv2
import json
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint


from src.config import DocuNetConfig
from src.pipeline import DocuNetPipeline

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="DocuNet")
def cli():
    pass


@cli.command()
@click.option("--image", "-i", required=True, help="Path to the ID card image.")
@click.option("--output", "-o", default=None, help="Output directory for results.")
@click.option("--skip-quality", is_flag=True, help="Skip quality gate checks.")
@click.option("--skip-ocr", is_flag=True, help="Skip OCR (tamper analysis only).")
@click.option("--json-output", is_flag=True, help="Output raw JSON instead of formatted table.")
def verify(image, output, skip_quality, skip_ocr, json_output):
    """Verify a single document image."""
    image_path = Path(image)
    if not image_path.exists():
        console.print(f"[red]Error:[/red] Image not found: {image}")
        sys.exit(1)

    console.print(Panel(f"[bold blue]DocuNet[/bold blue] — Verifying: {image_path.name}"))

    with console.status("[cyan]Initializing pipeline..."):
        pipeline = DocuNetPipeline()

    with console.status("[cyan]Processing image..."):
        result = pipeline.process_file(str(image_path), skip_quality, skip_ocr)

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    if result.quality_report:
        qr = result.quality_report
        status = "[green]PASSED[/green]" if qr.passed else "[red]FAILED[/red]"
        console.print(f"\n[bold]Quality Gate:[/bold] {status}")

        quality_table = Table(show_header=False)
        quality_table.add_row("Blur Score", f"{qr.blur_score:.1f}")
        quality_table.add_row("Brightness", f"{qr.brightness:.1f}")
        quality_table.add_row("Contrast", f"{qr.contrast:.1f}")
        quality_table.add_row("Glare Ratio", f"{qr.glare_ratio:.2%}")
        quality_table.add_row("Resolution", f"{qr.resolution[0]}x{qr.resolution[1]}")
        console.print(quality_table)

        if qr.issues:
            for issue in qr.issues:
                console.print(f"  [yellow]⚠ {issue}[/yellow]")

    if result.rectification_result:
        rr = result.rectification_result
        status = "[green]✓[/green]" if rr.success else "[yellow]✗[/yellow]"
        console.print(f"\n[bold]Rectification:[/bold] {status} ({rr.method_used})")

    if result.ela_result:
        er = result.ela_result
        if er.is_tampered:
            console.print(f"\n[bold]Tamper Detection:[/bold] [red]⚠ SUSPICIOUS[/red]")
        else:
            console.print(f"\n[bold]Tamper Detection:[/bold] [green]✓ CLEAN[/green]")

        console.print(f"  Anomaly Score: {er.anomaly_score:.4f}")
        console.print(f"  Suspicious Regions: {len(er.suspicious_regions)}")

    if result.ocr_result:
        ocr = result.ocr_result
        console.print(f"\n[bold]OCR Results:[/bold] ({ocr.engine_used})")
        console.print(f"  Avg Confidence: {ocr.avg_confidence:.3f}")
        console.print(f"  Text Regions: {len(ocr.boxes)}")

    if result.parsed_document:
        pd = result.parsed_document
        console.print(f"\n[bold]Document Type:[/bold] {pd.document_type}")

        if pd.fields:
            fields_table = Table(title="Extracted Fields")
            fields_table.add_column("Field", style="cyan")
            fields_table.add_column("Value", style="white")
            fields_table.add_column("Confidence", style="green")

            for name, field in pd.fields.items():
                conf_str = f"{field.confidence:.2f}"
                fields_table.add_row(name, field.value, conf_str)

            console.print(fields_table)

    console.print(f"\n[bold]Processing Time:[/bold]")
    timing_table = Table(show_header=True)
    timing_table.add_column("Stage", style="cyan")
    timing_table.add_column("Time (ms)", justify="right", style="yellow")

    for stage, ms in result.timings.items():
        timing_table.add_row(stage, f"{ms:.1f}")

    timing_table.add_row("[bold]TOTAL[/bold]", f"[bold]{result.total_time_ms:.1f}[/bold]")
    console.print(timing_table)

    if output:
        saved = pipeline.save_results(result, output, prefix=image_path.stem)
        console.print(f"\n[green]Results saved to {output}/[/green]")
        for name, path in saved.items():
            console.print(f"  {name}: {path}")


@cli.command()
@click.option("--image", "-i", required=True, help="Path to the image.")
@click.option("--output", "-o", default=None, help="Save ELA heatmap to this path.")
def ela(image, output):
    """Run ELA tamper detection only."""
    from src.forensics.ela_detector import ELADetector
    image_path = Path(image)
    if not image_path.exists():
        console.print(f"[red]Error:[/red] Image not found: {image}")
        sys.exit(1)

    img = cv2.imread(str(image_path))
    detector = ELADetector()
    result = detector.analyze(img)

    if result.is_tampered:
        console.print(f"[red]⚠ TAMPER DETECTED[/red] — Score: {result.anomaly_score:.4f}")
    else:
        console.print(f"[green]✓ No tampering detected[/green] — Score: {result.anomaly_score:.4f}")

    console.print(f"Suspicious regions: {len(result.suspicious_regions)}")

    if output:
        cv2.imwrite(output, result.heatmap)
        console.print(f"Heatmap saved to {output}")


@cli.command()
@click.option("--input", "-i", "input_dir", required=True, help="Directory of clean ID images.")
@click.option("--output", "-o", "output_dir", default="data/synthetic", help="Output directory.")
@click.option("--count", "-n", default=100, help="Number of augmented images per source.")
def generate(input_dir, output_dir, count):
    """Generate synthetic degraded document images."""
    from src.synthetic.generator import SyntheticDataGenerator
    console.print(Panel("[bold blue]Synthetic Data Generator[/bold blue]"))
    generator = SyntheticDataGenerator()
    records = generator.generate(input_dir, output_dir, num_per_image=count)

    console.print(f"[green]Generated {len(records)} augmented images[/green]")
    console.print(f"Output: {output_dir}/")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host.")
@click.option("--port", "-p", default=8000, help="Server port.")
def server(host, port):
    """Start the DocuNet API server."""
    from src.api.server import start_server
    console.print(Panel(f"[bold blue]DocuNet API Server[/bold blue] — {host}:{port}"))
    start_server(host=host, port=port)


@cli.command()
@click.option("--input", "-i", "input_dir", required=True, help="Directory of test images.")
@click.option("--output", "-o", "output_dir", default="benchmarks", help="Output directory.")
def benchmark(input_dir, output_dir):
    """Run benchmark on a directory of test images."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("**/*.jpg")) + list(input_path.glob("**/*.png"))

    if not image_files:
        console.print(f"[red]No images found in {input_dir}[/red]")
        sys.exit(1)

    console.print(f"[bold]Benchmarking {len(image_files)} images...[/bold]")

    pipeline = DocuNetPipeline()
    results = []

    for img_file in image_files:
        result = pipeline.process_file(str(img_file), skip_quality_gate=True)
        results.append({
            "file": str(img_file),
            "success": result.success,
            "total_time_ms": result.total_time_ms,
            "timings": result.timings,
            "tamper_score": result.ela_result.anomaly_score if result.ela_result else None,
            "ocr_confidence": result.ocr_result.avg_confidence if result.ocr_result else None,
        })

    times = [r["total_time_ms"] for r in results if r["success"]]
    if times:
        import numpy as np
        console.print(f"\n[bold]Latency (ms):[/bold]")
        console.print(f"  p50: {np.percentile(times, 50):.1f}")
        console.print(f"  p95: {np.percentile(times, 95):.1f}")
        console.print(f"  p99: {np.percentile(times, 99):.1f}")
        console.print(f"  Mean: {np.mean(times):.1f}")

    # Save benchmark results
    report_path = output_path / "benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Benchmark results saved to {report_path}[/green]")


if __name__ == "__main__":
    cli()
