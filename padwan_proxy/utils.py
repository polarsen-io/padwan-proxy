from collections.abc import Mapping

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def startup_banner(fields: Mapping[str, str]) -> Panel:
    """Boxed key/value summary of the running proxy; values may carry rich markup."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column(style="bold")
    for key, value in fields.items():
        grid.add_row(key, value)
    return Panel(grid, title="[green]padwan-proxy[/green]", expand=False)
