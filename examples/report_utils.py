"""Utilitários usados pelo script de exemplo."""


def format_report(data: dict) -> str:
    """Formata um dicionário como relatório de texto simples."""
    return "\n".join(f"{k}: {v}" for k, v in data.items())
