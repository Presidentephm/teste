"""
Script propositalmente quebrado para demonstrar o loop de auto-correção.

Contém dois defeitos que a estratégia heurística sabe resolver sozinha:
    1. usa ``json`` sem importar (NameError -> ``import json``);
    2. usa ``format_report`` de ``report_utils`` sem importar
       (NameError -> ``from report_utils import format_report``).
"""


def build_payload() -> dict:
    return {"agente": "autonomo", "versao": 1, "ok": True}


def main() -> None:
    payload = build_payload()
    print(json.dumps(payload))
    print(format_report(payload))


if __name__ == "__main__":
    main()
