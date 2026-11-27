from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai texto paginado de um PDF para curadoria de FAQs.")
    parser.add_argument("pdf", type=Path, help="Caminho do arquivo PDF de entrada.")
    parser.add_argument("--saida", type=Path, required=True, help="Caminho do arquivo .txt de saída.")
    args = parser.parse_args()

    from pypdf import PdfReader

    reader = PdfReader(str(args.pdf))
    blocos: list[str] = []
    for numero, pagina in enumerate(reader.pages, start=1):
        texto = (pagina.extract_text() or "").strip()
        blocos.append(f"=== PÁGINA {numero} ===\n{texto}")

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text("\n\n".join(blocos), encoding="utf-8")
    print(f"Extraídas {len(reader.pages)} páginas para: {args.saida}")


if __name__ == "__main__":
    main()