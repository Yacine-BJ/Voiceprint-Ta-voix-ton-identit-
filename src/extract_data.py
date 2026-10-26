from __future__ import annotations

import csv
import io
import os
from pathlib import Path
import tarfile
from dotenv import load_dotenv
import requests

load_dotenv()

API_KEY = os.getenv("MOZILLA_API_KEY")
DATASET_ID = "cmu5jplf300nwmh07iqvk9leo"


def clear_existing_data(output_dir: Path) -> None:
    """Supprime les anciens fichiers MP3 et le manifeste pour repartir sur du propre."""
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_file() and (path.suffix.lower() == ".mp3" or path.name == "dataset_manifest.csv"):
                path.unlink()
    print("Anciens fichiers nettoyés.")


def download_subset(
        target_size_gb: float = 1.0,
        output_dir: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
):
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_data(output_dir)

    target_bytes = target_size_gb * 1024 * 1024 * 1024
    current_bytes = 0
    saved_count = 0

    downloaded_records = []
    path_to_client = {}

    if not API_KEY:
        print("Erreur : La clé API 'MOZILLA_API_KEY' est introuvable dans le fichier .env.")
        return

    print("Demande de l'URL de téléchargement à l'API Mozilla...")
    api_url = f"https://mozilladatacollective.com/api/datasets/{DATASET_ID}/download"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, headers=headers)
    if response.status_code != 200:
        print(f"Erreur API ({response.status_code}) : {response.text}")
        return

    download_url = response.json().get("downloadUrl")
    if not download_url:
        print("Impossible de récupérer le lien de téléchargement.")
        return

    print(f"Téléchargement ciblé d'environ {target_size_gb} Go en cours...")

    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()

        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for member in tar:
                # Capture et chargement de validated.tsv dès qu'on le croise
                if member.isfile() and member.name.endswith("validated.tsv"):
                    print("\nFichier de métadonnées 'validated.tsv' atteint. Chargement des client_ids...")
                    f = tar.extractfile(member)
                    if f is not None:
                        tsv_bytes = f.read()
                        text_stream = io.TextIOWrapper(io.BytesIO(tsv_bytes), encoding="utf-8")
                        reader = csv.DictReader(text_stream, delimiter="\t")

                        for row in reader:
                            p = row.get("path")
                            cid = row.get("client_id")
                            if p and cid:
                                file_key = Path(p).name
                                path_to_client[file_key] = cid
                        print(f" -> {len(path_to_client)} correspondances chargées en mémoire.")

                # Capture et sauvegarde des fichiers MP3 à la volée
                elif member.isfile() and member.name.endswith(".mp3"):
                    if current_bytes >= target_bytes:
                        break

                    f = tar.extractfile(member)
                    if f is not None:
                        audio_bytes = f.read()
                        file_size = len(audio_bytes)

                        file_name = Path(member.name).name
                        dest_path = output_dir / file_name
                        with open(dest_path, "wb") as out_f:
                            out_f.write(audio_bytes)

                        current_bytes += file_size
                        saved_count += 1

                        client_id = path_to_client.get(file_name, "unknown")

                        downloaded_records.append({
                            "file_name": file_name,
                            "client_id": client_id,
                            "file_size_bytes": file_size
                        })

                        if saved_count % 100 == 0:
                            print(f"[{saved_count} fichiers] - Espace utilisé : {current_bytes / (1024 * 1024):.2f} Mo")

    # Écriture du manifeste final
    manifest_path = output_dir / "dataset_manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as csv_f:
        fieldnames = ["file_name", "client_id", "file_size_bytes"]
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in downloaded_records:
            writer.writerow(rec)

    print(f"\nObjectif atteint ! {saved_count} fichiers téléchargés ({current_bytes / (1024 * 1024 * 1024):.2f} Go).")
    print(f"Manifeste généré : {manifest_path}")


if __name__ == "__main__":
    download_subset(target_size_gb=1.0)