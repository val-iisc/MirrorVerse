import os
from loguru import logger as log
import shutil
from huggingface_hub import hf_hub_download
from pathlib import Path
import argparse
import requests
from tqdm import tqdm


def extract_contents(filename: str, extract_dir: str):
    try:
        log.info(f"Extracting {filename}")
        shutil.unpack_archive(filename, extract_dir)
        log.info("Extraction complete.")
        # Remove the file
        os.remove(filename)
    except Exception as e:
        log.error(f"Error Extracting {filename}: {e}")
        raise e


def download_file(url, local_filename):
    # NOTE: It is better to use wget instead of this.
    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)

    # Check if the file already exists and get its size
    if os.path.exists(local_filename):
        local_file_size = os.path.getsize(local_filename)
    else:
        local_file_size = 0

    headers = {"Range": f"bytes={local_file_size}-"}

    response = requests.get(url, headers=headers, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    if response.status_code == 200 or response.status_code == 206:
        with open(local_filename, "ab") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
    else:
        log.error(f"Failed to download file. Status code: {response.status_code}")
    progress_bar.close()


def download_zip_full(data_dir: str, id: str):
    assert len(id) == 2 and 0 <= int(id) <= 52, "00 <= id <= 52"
    zip_file = f"RenderedImage_perobj_zips/compressed_imgs_perobj_{id}.zip"

    log.info(f"Downloading {zip_file} from hub.")
    dataset = hf_hub_download(
        "tiange/Cap3D",
        zip_file,
        repo_type="dataset",
        local_dir=data_dir,
    )
    log.info("Download complete.")

    zip_path = Path(data_dir) / zip_file

    # Unzip
    extract_contents(zip_path, os.path.join(data_dir, "RenderedImage_perobj_zips"))

    # extract sub zip files. ex: unzip ed51a51909ee46c780db3a85e821feb2.zip -d ed51a51909ee46c780db3a85e821feb2
    data_subdir = (
        Path(data_dir) / "RenderedImage_perobj_zips/Cap3D_Objaverse_renderimgs"
    )
    for sub_zip in data_subdir.glob("*.zip"):
        try:
            sub_zip_dir = data_subdir / sub_zip.stem
            extract_contents(sub_zip, sub_zip_dir)
        except Exception as e:
            log.error(f"Error unzipping {sub_zip}: {e}")
    log.info("All zip files extracted.")


def download_zip_partial(data_dir: str, id: str):
    assert 0 <= int(id) <= 7, "0 <= id <= 7"
    zip_file = f"misc/RenderedImage_zips/compressed_imgs_view{id}.zip"

    log.info(f"Downloading {zip_file} from hub.")
    dataset = hf_hub_download(
        "tiange/Cap3D",
        zip_file,
        repo_type="dataset",
        local_dir=data_dir,
    )
    log.info("Download complete.")

    zip_path = Path(data_dir) / zip_file

    # Unzip
    extract_contents(zip_path, os.path.join(data_dir, "RenderedImage_perobj_zips"))


def main(args):
    if args.type == "file":
        assert args.file is not None, "Please provide file name to download"
        if "Cap3D" in args.file:
            log.info(f"Downloading {args.file} from hub.")
            dataset = hf_hub_download(
                "tiange/Cap3D",
                args.file,
                repo_type="dataset",
                local_dir=args.data_dir,
            )
            log.info("Download complete.")
            if args.file.endswith(".zip"):
                zip_path = Path(args.data_dir) / args.file
                extract_contents(zip_path, args.data_dir)
        else:
            local_filepath = os.path.join(args.data_dir, args.file.split("/")[-1])
            download_file(url=args.file, local_filename=local_filepath)
            extract_contents(local_filepath, args.data_dir)

    else:
        for id_str in args.id:
            if args.type == "full":
                id_str = f"{id_str:02}"
                download_zip_full(args.data_dir, id_str)
            else:
                id_str = str(id_str)
                download_zip_partial(args.data_dir, id_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download renderings by Cap3D")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/manan/data",
        help="Root Data directory",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="file",
        choices=["full", "partial", "file"],
        help="Type of renderings (full: 20 views link, partial: 6 views link)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz",
        help="Single file to download Cap3D metadata or other renderings. (use in --type=file)",
    )
    parser.add_argument(
        "--id",
        type=int,
        nargs="+",
        help="IDs of the zip folder at https://huggingface.co/datasets/tiange/Cap3D/tree/main/RenderedImage_perobj_zips. Example: --id 2 3 4",
        default=[0],
    )

    args = parser.parse_args()
    main(args)
