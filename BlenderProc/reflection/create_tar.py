import os
import tarfile


def create_tar_for_uids(uids, input_dir, relative_dir, output_dir, dry_run=False, prefix="batch"):
    """Create a tar file for a list of UIDs."""

    # count of files with prefix=prefix in output_dir
    count = len([name for name in os.listdir(output_dir) if name.startswith(prefix)])

    tar_name = f"{prefix}_{count + 1}.tar"
    tar_path = os.path.join(output_dir, tar_name)

    with tarfile.open(tar_path, "w") as tar:
        for uid in uids:
            for file_name in os.listdir(uid):
                file_path = os.path.join(uid, file_name)
                relative_path = os.path.relpath(file_path, start=relative_dir)
                if dry_run:
                    print(f"Would add: {relative_path}")
                else:
                    tar.add(file_path, arcname=relative_path)


def main(input_dir, relative_dir, output_dir, file_type="hdf5", dry_run=False, prefix="batch"):
    """Main function to create tar files in batches of 1000 UIDs."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect all UIDs
    uids = []
    for root, dirs, files in os.walk(input_dir):
        if any(file.endswith(f".{file_type}") for file in files):
            uids.append(root)

    # Process UIDs in batches of 500
    for i in range(0, len(uids), 500):
        batch_uids = uids[i : i + 500]
        create_tar_for_uids(batch_uids, input_dir, relative_dir, output_dir, dry_run=dry_run, prefix=prefix)


if __name__ == "__main__":
    input_dir = "/storage/users/ankitd/data/objaverse/SynMirror-v1/geometric_data"  # Replace with your input directory
    relative_dir = "/storage/users/ankitd/data/objaverse/SynMirror-v1"
    output_dir = "/storage/users/ankitd/data/objaverse/SynMirror/"  # Replace with your output directory
    dry_run = False
    prefix = "geometric_data"
    file_type = "npy"
    main(input_dir, relative_dir, output_dir, file_type=file_type, dry_run=dry_run, prefix=prefix)
