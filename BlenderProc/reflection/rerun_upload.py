import subprocess
import time


def run_upload_script():
    while True:
        try:
            # Run the upload.py script with the specified arguments
            subprocess.run(
                [
                    "python",
                    "upload.py",
                    "--root_directory",
                    "/storage/users/ankitd/data/objaverse/hf-objaverse-v3",
                    "--relative_root",
                    "/storage/users/ankitd/data/objaverse/",
                    "--num_thread",
                    "8",
                    "--group_size",
                    "100",
                ],
                check=True,
            )
            print("Upload completed successfully!")
            break  # If the script completes without errors, exit the loop
        except subprocess.CalledProcessError:
            print("Upload failed. Retrying in 15 seconds...")
            time.sleep(15)  # Wait for 15 seconds before retrying


if __name__ == "__main__":
    run_upload_script()
