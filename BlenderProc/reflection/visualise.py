import fiftyone as fo
import os
import argparse
from glob import glob

def main(args):

    if args.load_saved:
        dataset = fo.load_dataset("Mirror-Dataset")
    else:
        # file_paths = glob(os.path.join(args.data_dir, "**", "*.png"), recursive=True)
        file_paths = [
            "/home/ankitd/manan/Reflection-Exploration/BrushNet/runs/logs/brushnet_train_unet/checkpoint-50000-multi/inference_multiple_234_new/8343c810f80e42aa849cea818ef1b632_B075X4PTS8_B075X4J118_a0d08e45c4484b46976b44f881f6453d_2.png",
            "/home/ankitd/manan/Reflection-Exploration/BrushNet/runs/logs/brushnet_train_unet/checkpoint-50000-multi/inference_multiple_234_new/429db223039e4464a1bce14d0745be95_19ac02a101dc47968f58aba5eae4dcd2_efcba4fb2d15422580077e2160436d06_4af3c47765af45fd9b0d592a5cb7c7c2_2.png",
            "/home/ankitd/manan/Reflection-Exploration/BrushNet/runs/logs/brushnet_train_unet/checkpoint-50000-multi/inference_multiple_234_new/8343c810f80e42aa849cea818ef1b632_B075X4PTS8_1.png",
            "/home/ankitd/manan/Reflection-Exploration/BrushNet/runs/logs/brushnet_train_unet/checkpoint-50000-multi/inference_multiple_234_new/429db223039e4464a1bce14d0745be95_19ac02a101dc47968f58aba5eae4dcd2_efcba4fb2d15422580077e2160436d06_0.png",
            "/home/ankitd/manan/Reflection-Exploration/BrushNet/runs/logs/brushnet_train_unet/checkpoint-50000-multi/inference_multiple_234_new/4fc697a6dc25426ea920cf89f737a764_a0d08e45c4484b46976b44f881f6453d_5e7c981ad2974772bf85028039ab9d35_1.png"
        ]
        dataset = fo.Dataset.from_images(
            list(file_paths),
            name="Mirror-Dataset",
            overwrite=True,
            persistent=True,
            progress=True,
        )

    session = fo.launch_app(dataset, remote=True, port=args.port)
    session.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from HDF5 files")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/manan/.cache/hf-objaverse-v1",
        help="Data directory",
    )
    parser.add_argument(
        "--load_saved",
        action="store_true",
        help="After creating the fo.Dataset once, \
          use this flag to load the already created dataset",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port number for the FiftyOne session",
    )
    args = parser.parse_args()
    main(args)