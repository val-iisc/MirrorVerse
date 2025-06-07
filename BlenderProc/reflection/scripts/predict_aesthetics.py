import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from loguru import logger as log
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
import argparse
from accelerate import Accelerator
from glob import glob

# pip install simple-aesthetics-predictor
from aesthetics_predictor import AestheticsPredictorV2Linear

class ImageDataset(Dataset):
    def __init__(self, args, processor, df):
        self.processor = processor
        subdirs = os.listdir(args.input_dir)
        already_processed = df["uid"].tolist()
        already_processed.sort()
        log.info(f"Already processed {len(already_processed)} uids.")

        self.image_files = []
        for subdir in subdirs:
            if subdir in already_processed:
                continue
            images = glob(os.path.join(args.input_dir, subdir, "*.png"))
            if "Cap3D_Objaverse_renderimgs" in args.input_dir:
                # for the renderings from Cap3D, we need to remove the depth and alpha images
                images = [
                    img for img in images if "MatAlpha" not in img and "depth" not in img
                ]
            assert (
                len(images) == args.num_views
            ), f"Each subdir should contain exactly {args.num_views} images, but {subdir} contains {len(images)}"
            self.image_files.append(images)
        if args.end_idx is None:
            args.end_idx = len(self.image_files)
        self.image_files = self.image_files[args.start_idx:args.end_idx]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        images = [Image.open(img) for img in self.image_files[idx]]
        inputs = self.processor(images=images, return_tensors="pt")
        subdir_name = self.image_files[idx][0].split(os.sep)[-2]  # Extract subdir name from the first image path
        return inputs, subdir_name


def collate_fn(batch):
    """cat the inputs["pixel_values"] and return the subdir names as well."""
    inputs, subdir_names = zip(*batch)
    pixel_values = torch.cat([inp["pixel_values"] for inp in inputs], dim=0)
    return {"pixel_values": pixel_values}, subdir_names


def save_data(data, df, outfile):
    # append data to df
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    # create csv
    log.info(f"Saving checkpoint with {df.shape[0]} uids.")
    df.to_csv(outfile, index=False)
    return data, df


def main(args):
    accelerator = Accelerator()
    predictor = AestheticsPredictorV2Linear.from_pretrained(args.model_id)
    processor = CLIPProcessor.from_pretrained(args.model_id)

    data = {"uid": [], "aesthetic_score": []}
    if os.path.exists(args.output):
        log.info(f"Resuming from {args.output}")
        df = pd.read_csv(args.output)
    else:
        df = pd.DataFrame(data, columns=["uid", "aesthetic_score"])

    dataset = ImageDataset(args, processor, df)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    device = accelerator.device
    predictor = predictor.to(device)
    progress_bar = tqdm(total=len(dataset), disable=not accelerator.is_local_main_process)

    for idx, (inputs, subdir_names) in enumerate(dataloader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = predictor(**inputs)
        predictions = outputs.logits
        # chunk the predictions, each chunk corresponding to a subdir (uid)
        predictions = predictions.chunk(len(subdir_names), dim=0)
        averages = torch.stack([prediction.mean(dim=0) for prediction in predictions]).reshape(-1)
        # Add the subdirs and their corresponding averages to the data list
        data["uid"].extend(subdir_names)
        avaerages = averages.cpu().numpy().tolist()
        data["aesthetic_score"].extend(avaerages)
        if args.checkpoint is not None and len(data["uid"]) >= args.checkpoint and accelerator.is_local_main_process:
            _, df = save_data(data, df, args.output)
            data = {"uid": [], "aesthetic_score": []}
        progress_bar.update(len(subdir_names))
    
    if accelerator.is_local_main_process and len(data["uid"]) > 0:
        log.info("Saving final checkpoint")
        save_data(data, df, args.output)


if __name__ == "__main__":
    # accelerate launch --num_processes 2 --gpu_ids="0,7" --main_process_port=8000 predict_aesthetics.py --batch_size=32
    parser = argparse.ArgumentParser(description="Get aesthetics scores on objaverse renderings")
    parser.add_argument(
        "--model_id",
        type=str,
        default="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
        help="Model ID of the aesthetics predictor",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/manan/data/views_release",
        help="Input directory. Example subdir: 3500e52341194c94b51b73edd260c662",
    )
    parser.add_argument("--num_views", type=int, default=12, help="num of views to use for averaging aesthetic score.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index of dataset")
    parser.add_argument(
        "--end_idx", type=int, default=None, help="Ending index of the dataset"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--output", type=str, default="../resources/aesthetics.csv", help="Output CSV file. Use this flag when resuming as well"
    )
    parser.add_argument(
        "--checkpoint", type=int, default=None, help="Use checkpoint for storing intermediate results"
    )
    args = parser.parse_args()
    main(args)
