import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image

from utils import data_loader_tif as data_loader
from network.SemiModel import SemiModel


def test_unpaired(loader, save_path, model, batch_size):
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    dataset = loader.dataset

    with torch.no_grad():
        for batch_idx, (A, B) in enumerate(tqdm(loader, desc="Testing Unpaired")):
            A = A.cuda()
            B = B.cuda()
            _, logits = model(A, B)
            probs = torch.sigmoid(logits)

            start = batch_idx * batch_size
            end = start + A.size(0)
            paths = dataset.present_paths[start:end]
            fnames = [os.path.splitext(os.path.basename(p))[0] for p in paths]

            for i, fname in enumerate(fnames):
                mask_img = (probs[i,0].cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(os.path.join(save_path, f"{fname}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SemiModel on folder of Unpaired TIFFs')
    parser.add_argument('--data_dir',    type=str, required=True,  help='Folder containing unpaired_pairs.csv')
    parser.add_argument('--batchsize',   type=int, default=16,       help='Batch size for testing')
    parser.add_argument('--trainsize',   type=int, default=256,      help='Image resize size')
    parser.add_argument('--gpu_id',      type=str, default='0',      help='CUDA_VISIBLE_DEVICES index')
    parser.add_argument('--checkpoint',  type=str, required=True,    help='Path to .pth checkpoint or directory containing .pth')
    parser.add_argument('--save_path',   type=str, required=True,    help='Directory to save predicted masks')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print(f"Using GPU {args.gpu_id}")

    # locate checkpoint file
    ckpt_arg = args.checkpoint
    if os.path.isdir(ckpt_arg):
        pth_files = [f for f in os.listdir(ckpt_arg) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"No .pth files found in directory {ckpt_arg}")
        # pick latest by name sort
        ckpt_file = os.path.join(ckpt_arg, sorted(pth_files)[-1])
        print(f"Using latest checkpoint from dir: {ckpt_file}")
    else:
        ckpt_file = ckpt_arg
        print(f"Loading checkpoint: {ckpt_file}")

    # data loader
    csv_path = os.path.join(args.data_dir, 'unpaired_pairs.csv')
    unpaired_loader = data_loader.get_unpaired_loader(
        csv_path, args.batchsize, args.trainsize,
        num_workers=4, shuffle=False, pin_memory=True
    )

    # model
    model = SemiModel().cuda()
    checkpoint = torch.load(ckpt_file)
    if isinstance(checkpoint, dict):
        for key in ['best_student_net', 'model_state_dict', 'state_dict']:
            if key in checkpoint:
                checkpoint = checkpoint[key]; break
    model.load_state_dict(checkpoint)
    print("Checkpoint loaded.")

    # run test on unpaired data
    test_unpaired(unpaired_loader, args.save_path, model, args.batchsize)