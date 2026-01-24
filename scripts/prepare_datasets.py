import shutil
from pathlib import Path
import os
from tqdm.auto import tqdm

def copy_images_and_split(miniimagenet_src, cifarfs_src, pipeline_root):
    """
    Copy Mini-ImageNet and CIFAR-FS images from downloaded sources to pipeline structure.
    Args:
        miniimagenet_src: Path to folder containing Mini-ImageNet images (should contain train, val, test or images/)
        cifarfs_src: Path to folder containing CIFAR-FS images (should contain train, val, test)
        pipeline_root: Path to pipeline datasets folder (should contain minimagenet/ and cifar_fs/)
    """
    # Mini-ImageNet
    mini_dst = Path(pipeline_root) / 'minimagenet'
    if (miniimagenet_src / 'images').exists():
        # meta-learning-lstm format
        src_img = miniimagenet_src / 'images'
        for split in ['train', 'val', 'test']:
            split_csv = miniimagenet_src / f'{split}.csv'
            split_dir = mini_dst / split
            split_dir.mkdir(parents=True, exist_ok=True)
            # Count lines for tqdm
            with open(split_csv, 'r') as f:
                total = sum(1 for _ in f) - 1
            with open(split_csv, 'r') as f:
                next(f)  # skip header
                for line in tqdm(f, total=total, desc=f'Copy Mini-ImageNet {split}'):
                    img, label = line.strip().split(',')
                    src_file = src_img / img
                    dst_file = split_dir / img
                    if src_file.exists():
                        if not dst_file.exists():
                            shutil.copy2(src_file, dst_file)
    elif (miniimagenet_src / 'data').exists():
        # r2d2 format
        src_img = miniimagenet_src / 'data'
        for split in ['train', 'val', 'test']:
            split_dir = mini_dst / split
            split_dir.mkdir(parents=True, exist_ok=True)
            split_classes = [d for d in (src_img).iterdir() if d.is_dir()]
            img_files = []
            for cls in split_classes:
                img_files.extend([(cls, img_file) for img_file in cls.iterdir()])
            for cls, img_file in tqdm(img_files, desc=f'Copy Mini-ImageNet {split}'):
                dst_file = split_dir / f'{cls.name}_{img_file.name}'
                if not dst_file.exists():
                    shutil.copy2(img_file, dst_file)
    else:
        print('Mini-ImageNet source not found or invalid format.')

    # CIFAR-FS
    cifar_dst = Path(pipeline_root) / 'cifar_fs'
    # Support both: (1) train/val/test subfolders, (2) data + splits/bertinetto
    if (cifarfs_src / 'train').exists():
        for split in ['train', 'val', 'test']:
            src_split = cifarfs_src / split
            dst_split = cifar_dst / split
            dst_split.mkdir(parents=True, exist_ok=True)
            img_files = [img_file for img_file in src_split.glob('**/*') if img_file.is_file()]
            for img_file in tqdm(img_files, desc=f'Copy CIFAR-FS {split}'):
                dst_file = dst_split / img_file.name
                if not dst_file.exists():
                    shutil.copy2(img_file, dst_file)
    elif (cifarfs_src / 'data').exists() and (cifarfs_src / 'splits' / 'bertinetto').exists():
        # Use split files to copy from data/ to train/val/test
        data_dir = cifarfs_src / 'data'
        splits_dir = cifarfs_src / 'splits' / 'bertinetto'
        split_map = {'train': 'train.txt', 'val': 'val.txt', 'test': 'test.txt'}
        for split, split_file in split_map.items():
            split_path = splits_dir / split_file
            out_dir = cifar_dst / split
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(split_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            # Deteksi: jika entry split adalah nama kelas (folder), copy semua file di folder tsb
            if all((data_dir / l).is_dir() for l in lines):
                # Semua entry adalah nama kelas
                total = sum(len(list((data_dir / l).glob('*'))) for l in lines)
                pbar = tqdm(total=total, desc=f'Copy CIFAR-FS {split}')
                for class_name in lines:
                    class_dir = data_dir / class_name
                    if class_dir.is_dir():
                        for img_file in class_dir.glob('*'):
                            dst_file = out_dir / f'{class_name}_{img_file.name}'
                            if img_file.is_file() and not dst_file.exists():
                                shutil.copy2(img_file, dst_file)
                            pbar.update(1)
                    else:
                        print(f'Class folder not found: {class_dir}')
                pbar.close()
            else:
                # Entry split adalah path file (class_name/image.png)
                for line in tqdm(lines, desc=f'Copy CIFAR-FS {split}'):
                    cls_img = line
                    src_file = data_dir / cls_img
                    if src_file.is_file():
                        dst_file = out_dir / f'{cls_img.replace("/", "_")}'
                        if not dst_file.exists():
                            shutil.copy2(src_file, dst_file)
                    else:
                        if not src_file.is_dir():
                            print(f'File not found: {src_file}')
    else:
        print('CIFAR-FS source not found or invalid format.')

if __name__ == '__main__':
    # Example usage:
    # python scripts/prepare_datasets.py --mini_src datasets/meta-learning-lstm/data/miniImagenet --cifar_src datasets/r2d2/cifarfs --pipeline datasets
    import argparse
    parser = argparse.ArgumentParser(description='Copy Mini-ImageNet and CIFAR-FS to pipeline structure')
    parser.add_argument('--mini_src', type=Path, required=True, help='Path to Mini-ImageNet source folder')
    parser.add_argument('--cifar_src', type=Path, required=True, help='Path to CIFAR-FS source folder')
    parser.add_argument('--pipeline', type=Path, default=Path('datasets'), help='Path to pipeline datasets folder')
    args = parser.parse_args()
    copy_images_and_split(args.mini_src, args.cifar_src, args.pipeline)
