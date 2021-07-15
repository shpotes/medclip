import csv
from pathlib import Path
import torchvision

def main(roco_root: str):
    root = Path(roco_root)

    check_images(
        root / 'train/radiology', 'traindata.csv', 'train.csv'
    )

    check_images(
        root / 'validate/radiology', 'valdata.csv', 'validate.csv'
    )

    check_images(
        root / 'test/radiology', 'testdata.csv', 'test.csv'
    )

def check_images(split_dir: Path, input_csv: str, target_output: str):
    with open(split_dir / input_csv, 'r') as buf:
        csv_reader = csv.reader(buf)
        next(csv_reader, None)

        filtered_csv = []

        for row in csv_reader:
            image_path = split_dir / 'images' / row[1]
            try:
                torchvision.io.read_image(str(image_path))
            except:
                continue
            filtered_csv.append(row)

    with open(split_dir / target_output, 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        for row in filtered_csv:
            spamwriter.writerow(row)


if __name__ == '__main__':
    main('/home/shpotes/medclip/data/roco-dataset')
