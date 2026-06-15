from src.download import main as download_data
from src.features import main as generate_features
from src.train import main as train_model
from reports.report import main as generate_report


def main():
    download_data()
    generate_features()
    train_model()
    generate_report()


if __name__ == "__main__":
    main()