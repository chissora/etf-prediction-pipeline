from src.download import main as download_data
from src.features import main as generate_features
from src.train import main as train_model


def main():
    download_data()
    generate_features()
    train_model()


if __name__ == "__main__":
    main()