from src.download import main as download_data
from src.features import main as generate_features


def main():
    download_data()
    generate_features()


if __name__ == "__main__":
    main()