from src.data.elliptic import (
    build_elliptic_pyg_data,
    add_weber_2019_masks,
    add_marasi_2024_masks,
    describe_data,
)


def print_summary(title, summary):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key}: {value}")


def main():
    print("Loading Elliptic dataset...")

    data = build_elliptic_pyg_data(
        data_dir="data/raw/elliptic",
        make_undirected=False,
        include_time_as_feature=False,
    )

    print_summary("Raw Elliptic Graph", describe_data(data))

    weber_data = add_weber_2019_masks(data.clone())
    print_summary("Weber 2019 Temporal Split", describe_data(weber_data))

    marasi_data = add_marasi_2024_masks(data.clone(), seed=42)
    print_summary("Marasi 2024 Stratified Split", describe_data(marasi_data))


if __name__ == "__main__":
    main()