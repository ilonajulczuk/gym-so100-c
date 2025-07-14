import os
import pickle
import argparse


def load_demonstrations(filename):
    """Load expert demonstrations from pickle file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Demonstrations file not found: {filename}")

    with open(filename, "rb") as f:
        demonstrations = pickle.load(f)

    print(f"Loaded {len(demonstrations)} episodes from {filename}")
    return demonstrations


def merge_demonstrations(demo_list):
    """Merge multiple demonstration datasets."""
    merged = []
    for demos in demo_list:
        merged.extend(demos)

    print(f"Merged demonstrations: {len(merged)} episodes")
    return merged


def main():
    """Main function to handle command line arguments and merge demonstrations."""
    parser = argparse.ArgumentParser(
        description="Merge multiple expert demonstration pickle files"
    )
    parser.add_argument(
        "demo_files",
        nargs="+",
        help="Input demonstration pickle files to merge"
    )
    parser.add_argument(
        "-o", "--output",
        default="merged_expert_demonstrations.pkl",
        help="Output filename for merged demonstrations (default: merged_expert_demonstrations.pkl)"
    )
    
    args = parser.parse_args()
    
    # Load all demonstration files
    demonstrations = [load_demonstrations(f) for f in args.demo_files]
    merged_demos = merge_demonstrations(demonstrations)

    # Save merged demonstrations
    with open(args.output, "wb") as f:
        pickle.dump(merged_demos, f)

    print(f"Merged demonstrations saved to '{args.output}'")


if __name__ == "__main__":
    main()
