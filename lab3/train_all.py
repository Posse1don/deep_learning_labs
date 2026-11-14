import argparse
import prepare_split, check_batch, check_loss, check_modes, train_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=".")
    args = parser.parse_args()

    prepare_split.main(args.data)
    check_batch.main(args.data)
    check_loss.main(args.data)
    check_modes.main(args.data)
    train_experiment.main()