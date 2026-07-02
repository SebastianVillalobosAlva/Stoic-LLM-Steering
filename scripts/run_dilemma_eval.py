import argparse


def main() -> None:
    from stoic_llm.model import ModelLoader
    from stoic_llm.eval.dilemma import DilemmaEval

    parser = argparse.ArgumentParser(description="CAA forced-choice dilemma eval.")
    parser.add_argument(
        "--model",
        choices=["1B", "3B"],
        default="3B",
        help="Base model size (default: 3B; 1B is legacy).",
    )
    args = parser.parse_args()

    model, tokenizer = ModelLoader(args.model).load()

    # configs = {
    #     "marcus": {
    #         "layer": 26,
    #         "coeff": 0.11,
    #         "vector_file": f"marcus_aurelius_steering_{args.model}.pt",
    #     },
    #     "seneca": {"layer": 4, "coeff": 0.11, "vector_file": f"seneca_steering_{args.model}.pt"},
    #     "epictetus": {
    #         "layer": 8,
    #         "coeff": 0.11,
    #         "vector_file": f"epictetus_steering_{args.model}.pt",
    #     },
    # }

    # ev = DilemmaEval(model, tokenizer)
    # results = ev.run_all(configs)
    # ev.save_results(results)
    # print()
    # print(ev.summarize(results))

    ev = DilemmaEval(model, tokenizer)
    sweep = ev.sweep_coefficients(
        "epictetus",
        layer=8,
        vector_file=f"epictetus_steering_{args.model}.pt",
        coefficients=[0.11, 0.2, 0.4, 0.8, 1.5],
    )
    print(ev.summarize_sweep(sweep))


if __name__ == "__main__":
    main()
