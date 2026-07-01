def main() -> None:
    from stoic_llm.model import ModelLoader
    from stoic_llm.eval.dilemma import DilemmaEval

    model, tokenizer = ModelLoader("3B").load()

    # configs = {
    #     "marcus": {
    #         "layer": 26,
    #         "coeff": 0.11,
    #         "vector_file": "marcus_aurelius_steering_3B.pt",
    #     },
    #     "seneca": {"layer": 4, "coeff": 0.11, "vector_file": "seneca_steering_3B.pt"},
    #     "epictetus": {
    #         "layer": 8,
    #         "coeff": 0.11,
    #         "vector_file": "epictetus_steering_3B.pt",
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
        vector_file="epictetus_steering_3B.pt",
        coefficients=[0.11, 0.2, 0.4, 0.8, 1.5],
    )
    print(ev.summarize_sweep(sweep))


if __name__ == "__main__":
    main()
