from pamae_rag.cli import main

if __name__ == "__main__":
    main(["validate-data", *(__import__("sys").argv[1:])])
