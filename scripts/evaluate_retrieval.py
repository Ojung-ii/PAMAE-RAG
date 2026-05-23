from pamae_rag.cli import main

if __name__ == "__main__":
    main(["eval", *(__import__("sys").argv[1:])])
