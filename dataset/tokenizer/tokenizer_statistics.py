from transformers import AutoTokenizer


def main():
    mbert_tokenizer_name = "google-bert/bert-base-multilingual-cased"
    cpt_mbert_tokenizer_name = "OMRIDRORI/mbert-tibetan-continual-wylie-final"

    print("Loading text...")
    text_path = "tibetan_bert_ready_512.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Loading tokenizers...")
    mbert_tokenizer = AutoTokenizer.from_pretrained(mbert_tokenizer_name)
    cpt_mbert_tokenizer = AutoTokenizer.from_pretrained(cpt_mbert_tokenizer_name)

    print("Tokenizing text...")
    mBERT_tokens = mbert_tokenizer.encode(text)
    cpt_mBERT_tokens = cpt_mbert_tokenizer.encode(text)

    print("Calculating statistics...")
    n_tokens_mBERT = len(mBERT_tokens)
    n_tokens_cpt_mBERT = len(cpt_mBERT_tokens)
    n_words = len(text.split())

    # Compression
    mBERT_compression = (n_words / n_tokens_mBERT) if n_tokens_mBERT else float("nan")
    cpt_mBERT_compression = (n_words / n_tokens_cpt_mBERT) if n_tokens_cpt_mBERT else float("nan")

    print(f"mBERT Tokens: {n_tokens_mBERT}")
    print(f"cpt-mBERT Tokens: {n_tokens_cpt_mBERT}")
    print(f"Words: {n_words}")
    print(f"Compression chars/token: {mBERT_compression:.6f}")
    print(f"Compression words/token: {cpt_mBERT_compression:.6f}")


    # Fertility
    mBERT_fertility = (n_tokens_mBERT / n_words) if n_words else float("nan")
    cpt_mBERT_fertility = (n_tokens_cpt_mBERT / n_words) if n_words else float("nan")

    print(f"mBERT Fertility tokens/words: {mBERT_fertility:.6f}")
    print(f"cpt-mBERT Fertility tokens/words: {cpt_mBERT_fertility:.6f}")


if __name__ == "__main__":
    main()