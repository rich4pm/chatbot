# Demo
if __name__ == "__main__":
    from help_desk import HelpDesk

    model = HelpDesk(new_db=True)

    print(model.db._collection.count())

    prompt = "How can I help you?"
    result, sources = model.retrieval_qa_inference(prompt)
    print(result, sources)
