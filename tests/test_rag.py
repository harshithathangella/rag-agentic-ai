from src.graph import build_rag_graph


def run_query(question):
    graph = build_rag_graph()

    result = graph.invoke(
        {
            "question": question,
            "context": [],
            "answer": "",
            "score": 0.0,
        }
    )

    return result


def test_agentic_ai_question():
    result = run_query(
        "What is Agentic AI according to the eBook?"
    )

    assert result["answer"]
    assert len(result["context"]) > 0
    assert result["score"] >= 0.0


def test_automation_question():
    result = run_query(
        "How do AI agents differ from traditional automation systems?"
    )

    assert result["answer"]
    assert len(result["context"]) > 0
    assert result["score"] >= 0.0


def test_agentic_architecture_question():
    result = run_query(
        "What are the core components of an Agentic Architecture?"
    )

    assert result["answer"]
    assert len(result["context"]) > 0
    assert result["score"] >= 0.0


def test_memory_question():
    result = run_query(
        "What role does memory play in Agentic AI workflows?"
    )

    assert result["answer"]
    assert len(result["context"]) > 0
    assert result["score"] >= 0.0


def test_unrelated_question():
    result = run_query(
        "Who won the 2022 FIFA World Cup?"
    )

    assert (
        "cannot answer" in result["answer"].lower()
        or "not provided" in result["answer"].lower()
        or "not available" in result["answer"].lower()
    )

    assert result["score"] >= 0.0