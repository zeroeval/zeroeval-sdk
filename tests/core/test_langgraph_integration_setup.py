import builtins


def test_langgraph_safe_setup_fails_when_stategraph_import_fails(monkeypatch):
    # Import inside the test so it doesn't eagerly import langgraph.
    from zeroeval.observability.integrations.langgraph.integration import LangGraphIntegration

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "langgraph.graph.state":
            raise ImportError("forced failure for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    integ = LangGraphIntegration(tracer=object())
    assert integ.safe_setup() is False
    err = integ.get_setup_error()
    assert isinstance(err, ImportError)

