from zeroeval.observability.integrations.base import Integration


class _Dummy:
    def value(self):
        return "original"


class _DummyIntegration(Integration):
    PACKAGE_NAME = "zeroeval"  # always available in tests

    def setup(self) -> None:
        self._patch_method(_Dummy, "value", self._wrap)

    def _wrap(self, original):
        def patched(self, *args, **kwargs):
            return f"patched:{original(self, *args, **kwargs)}"

        return patched


def test_integration_teardown_restores_original_method():
    tracer = object()
    integ = _DummyIntegration(tracer)
    obj = _Dummy()

    assert obj.value() == "original"

    assert integ.safe_setup() is True
    assert obj.value() == "patched:original"

    integ.teardown()
    assert obj.value() == "original"

