# Owner: Samya
# Responsibility: Small utilities and smoke tests to validate loaded models produce expected outputs.
# Goals:
# - Quick tests that run a tiny batch through the model and verify output shape and type
# Integration points:
# - Uses `InferenceEngine` to load models and perform a forward pass
# Testing requirements:
# - Include tests for edge-cases: empty input, long input, multilingual inputs

from .inference_engine import InferenceEngine


class TestModels:
    """
    Run lightweight checks on models to ensure inference pipeline is working.

    Methods:
        - smoke_test_text_model()
        - smoke_test_email_model()
    """

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def smoke_test_text_model(self) -> bool:
        # TODO: run a small batch and check types
        r = self.engine.classify_text('Test')
        return isinstance(r.get('scam_probability', None), float)

    def smoke_test_email_model(self) -> bool:
        fake_email = type('E', (), {'subject': 'Win money', 'text': 'Click this link'})
        r = self.engine.classify_email(fake_email)
        return isinstance(r.get('scam_probability', None), float)


if __name__ == '__main__':
    engine = InferenceEngine()
    t = TestModels(engine)
    print('text smoke:', t.smoke_test_text_model())
    print('email smoke:', t.smoke_test_email_model())
