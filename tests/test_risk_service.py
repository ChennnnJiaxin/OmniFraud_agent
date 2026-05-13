import unittest

from schemas.risk_schema import RiskProfileInput
from services.risk_service import generate_risk_report


class RiskServiceTestCase(unittest.TestCase):
    def test_elderly_profile_returns_structured_report(self):
        profile = RiskProfileInput(
            age=68,
            occupation="退休人员",
            fraud_types=["保健品诈骗", "情感诈骗"],
            loss_amount=3000,
            report_police=False,
            social_media_hours=8,
        )
        result = generate_risk_report(profile)
        self.assertTrue(result.success)
        self.assertIn(result.risk_level, {"中风险", "高风险"})
        self.assertGreater(len(result.reasons), 0)
        self.assertGreater(len(result.suggestions), 0)

    def test_insufficient_info_returns_stable_structure(self):
        profile = RiskProfileInput()
        result = generate_risk_report(profile)
        self.assertTrue(result.success)
        self.assertIn(result.risk_level, {"低风险", "中风险", "高风险"})
        self.assertIsInstance(result.suggestions, list)


if __name__ == "__main__":
    unittest.main()
