import unittest
from unittest.mock import patch

from services.case_service import search_cases


class _FakeNeo4jClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def query(self, *_args, **_kwargs):
        self.calls += 1
        return [{"count": len(self.rows)}] if self.calls == 1 else self.rows


class CaseServiceTestCase(unittest.TestCase):
    @patch("services.case_service.Neo4jClient")
    def test_normal_query_returns_cases(self, mock_client_cls):
        rows = [
            {
                "name": "虚假投资诈骗案",
                "description": "通过投资平台诱导转账",
                "type": "刑事",
                "types": ["投资理财"],
                "subtypes": ["虚假投资"],
                "suspects": ["张三"],
                "victims": ["李四"],
                "money": 128000.0,
                "locations": ["广东"],
                "laws": ["刑法相关条款"],
            }
        ]
        mock_client_cls.return_value = _FakeNeo4jClient(rows)
        result = search_cases("投资")
        self.assertTrue(result.success)
        self.assertEqual(result.total_count, 1)
        self.assertEqual(result.cases[0].title, "虚假投资诈骗案")

    @patch("services.case_service.Neo4jClient")
    def test_no_result_query_returns_empty_list(self, mock_client_cls):
        mock_client_cls.return_value = _FakeNeo4jClient([])
        result = search_cases("不存在的案件关键词")
        self.assertTrue(result.success)
        self.assertEqual(result.total_count, 0)
        self.assertEqual(result.cases, [])


if __name__ == "__main__":
    unittest.main()
