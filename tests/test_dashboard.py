from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=10).run()

    def test_dashboard_loads_without_runtime_errors(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.title[0].value, "Global GARCH Volatility Monitor")
        self.assertGreaterEqual(len(self.app.metric), 3)

    def test_sidebar_exposes_index_and_date_filters(self):
        self.assertEqual(self.app.multiselect[0].label, "Indices")
        self.assertEqual(self.app.date_input[0].label, "Date range")
        self.assertEqual(len(self.app.multiselect[0].value), 4)

    def test_empty_index_selection_is_handled(self):
        self.app.multiselect[0].set_value([]).run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("Select at least one index", self.app.info[0].value)


if __name__ == "__main__":
    unittest.main()
