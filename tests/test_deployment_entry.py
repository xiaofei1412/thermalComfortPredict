import unittest
import tempfile

from scripts.serve_workbench import engine_url
from thermal_comfort.simulation import Client
from thermal_comfort.workbench import Workbench


class DeploymentEntryTests(unittest.TestCase):
    def test_independent_engine_is_explicit_and_default_is_unchanged(self):
        self.assertEqual(engine_url('http://127.0.0.1:8081/'), 'http://127.0.0.1:8081')
        self.assertEqual(Client(engine_url('http://localhost:8081')).url, 'http://localhost:8081')
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(Workbench(folder).client.url, 'http://127.0.0.1:8080')

    def test_nonlocal_or_ambiguous_engine_is_rejected(self):
        for value in ('https://127.0.0.1:8081', 'http://example.com:8081',
                      'http://127.0.0.1', 'http://127.0.0.1:80',
                      'http://user:pass@127.0.0.1:8081', 'http://127.0.0.1:8081/api',
                      'http://127.0.0.1:8081?engine=other', 'http://127.0.0.1:8081#fragment'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                engine_url(value)


if __name__ == '__main__':
    unittest.main()
