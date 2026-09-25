"""Truncated local-model output must recover without accepting partial scores."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tomllib
import unittest
from unittest.mock import Mock, patch

from photo_selector.vision import Assessor, schema_for


class VisionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = tomllib.loads((Path(__file__).resolve().parents[1]/'selector.example.toml').read_text())
        self.result = dict(excluded=False, exclusion_reason='', category='portrait', summary='Relaxed expression.',
                           scores={name: {'score': 7, 'reason': 'Clear visible detail.'} for name in self.cfg['criteria']})
        self.payload = {'messages': [{'role': 'system', 'content': 'Original instructions'}],
                        'max_tokens': 1400, 'response_format': {'type': 'json_object', 'schema': schema_for(self.cfg)}}

    def response(self, reason, content=None):
        return io.BytesIO(json.dumps({'choices': [{'finish_reason': reason,
                             'message': {'content': content if content is not None else json.dumps(self.result)}}]}).encode())

    def test_truncation_retries_with_more_room_and_bounded_explanations(self):
        original = deepcopy(self.payload)
        opener = Mock()
        opener.open.side_effect = [self.response('length', '{"scores":'), self.response('stop')]
        with patch('photo_selector.vision.urllib.request.build_opener', return_value=opener), redirect_stdout(io.StringIO()):
            result = Assessor('http://127.0.0.1:1234', self.cfg).complete(self.payload)
        self.assertEqual(result, self.result)
        requests = [json.loads(call.args[0].data) for call in opener.open.call_args_list]
        self.assertEqual([r['max_tokens'] for r in requests], [1400, 2800])
        self.assertEqual(requests[0], original)
        self.assertEqual(self.payload, original)
        props = requests[1]['response_format']['schema']['properties']
        self.assertEqual(props['summary']['maxLength'], 480)
        self.assertEqual(props['scores']['properties']['expression']['properties']['reason']['maxLength'], 240)

    def test_persistent_truncation_has_a_bounded_failure(self):
        opener = Mock()
        opener.open.side_effect = [self.response('length') for _ in range(3)]
        with patch('photo_selector.vision.urllib.request.build_opener', return_value=opener), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'three recovery attempts'):
                Assessor('http://127.0.0.1:1234', self.cfg).complete(self.payload)
        self.assertEqual([json.loads(c.args[0].data)['max_tokens'] for c in opener.open.call_args_list], [1400,2800,4096])

    def test_other_finish_reasons_and_invalid_scores_are_not_accepted(self):
        for reason, content in [('error', None), ('stop', '{}')]:
            with self.subTest(reason=reason):
                opener = Mock(); opener.open.return_value = self.response(reason, content)
                with patch('photo_selector.vision.urllib.request.build_opener', return_value=opener):
                    with self.assertRaises(ValueError):
                        Assessor('http://127.0.0.1:1234', self.cfg).complete(self.payload)
                self.assertEqual(opener.open.call_count, 1)


if __name__ == '__main__':
    unittest.main()
