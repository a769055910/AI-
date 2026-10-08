import io
import json
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import app as web


class DetectionStreamTests(unittest.TestCase):
    def setUp(self):
        self.uploads = tempfile.TemporaryDirectory()
        self.addCleanup(self.uploads.cleanup)
        self.patchers = [
            patch.object(web, 'UPLOAD_FOLDER', self.uploads.name),
            patch.object(web, 'analyze_camera_imaging', return_value={'available': False}),
            patch.object(web, 'detect_hidden_watermark', return_value={'detected': False}),
            patch.object(web, 'detect_visible_ai_watermark', return_value={'detected': False}),
            patch.object(web, 'npr_analyze', return_value={'score': 0.2, 'verdict': 'real', 'features': {}}),
            patch.object(web, 'deepfake_zero_model_analysis', return_value={
                'deepfake_score': 0.2, 'verdict': 'real', 'face_detected': False,
            }),
            patch.object(web, 'tamper_zero_model_analysis', return_value={
                'tamper_score': 0.2, 'verdict': 'real',
                'trufor': {'available': True, 'score': 0.2, 'reliability': 0.8},
            }),
            patch.object(web, 'run_content_risk_analysis', return_value=None),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        gpu_response = Mock()
        gpu_response.json.return_value = {
            'code': 200, 'data': {'details': {}, 'specialized_models': {}},
        }
        gpu_patcher = patch.object(web.requests, 'post', return_value=gpu_response)
        self.gpu = gpu_patcher.start()
        self.addCleanup(gpu_patcher.stop)
        self.client = web.app.test_client()
        image = Image.frombytes('RGB', (400, 400), random.Random(7).randbytes(400 * 400 * 3))
        self.image = io.BytesIO()
        image.save(self.image, format='JPEG', quality=90)

    def detect(self, models):
        response = self.client.post('/api/detect/image/stream', data={
            'file': (io.BytesIO(self.image.getvalue()), 'sample.jpg'),
            'models': models,
        }, buffered=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'text/event-stream')
        events = []
        for block in response.get_data(as_text=True).split('\n\n'):
            if not block.strip():
                continue
            lines = block.splitlines()
            events.append((lines[0].removeprefix('event: '), json.loads(lines[1].removeprefix('data: '))))
        self.assertEqual(events[-1][0], 'done')
        self.assertNotIn('error', [event for event, _ in events])
        self.progress = [data for event, data in events if event == 'progress']
        percentages = [data['percent'] for data in self.progress]
        self.assertEqual(percentages, sorted(percentages), 'Progress must not move backwards')
        self.assertEqual(percentages[-1], 100)
        self.assertNotIn('active', self.progress[-1]['stages'].values())
        return next(data for event, data in events if event == 'result')

    def test_each_selected_model_and_combined_request_finish(self):
        for models in [('ai_generated',), ('deepfake',), ('tamper',), web.DETECTION_MODELS]:
            with self.subTest(models=models):
                result = self.detect(list(models))
                self.assertEqual(result['selected_models'], list(models))
                self.assertNotIn('demo_mode', result)
                for key, model in [('ai', 'ai_generated'), ('deepfake', 'deepfake'), ('tamper', 'tamper')]:
                    self.assertEqual(self.progress[-1]['stages'][key], 'done' if model in models else 'skipped')
                self.assertEqual(self.progress[-1]['stages']['content'], 'skipped')
                if models == ('tamper',):
                    self.assertEqual(self.progress[-1]['stages']['gpu'], 'skipped')

    def test_c2pa_expands_request_without_shadowing_selected_models(self):
        with patch.object(web, 'detect_hidden_watermark', return_value={
            'detected': False, 'c2pa_verification': {'present': True},
        }):
            result = self.detect(['tamper'])
        self.assertEqual(set(result['selected_models']), set(web.DETECTION_MODELS))
        self.assertEqual(set(self.gpu.call_args.kwargs['data']['models'].split(',')), set(web.DETECTION_MODELS))
        self.assertTrue(all(self.progress[-1]['stages'][key] == 'done' for key in ('gpu', 'ai', 'deepfake', 'tamper')))

    def test_watermark_shortcuts_mark_models_as_skipped(self):
        for detector in ('detect_hidden_watermark', 'detect_visible_ai_watermark'):
            with self.subTest(detector=detector), patch.object(web, detector, return_value={
                'detected': True, 'source': 'test-platform', 'confidence': 0.97,
            }):
                result = self.detect(['ai_generated', 'deepfake', 'tamper'])
                self.assertTrue(result['shortcut'])
                states = self.progress[-1]['stages']
                self.assertTrue(all(states[key] == 'skipped' for key in ('gpu', 'ai', 'deepfake', 'tamper')))
                self.assertTrue(all(states[key] == 'done' for key in ('watermark', 'fusion', 'content')))

    def test_unavailable_gpu_is_visible_in_progress(self):
        self.gpu.side_effect = web.requests.exceptions.ConnectionError('offline')
        self.detect(['ai_generated'])
        self.assertEqual(self.progress[-1]['stages']['gpu'], 'warning')


if __name__ == '__main__':
    unittest.main()
