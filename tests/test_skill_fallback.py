import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as web


class SkillFallbackTests(unittest.TestCase):
    def setUp(self):
        self.client = web.app.test_client()
        self.skill_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.skill_dir.cleanup)
        skill_dir_patch = patch.object(web, '_CODEX_SKILLS_DIR', self.skill_dir.name)
        skill_dir_patch.start()
        self.addCleanup(skill_dir_patch.stop)

    def test_missing_installed_skills_use_matching_knowledge_base_sections(self):
        cases = {
            'fraud': ('AI 深度伪造类犯罪知识库', 'AI 生成虚假信息类犯罪知识库'),
            'rumor': ('AI 生成虚假信息类犯罪知识库', 'AI 制作淫秽物品类犯罪知识库'),
            'porn': ('AI 制作淫秽物品类犯罪知识库', 'AI 图像生成相关典型案例库'),
            'common': ('AI 图像生成相关典型案例库', None),
        }
        for key, (included, excluded) in cases.items():
            with self.subTest(key=key):
                response = self.client.get(f'/api/skills/{key}')
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assertEqual(data['source'], 'knowledge_base')
                self.assertFalse(data['editable'])
                self.assertIn(included, data['markdown'])
                if excluded:
                    self.assertNotIn(excluded, data['markdown'])

    def test_installed_skill_takes_precedence(self):
        skill_path = Path(self.skill_dir.name) / 'fraud-ai-image-forensics' / 'SKILL.md'
        skill_path.parent.mkdir()
        skill_path.write_text('---\nname: fraud-ai-image-forensics\n---\n\n## Installed\n', encoding='utf-8')
        response = self.client.get('/api/skills/fraud')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['source'], 'codex_skill')
        self.assertTrue(response.get_json()['editable'])
        self.assertIn('## Installed', response.get_json()['markdown'])

    def test_knowledge_base_fallback_cannot_be_edited_as_installed_skill(self):
        response = self.client.post('/api/skills/fraud', json={
            'markdown': '---\nname: fraud-ai-image-forensics\n---\nchanged'
        })
        self.assertEqual(response.status_code, 405)


if __name__ == '__main__':
    unittest.main()
