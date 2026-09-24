from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image

from photo_selector.config import load_config
from photo_selector.recommend import project_path, run, select_top, weighted_score
from photo_selector.vision import assessment_signature, validate_assessment
from photo_selector.core import digest

DEFAULT = Path(__file__).resolve().parents[1]/'selector.example.toml'


def config():
    return tomllib.loads(DEFAULT.read_text())


def assessment(cfg, a=8, b=4, excluded=False):
    return dict(excluded=excluded, exclusion_reason='Document' if excluded else '', category='portrait', summary='Visible evidence.',
                scores={name: {'score': a if name=='expression' else b, 'reason': 'Visible evidence.'} for name in cfg['criteria']})


class Engine:
    def __init__(self, *args):
        pass

    def faces(self, image):
        return [dict(box=[10,10,90,90], blur=100., brightness=120., face_pixels=90., face_fraction=.2,
                     embedding=np.array([1.,0.]))]


class RecommendationTests(unittest.TestCase):
    def test_weights_do_not_invalidate_visual_cache_but_rubric_does(self):
        cfg=config();other=deepcopy(cfg)
        other['criteria']['expression']['weight']=100
        other['selection']['top']=3
        self.assertEqual(assessment_signature(cfg),assessment_signature(other))
        self.assertGreater(weighted_score(assessment(cfg),other['criteria']),weighted_score(assessment(cfg),cfg['criteria']))
        other['criteria']['expression']['description']='Prefer a neutral expression'
        self.assertNotEqual(assessment_signature(cfg),assessment_signature(other))

    def test_validation_rejects_missing_scores_and_nonfinite_values(self):
        cfg=config();value=assessment(cfg)
        del value['scores']['expression']
        with self.assertRaises(ValueError):validate_assessment(value,cfg)
        for bad in [float('nan'),float('inf'),True,11,-1]:
            value=assessment(cfg);value['scores']['expression']['score']=bad
            with self.assertRaises(ValueError):validate_assessment(value,cfg)

    def test_config_paths_and_mistakes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'library').mkdir();(root/'refs').mkdir()
            text=DEFAULT.read_text().replace('${PHOTO_LIBRARY}','library').replace('${PHOTO_REFERENCES}','refs')
            p=root/'config.toml';p.write_text(text)
            cfg=load_config(p)
            self.assertEqual(cfg['paths']['library'],str(root/'library'))
            for before,after in [('top = 10','top = 0'),('weight = 3.0','weight = -1'),('threshold = 0.45','threshold = nan'),('not_before = ""','not_before = 0')]:
                p.write_text(text.replace(before,after))
                with self.assertRaises(ValueError):load_config(p)

    def test_ranking_exclusions_duplicates_and_diversity(self):
        cfg=config();cfg['selection']['top']=2;cfg['selection']['minimum_score']=0
        def item(name,score,sha,phash,category='portrait',excluded=False):
            value=assessment(cfg,score,score,excluded);value['category']=category
            return dict(path=name,assessment=value,sha256=sha,phash=phash,width=200,height=200,date=None)
        items=[item('best',9,'a','0'),item('copy',8,'a','0'),item('activity',8.9,'b','ffffffffffffffff','activity'),item('bad',10,'c','aaaaaaaaaaaaaaaa',excluded=True)]
        chosen=select_top(items,cfg)
        self.assertEqual([i['path'] for i in chosen],['best','activity'])
        self.assertEqual(items[1]['status'],'duplicate')
        self.assertEqual(items[3]['status'],'visual_exclusion')
        cfg['selection']['top']=10
        self.assertEqual(len(select_top(items,cfg)),2)

    @patch('photo_selector.cli.Engine',Engine)
    def test_end_to_end_cache_reranking_source_preservation_and_new_files(self):
        cfg=config();cfg['selection']['minimum_score']=0;cfg['selection']['near_duplicate_distance']=-1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);library=root/'library';refs=root/'refs';state=root/'state'
            library.mkdir();refs.mkdir();state.mkdir()
            for folder,count in [(library,3),(refs,2)]:
                for i in range(count):Image.new('RGB',(200,200),(i*60,100,150)).save(folder/f'{i}.jpg')
            cfg['paths']=dict(library=str(library),references=str(refs))
            before={str(p):(digest(p),p.stat().st_mtime_ns) for folder in [library,refs] for p in folder.iterdir()}
            calls=[]
            class FakeAssessor:
                def assess(self,item):
                    calls.append(item['path'])
                    return assessment(cfg,9 if Path(item['path']).stem=='0' else 3,3 if Path(item['path']).stem=='0' else 9)
            @contextmanager
            def fake(*args):yield FakeAssessor()
            args=SimpleNamespace(config=DEFAULT,command='run',scan_only=False)
            with patch('photo_selector.recommend.load_config',return_value=cfg),patch('photo_selector.recommend.local_assessor',fake),redirect_stdout(io.StringIO()):
                out=run(args,state)
                first=json.loads((out/'recommendations.json').read_text())[0]['path']
                self.assertEqual(len(calls),3)
                cfg['criteria']['expression']['weight']=100
                args.command='rank'
                out=run(args,state)
                self.assertEqual(len(calls),3)
                self.assertNotEqual(first,json.loads((out/'recommendations.json').read_text())[0]['path'])
                self.assertEqual(before,{str(p):(digest(p),p.stat().st_mtime_ns) for folder in [library,refs] for p in folder.iterdir()})
                Image.new('RGB',(200,200)).save(library/'new.jpg')
                with self.assertRaisesRegex(ValueError,'not been scanned'):run(args,state)
                cfg['criteria']['expression']['description']='Different expression rubric'
                args.command='run';run(args,state)
                self.assertEqual(len(calls),7)
                # Replacing the reference images cannot reuse the old identity.
                Image.new('RGB',(200,200),(90,80,70)).save(refs/'0.jpg')
                args.command='rank'
                with self.assertRaisesRegex(ValueError,'Reference photos changed'):run(args,state)


if __name__=='__main__':unittest.main()
