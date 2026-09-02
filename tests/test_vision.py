"""Visão: fontes, frames, frame inválido, ausência de dispositivo, processamento, mudança, captura."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from agent_core.observations import ObservationKind
from agent_core.vision import (
    CameraSource,
    ChangeDetector,
    FramePreprocessor,
    ImageSource,
    ScreenSource,
    VisionCapture,
    VisionObserver,
    VisionUnavailableError,
    VisualAnalyzer,
    VisualPipeline,
    Frame,
    encode_image,
    vision_available,
    build_vision_capture,
)
from tests._helpers import run

if vision_available():
    import cv2
    import numpy as np


def img(w=160, h=120, color=(200, 200, 200)):
    return np.full((h, w, 3), color, np.uint8)


@unittest.skipUnless(vision_available(), "OpenCV não instalado")
class FrameTests(unittest.TestCase):
    def test_valid_and_invalid_frames(self):
        self.assertTrue(Frame(img(), "t").is_valid())
        self.assertTrue(Frame(np.zeros((10, 10), np.uint8), "t").is_valid())
        self.assertFalse(Frame(None, "t").is_valid())
        self.assertFalse(Frame(np.zeros((0, 10, 3), np.uint8), "t").is_valid())
        self.assertFalse(Frame(np.zeros((10, 10, 3), np.float32), "t").is_valid())
        self.assertFalse(Frame(np.zeros((10, 10, 5), np.uint8), "t").is_valid())
        self.assertEqual(Frame(img(), "t").resolution, (160, 120))

    def test_encode(self):
        jpeg = encode_image(Frame(img(), "t"))
        self.assertTrue(jpeg.data.startswith(b"\xff\xd8"))
        self.assertEqual((jpeg.width, jpeg.height, jpeg.media_type), (160, 120, "image/jpeg"))
        png = encode_image(Frame(img(), "t"), fmt="png")
        self.assertTrue(png.data.startswith(b"\x89PNG"))


@unittest.skipUnless(vision_available(), "OpenCV não instalado")
class SourceTests(unittest.TestCase):
    def test_image_source_from_arrays(self):
        src = ImageSource([img(), img(color=(0, 0, 0))])
        with src:
            f1, f2, f3 = src.read(), src.read(), src.read()
        self.assertEqual((f1.index, f2.index), (0, 1))
        self.assertIsNone(f3)
        self.assertFalse(src.is_open)

    def test_image_source_loop_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.png"
            cv2.imwrite(str(path), img())
            src = ImageSource([path], loop=True)
            frames = [src.read() for _ in range(3)]
            self.assertTrue(all(f is not None for f in frames))
            src.close()
            self.assertEqual(ImageSource.from_directory(d).items, [path])
        with self.assertRaises(VisionUnavailableError):
            ImageSource(["/nao/existe.png"]).open()
        self.assertIsNone(ImageSource(["/nao/existe.png"]).read())  # read nunca levanta
        with self.assertRaises(ValueError):
            ImageSource([])

    def test_invalid_array_is_rejected(self):
        src = ImageSource([np.zeros((10, 10, 3), np.float32)])
        self.assertIsNone(src.read())
        self.assertEqual(src.errors, 1)

    def test_camera_unavailable(self):
        src = CameraSource(index=99)
        with self.assertRaises(VisionUnavailableError):
            src.open()
        self.assertIsNone(src.read())
        self.assertGreaterEqual(src.errors, 1)
        src.close()  # idempotente

    def test_screen_contract(self):
        src = ScreenSource(monitor=1)
        try:
            src.open()
        except VisionUnavailableError:
            self.assertFalse(src.is_open)  # sem display: falha limpa
            return
        frame = src.read()
        src.close()
        self.assertTrue(frame is None or frame.is_valid())


@unittest.skipUnless(vision_available(), "OpenCV não instalado")
class ProcessingTests(unittest.TestCase):
    def test_preprocessor(self):
        pre = FramePreprocessor(max_width=80)
        out = pre.process(Frame(img(160, 120), "t"))
        self.assertEqual(out.resolution, (80, 60))
        self.assertEqual(out.metadata["scale"], 0.5)
        gray = FramePreprocessor(max_width=0, grayscale=True).process(Frame(img(color=(10, 200, 30)), "t"))
        self.assertTrue((gray.image[..., 0] == gray.image[..., 1]).all())
        bgra = FramePreprocessor().process(Frame(np.zeros((8, 8, 4), np.uint8), "t"))
        self.assertEqual(bgra.image.shape[2], 3)
        two_d = FramePreprocessor().process(Frame(np.zeros((8, 8), np.uint8), "t"))
        self.assertEqual(two_d.image.shape[2], 3)

    def test_change_detection(self):
        det = ChangeDetector(threshold=0.01)
        base = img()
        first = det.detect(Frame(base, "t"))
        self.assertTrue(first.changed and first.first)
        same = det.detect(Frame(base.copy(), "t"))
        self.assertFalse(same.changed)
        self.assertEqual(same.score, 0.0)
        moved = base.copy()
        cv2.rectangle(moved, (40, 30), (100, 80), (0, 0, 0), -1)
        res = det.detect(Frame(moved, "t"))
        self.assertTrue(res.changed)
        self.assertGreater(res.score, 0.1)
        self.assertEqual(len(res.regions), 1)
        r = res.regions[0]
        self.assertLessEqual(r["x"], 40)
        self.assertGreaterEqual(r["x"] + r["w"], 100)
        det.reset()
        self.assertTrue(det.detect(Frame(base, "t")).first)

    def test_analyzer(self):
        an = VisualAnalyzer()
        white = an.analyze(Frame(img(color=(255, 255, 255)), "t"))
        black = an.analyze(Frame(img(color=(0, 0, 0)), "t"))
        self.assertGreater(white["brightness"], black["brightness"])
        self.assertEqual(white["resolution"], {"width": 160, "height": 120})
        self.assertIn(white["ocr"], ("tesseract", "unavailable"))
        if white["ocr"] == "unavailable":
            self.assertIsNone(white["text"])
        self.assertAlmostEqual(sum(c["share"] for c in white["dominant_colors"]), 1.0, places=2)
        scene = img()
        cv2.rectangle(scene, (20, 20), (80, 80), (0, 0, 255), -1)
        rich = an.analyze(Frame(scene, "t"))
        self.assertGreaterEqual(len(rich["elements"]), 1)
        self.assertIn("elementos salientes", rich["description"])

    def test_pipeline_observation(self):
        pipe = VisualPipeline(preprocessor=FramePreprocessor(max_width=64))
        obs = pipe.process(Frame(img(), "cam"))
        self.assertEqual(obs.kind, ObservationKind.VISION)
        self.assertEqual(obs.source, "cam")
        self.assertTrue(obs.has_image)
        self.assertEqual(obs.image.width, 64)
        self.assertTrue(obs.metadata["changed"])
        self.assertIn("resolution", obs.extracted)
        bad = pipe.process(Frame(np.zeros((0, 0, 3), np.uint8), "cam"))
        self.assertEqual(bad.confidence, 0.0)
        self.assertTrue(bad.metadata["invalid"])
        self.assertFalse(bad.has_image)
        no_img = VisualPipeline(attach_image=False).process(Frame(img(), "cam"))
        self.assertFalse(no_img.has_image)


@unittest.skipUnless(vision_available(), "OpenCV não instalado")
class CaptureTests(unittest.TestCase):
    def test_continuous_capture_and_stop(self):
        seen = []

        async def hook(obs):
            seen.append(obs)

        async def scenario():
            frames = [img(), img(color=(0, 0, 0)), img()]
            with tempfile.TemporaryDirectory() as d:
                cap = VisionCapture(ImageSource(frames, loop=True), fps=50, observation_interval=0.05, store_dir=Path(d), on_observation=hook)
                self.assertTrue(await cap.start())
                self.assertTrue(cap.is_running)
                await asyncio.sleep(0.4)
                await cap.stop()
                status = cap.status()
                self.assertFalse(cap.is_running)
                self.assertGreater(status["frames_read"], 3)
                self.assertGreaterEqual(status["observations"], 2)
                self.assertGreaterEqual(status["stored"], 1)
                self.assertTrue(list(Path(d).glob("*.jpg")))
                self.assertIsNotNone(cap.latest())
                self.assertGreaterEqual(len(cap.history()), 2)
                self.assertGreaterEqual(len(seen), 2)
                self.assertIsNone(status["error"])
                await cap.stop()  # idempotente

        run(scenario())

    def test_snapshot_without_start(self):
        cap = VisionCapture(ImageSource([img()]))
        obs = run(cap.snapshot())
        self.assertIsNotNone(obs)
        self.assertEqual(cap.frames_read, 1)
        self.assertIsNone(run(cap.snapshot()))  # fonte esgotada
        run(cap.stop())

    def test_device_unavailable_does_not_raise(self):
        cap = VisionCapture(CameraSource(index=99))
        self.assertFalse(run(cap.start()))
        self.assertFalse(cap.is_running)
        self.assertIn("câmera", cap.error)
        self.assertIsNone(run(cap.snapshot()))
        run(cap.stop())
        self.assertEqual(run(VisionObserver(cap).observe()), [])

    def test_observer_returns_observation(self):
        cap = VisionCapture(ImageSource([img()], loop=True))
        obs = run(VisionObserver(cap).observe())
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].kind, ObservationKind.VISION)
        run(cap.stop())

    def test_build_from_config(self):
        from agent_core import AgentConfig

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.png"
            cv2.imwrite(str(path), img())
            cfg = AgentConfig(project_root=d, vision_enabled=True, vision_source="image", vision_images=(str(path),), vision_fps=3, observation_interval=1, vision_store_frames=True)
            cap = build_vision_capture(cfg)
            self.assertEqual(cap.fps, 3)
            self.assertEqual(cap.store_dir, cfg.backup_dir / "frames")
            self.assertIsNotNone(run(cap.snapshot()))
            run(cap.stop())
            with self.assertRaises(VisionUnavailableError):
                build_vision_capture(AgentConfig(project_root=d, vision_source="image"))
            with self.assertRaises(ValueError):
                AgentConfig(project_root=d, vision_source="lidar")
            with self.assertRaises(ValueError):
                VisionCapture(ImageSource([img()]), fps=0)
