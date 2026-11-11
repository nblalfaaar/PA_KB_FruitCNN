"""Inferensi model buah (TFLite) tanpa TensorFlow: ringan, cocok untuk hosting gratis.
Preprocessing sama dengan notebook: RGB (latar putih untuk PNG transparan), resize BILINEAR ke IMG_SIZE, piksel 0-255 (Rescaling ada di dalam model).
"""
import json
import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:                                  # cadangan jika paket lama yang terpasang
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter


class FruitClassifier:
    def __init__(self, model_path="fruit_model.tflite", class_path="class_names.json", config_path="config.json"):
        self.class_names = json.load(open(class_path, encoding="utf-8"))
        self.img_size = json.load(open(config_path))["img_size"]
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]["index"]
        self.out = self.interp.get_output_details()[0]["index"]

    def _prepare(self, img):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        return img.resize((self.img_size, self.img_size), Image.BILINEAR)

    def _views(self, img, tta):
        base = self._prepare(img)
        views = [base]
        if tta:                                       # 4 sudut pandang, sama seperti TTA multi-view di notebook
            crop = int(round(self.img_size * 0.9)); off = (self.img_size - crop) // 2
            zoom = base.crop((off, off, off + crop, off + crop)).resize((self.img_size, self.img_size), Image.BILINEAR)
            views += [base.transpose(Image.FLIP_LEFT_RIGHT), zoom, zoom.transpose(Image.FLIP_LEFT_RIGHT)]
        return [np.asarray(v, dtype="float32")[None] for v in views]

    def predict_proba(self, img, tta=True):
        probs = []
        for v in self._views(img, tta):
            self.interp.set_tensor(self.inp, v)
            self.interp.invoke()
            probs.append(self.interp.get_tensor(self.out)[0].copy())
        return np.mean(probs, axis=0)

    def predict_topk(self, img, k=5, tta=True):
        p = self.predict_proba(img, tta)
        top = np.argsort(-p)[:k]
        return [(self.class_names[i], float(p[i])) for i in top]
