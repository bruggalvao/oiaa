"""Testes pequenos: python -m unittest test_redes_neurais -v."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
import redes_neurais as rn


class TestRedes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_extrator_identico_ao_notebook(self):
        nb = json.loads(Path(__file__).with_name("D2_Visao_COLAB.ipynb").read_text(encoding="utf-8"))
        celula = next(c for c in nb["cells"] if c.get("id") == "41fae883")
        arvore = ast.parse("".join(celula["source"]))
        funcao = next(n for n in arvore.body if isinstance(n, ast.FunctionDef) and n.name == "extrair_caracteristicas")
        escopo = {"np": np, "pd": pd}
        exec(compile(ast.Module(body=[funcao], type_ignores=[]), "extrator_original", "exec"), escopo)
        imagens = np.random.default_rng(4).random((8, 48, 48, 3), dtype=np.float32)
        pd.testing.assert_frame_equal(rn.extrair_originais(imagens), escopo["extrair_caracteristicas"](imagens))

    def test_normalizacao_nao_ajusta_validacao(self):
        treino = np.arange(140, dtype=np.float32).reshape(10, 14)
        valida = np.full((3, 14), 10000, dtype=np.float32)
        norm = rn.ajustar_normalizacao(treino, "MLP")
        media = norm["media"].copy()
        rn.transformar(valida, "MLP", norm)
        np.testing.assert_array_equal(norm["media"], media)
        np.testing.assert_allclose(media, treino.mean(axis=0))

    def test_formato_rgb_e_cinco_classes(self):
        fotos = np.ones((2, 48, 48, 3), dtype=np.float32)
        norm = rn.ajustar_normalizacao(fotos, "CNN")
        x = rn.transformar(fotos, "CNN", norm)
        self.assertEqual(tuple(x.shape), (2, 3, 48, 48))
        self.assertTrue(torch.isfinite(x).all())
        self.assertEqual(tuple(rn.criar_rede("CNN")(x).shape), (2, 5))

    def test_treino_das_duas_redes_e_reproducibilidade(self):
        rng = np.random.default_rng(42)
        y = np.tile(np.arange(5), 4)
        for tipo, x in [("MLP", rng.random((20, 14), dtype=np.float32)),
                        ("CNN", rng.random((20, 48, 48, 3), dtype=np.float32))]:
            with self.subTest(tipo=tipo):
                previsoes = []
                for _ in range(2):
                    rede, norm, loader, perda, opt = rn.preparar_treino(tipo, x, y, 42)
                    antes = next(rede.parameters()).detach().clone()
                    loss = rn.treinar_epoca(rede, loader, perda, opt)
                    self.assertTrue(np.isfinite(loss))
                    self.assertFalse(torch.equal(antes, next(rede.parameters())))
                    pred = rn.prever(rede, x, tipo, norm)
                    self.assertEqual(pred.shape, (20,))
                    previsoes.append(pred)
                np.testing.assert_array_equal(*previsoes)

    def test_parada_interna_sem_validacao_externa(self):
        x = np.random.default_rng(42).random((25, 14), dtype=np.float32)
        y = np.tile(np.arange(5), 5)
        epocas, hist = rn.selecionar_epocas("MLP", x, y, np.arange(20), np.arange(20, 25), 42, 2, 2)
        self.assertIn(epocas, [1, 2])
        self.assertEqual(len(hist), 2)


if __name__ == "__main__":
    unittest.main()
