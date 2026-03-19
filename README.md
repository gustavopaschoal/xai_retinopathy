# xai_retinopathy

**Técnicas de Explicabilidade em Modelos de Redes Neurais Convolucionais para Diagnóstico da Retinopatia Diabética por Visão Computacional**

Trabalho de Conclusão de Curso — Engenharia de Controle e Automação  
Instituto Federal do Espírito Santo (IFES) — Campus Serra — 2026  
Autor: Gustavo Paschoal Miranda  
Orientador: Prof. Dr. Gabriel Tozatto Zago

---

## Sobre o projeto

Este repositório contém os scripts utilizados para treinamento, avaliação e análise de explicabilidade de modelos CNN aplicados à detecção e classificação de severidade da retinopatia diabética. São avaliadas 4 arquiteturas (DenseNet121, EfficientNetB3, ResNet50 e VGG16) combinadas com 3 estratégias de pré-processamento (Normalização Simples, CLAHE e Ben Graham Method), totalizando 12 configurações por tarefa.

A avaliação integra métricas convencionais de classificação com análise de explicabilidade via **Grad-CAM** e **Score-CAM**, validadas anatomicamente por sobreposição com máscaras de lesões anotadas do DIARETDB1.

---

## Estrutura do repositório

```
xai_retinopathy/
│
├── training_hub.py                      # Treinamento dos 12 modelos
├── xai_generate_heatmaps.py             # Geração de mapas Grad-CAM e Score-CAM
├── xai_quantitative_evaluation.py       # Métricas espaciais vs máscaras DIARETDB1
├── xai_similarity_analysis.py           # SSIM e Pearson entre técnicas CAM
├── xai_final_ranking.py                 # Ranking multicritério (S_perf + S_estab + S_anat)
├── xai_metrics_visualization.py         # Gráficos de distribuição e tabelas
├── xai_generate_comparison_figures.py   # Figuras comparativas de heatmaps
│
├── ddb1_fundusimages/                   # Imagens do DIARETDB1 (ver nota de licença)
├── ddb1_groundtruth/                    # Máscaras GT do DIARETDB1
│
├── outputs/                             # Modelos treinados (.h5) e resultados
├── xai_outputs/                         # Mapas de ativação gerados
├── visual_outputs/                      # Figuras e visualizações
│
├── requirements.txt
├── README.md
└── download_databases_instructions.md   # Instruções de obtenção dos datasets
```

---

## Requisitos

- Python 3.10
- TensorFlow 2.11 / Keras 2.11
- GPU com suporte CUDA recomendada (testado em NVIDIA Tesla T4, 16 GB VRAM)

### Instalação

```bash
git clone https://github.com/gustavopaschoal/xai_retinopathy
cd xai_retinopathy
pip install -r requirements.txt
```

---

## Dados

Os datasets **não são redistribuídos** neste repositório por restrições de licenciamento. Consulte [`download_databases_instructions.md`](./download_databases_instructions.md) para instruções de obtenção diretamente das fontes oficiais.

| Dataset | Uso | Fonte |
|---|---|---|
| MESSIDOR + MESSIDOR-2 | Treinamento e avaliação preditiva | https://www.adcis.net/en/third-party/messidor/ |
| DIARETDB1 | Validação anatômica dos mapas CAM | https://www.it.lut.fi/project/imageret/diaretdb1/ |

---

## Ordem de execução

```
1. training_hub.py                     → treina e salva modelos em outputs/
2. xai_generate_heatmaps.py            → gera mapas CAM em xai_outputs/
3. xai_quantitative_evaluation.py      → calcula métricas espaciais
4. xai_similarity_analysis.py          → calcula SSIM e Pearson
5. xai_final_ranking.py                → gera ranking multicritério
6. xai_metrics_visualization.py        → gera gráficos e tabelas
7. xai_generate_comparison_figures.py  → gera figuras comparativas
```

---

## Reprodutibilidade

Todos os experimentos utilizam `random_seed = 42`:

```python
import random, numpy as np, tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

**Ambiente testado:**
- Ubuntu 22.04 LTS
- Python 3.10.x
- TensorFlow 2.11.0 / Keras 2.11.0
- NVIDIA Tesla T4 (16 GB VRAM), CUDA 11.8

---

## Licença

O código-fonte está disponível sob licença MIT.  
As imagens dos datasets MESSIDOR, MESSIDOR-2 e DIARETDB1 seguem as licenças originais dos respectivos provedores e **não podem ser redistribuídas**.

---

## Citação

```bibtex
@misc{miranda2026github,
  author       = {Miranda, Gustavo Paschoal},
  title        = {xai\_retinopathy: Scripts para Avaliação Multicritério de CNNs
                  para Diagnóstico de Retinopatia Diabética},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/gustavopaschoal/xai_retinopathy}}
}
```
