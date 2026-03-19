# Instruções de Download dos Datasets

Este projeto utiliza três bases de dados públicas que **não podem ser redistribuídas**
por restrições de licenciamento. Siga as instruções abaixo para obtê-las diretamente
das fontes oficiais antes de executar qualquer script.

---

## 1. MESSIDOR + MESSIDOR-2

**Uso no projeto:** treinamento supervisionado e avaliação preditiva dos modelos.  
**Imagens:** 1.200 (MESSIDOR) + 1.748 (MESSIDOR-2) = 2.948 imagens antes da deduplicação.  
**Rótulos:** Retinopathy Grade 0–3 (critérios oficiais por contagem de microaneurismas, hemorragias e neovascularização).

### Como obter

1. Acesse a página oficial: https://www.adcis.net/en/third-party/messidor/
2. Leia e aceite os **termos de uso** (uso permitido apenas para pesquisa e educação; redistribuição proibida).
3. Preencha o formulário de solicitação de acesso. O link de download será enviado por e-mail.
4. Repita o processo para o MESSIDOR-2: https://www.adcis.net/en/third-party/messidor2/

### Organização esperada

Após o download, extraia os arquivos de forma que o script `training_hub.py` possa encontrá-los:

```
<projeto>/
├── data/
│   ├── MESSIDOR/
│   │   ├── Base11/
│   │   ├── Base12/
│   │   └── ...
│   └── MESSIDOR2/
│       └── images/
```

> **Nota sobre duplicatas:** o MESSIDOR original contém imagens duplicadas. O script
> `training_hub.py` executa automaticamente uma etapa de deduplicação por hash MD5
> antes do particionamento, removendo entradas idênticas e pares com SSIM > 0,98.

---

## 2. DIARETDB1

**Uso no projeto:** validação anatômica dos mapas de ativação (Grad-CAM e Score-CAM).
As imagens deste dataset são usadas **exclusivamente** para comparação digital com as
máscaras GT — sem inferência de grau clínico, uma vez que o DIARETDB1 não contém
rotulação de severidade de retinopatia.

**Imagens:** 89 imagens coloridas de fundo de retina com anotações manuais de lesões.  
**Máscaras disponíveis:** hemorragias, exsudatos duros, exsudatos moles, red small dots.

### Como obter

1. Acesse: https://www.it.lut.fi/project/imageret/diaretdb1/
2. Faça o download do arquivo `diaretdb1_v_1_1.zip` (ou equivalente).
3. Extraia o conteúdo.

### Organização esperada

Este projeto já usa os nomes de pasta padrão do DIARETDB1 extraído:

```
<projeto>/
├── ddb1_fundusimages/     ← imagens de retina (.png)
└── ddb1_groundtruth/      ← máscaras binárias por tipo de lesão
    ├── hardexudates/
    ├── hemorrhages/
    ├── redsmalldots/
    └── softexudates/
```

> Se os nomes das pastas do seu download forem diferentes, ajuste os caminhos
> na variável `DIARETDB1_PATH` no topo dos scripts `xai_quantitative_evaluation.py`
> e `xai_generate_heatmaps.py`.


---

## Referências dos datasets

- Decencière et al. (2014). *Feedback on a Publicly Distributed Image Database: The MESSIDOR Database*. Image Analysis & Stereology, 33(3), 231–234. https://doi.org/10.5566/ias.1155
- LATIM (2014). *MESSIDOR-2 Dataset*. Disponível em: https://www.adcis.net/en/third-party/messidor2/
- Kauppi et al. (2007). *DIARETDB1 Diabetic Retinopathy Database and Evaluation Protocol*. MIUA 2007.
