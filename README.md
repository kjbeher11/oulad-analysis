# Análise do Dataset OULAD

Análise acadêmica completa, estruturada e reprodutível do **Open University
Learning Analytics Dataset (OULAD)**: desempenho, evasão, perfis demográficos,
comparação de cursos e segmentação de estudantes.


## Descrição do Projeto

O **OULAD** é um dataset público da The Open University (Reino Unido) que combina
informações demográficas, registros de matrícula, notas e interações diárias com
o Ambiente Virtual de Aprendizagem (VLE) de mais de 32.000 matrículas
estudante-curso e ~10,66 milhões de cliques. O objetivo deste projeto é
**caracterizar o sucesso e a evasão estudantil** e **quantificar quais fatores os
predizem**, gerando um relatório e um conjunto de visualizações prontos para
publicação.

## Perguntas de Pesquisa

1. **Desempenho acadêmico** — Como se distribuem os resultados finais?
2. **Interação com o VLE** — Mais cliques implicam melhores resultados?
3. **Perfil dos estudantes** — Como influem idade, gênero, escolaridade e deficiência?
4. **Evasão do curso** — Em que momento os estudantes desistem?
5. **Comparação entre módulos** — Quais cursos têm maior aprovação ou evasão?

Análise de avaliações e segmentação de estudantes com K-Means.

## Estrutura do Repositório

```
oulad-analysis/
├── data/
│   └── raw/                  # Os 7 CSV originais do dataset OULAD
├── notebooks/
│   └── analysis.ipynb        # Notebook principal (executável do início ao fim)
├── src/
│   ├── __init__.py
│   ├── load_data.py          # Carregamento, verificação e inspeção dos CSV
│   ├── preprocess.py         # Limpeza e construção do df_master
│   └── visualizations.py     # Utilitários de plotagem reutilizáveis
├── outputs/
│   ├── figures/              # 28 visualizações em PNG (150 dpi), rótulos em inglês
│   └── report/
│       ├── informe_oulad.md  # Relatório final completo (português)
│       └── findings.json     # Achados numéricos exportados pelo notebook
├── requirements.txt          # Dependências com versões fixas
└── README.md
```

## Dataset

| Arquivo | Linhas | Colunas principais |
|---|---:|---|
| `courses.csv` | 22 | code_module, code_presentation, module_presentation_length |
| `assessments.csv` | 206 | id_assessment, assessment_type (TMA/CMA/Exam), date, weight |
| `vle.csv` | 6.364 | id_site, activity_type, week_from, week_to |
| `studentInfo.csv` | 32.593 | gender, region, highest_education, imd_band, age_band, disability, final_result |
| `studentRegistration.csv` | 32.593 | date_registration, date_unregistration |
| `studentAssessment.csv` | 173.912 | id_assessment, id_student, date_submitted, is_banked, score |
| `studentVle.csv` | 10.655.280 | id_student, id_site, date, sum_click |

Chave de junção do estudante: `(id_student, code_module, code_presentation)`.

## Instalação e Uso

```bash
git clone <url-do-repositorio>
cd oulad-analysis

# (Opcional) ambiente virtual
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Coloque os 7 CSV do OULAD em data/raw/ e abra o notebook:
jupyter notebook notebooks/analysis.ipynb
```

> O notebook verifica automaticamente que os 7 CSV existam em `data/raw/` e
> lança um erro claro se faltar algum. Pode ser executado do início ao fim com
> **Run All Cells**. Também pode ser executado em lote com
> `jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb`.

## Metodologia

1. **Carregamento** otimizado dos 7 CSV (dtypes reduzidos para os 10,6M de
   registros de `studentVle`).
2. **EDA**: shapes, dtypes, nulos (%), duplicados, `value_counts` e matriz de
   nulos com *missingno*.
3. **Limpeza**: mapeamento de `final_result`, conversão para `Categorical`,
   imputação de `imd_band` com a moda, agregação de cliques e construção do
   **`df_master`** (left joins).
4. **Análise** por pergunta, com testes estatísticos (Kruskal-Wallis, Qui²,
   Spearman, Mann-Whitney) e visualizações consistentes (paleta *Set2*).
5. **Segmentação** com StandardScaler + K-Means + PCA.

Todo o código usa `pathlib` (caminhos relativos) e `random_state=42`.

## Principais Achados

1. **Evasão precoce elevada**: 31,2% de evasão global; a **mediana do dia de
   desistência é 27** e **50,9%** desiste nos primeiros 30 dias.
2. **A atividade no VLE prediz o sucesso**: estudantes com Distinction acumulam
   **~2.667 cliques** em média contra **314** dos que desistem
   (Kruskal-Wallis **p < 0,001**; Spearman cliques↔score = **0,36**).
3. **A escolaridade prévia é o fator demográfico mais determinante**
   (Qui² **p ≈ 10⁻²¹²**); todas as variáveis demográficas são significativas.
4. **A deficiência aumenta o risco de evasão**: **39,3%** de evasão vs **30,3%**
   sem deficiência (**Odds Ratio = 1,49**).
5. **Forte variabilidade entre módulos**: **AAA** aprova **71%** enquanto **CCC**
   tem evasão de **44,5%**. O K-Means (K=4) isola um segmento *em risco* de
   ~4.300 estudantes com baixa atividade e 37% de evasão.

## Visualizações

As 28 figuras são salvas em `outputs/figures/` (rótulos em inglês). Algumas em
destaque:

| Figura | Descrição |
|---|---|
| `p1_B_resultados_torta.png` | Proporção global dos 4 resultados finais. |
| `p2_A_boxplot_clicks.png` | Cliques totais por resultado (escala log). |
| `p3_E_heatmap_correlacion.png` | Correlação demografia ↔ resultado. |
| `p4_D_curva_abandono.png` | Curva acumulada de evasão por dia do curso. |
| `p5_D_heatmap_metricas.png` | Métricas (aprovação/evasão/...) por módulo. |
| `p7_B_pca_clusters.png` | Segmentação K-Means no espaço PCA 2D. |

Completo: [`outputs/report/informe_oulad.md`](outputs/report/informe_oulad.md)

## Tecnologias Utilizadas

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scipy` · `scikit-learn` ·
`missingno` · `plotly` · `jupyter`


