# Análise do Dataset OULAD — Relatório Completo

> Todos os números provêm da execução completa do pipeline sobre os 7 CSV
> originais (32.593 matrículas estudante-curso e ~10,66 milhões de interações
> com o VLE).

---

## 1. Introdução

### Descrição do dataset
O **Open University Learning Analytics Dataset (OULAD)** é um conjunto de dados
público publicado pela The Open University (Reino Unido). Reúne informações
anonimizadas sobre estudantes matriculados em 7 módulos (cursos) ao longo de 4
apresentações (2013B, 2013J, 2014B, 2014J), incluindo dados demográficos,
registros de matrícula, notas de avaliações e interações diárias com o Ambiente
Virtual de Aprendizagem (VLE).

### Objetivos da análise
1. Caracterizar o desempenho acadêmico global e por curso.
2. Quantificar a relação entre a atividade no VLE e os resultados.
3. Identificar perfis demográficos associados ao sucesso ou à evasão.
4. Determinar quando os estudantes desistem para orientar intervenções.
5. Comparar módulos e segmentar estudantes em perfis de aprendizagem.

### Perguntas de pesquisa
1. **Desempenho acadêmico** — Como se distribuem os resultados finais?
2. **Interação com o VLE** — Mais cliques implicam melhores resultados?
3. **Perfil dos estudantes** — Como influem idade, gênero, escolaridade e deficiência?
4. **Evasão do curso** — Em que momento os estudantes desistem?
5. **Comparação entre módulos** — Quais cursos têm maior aprovação ou evasão?

Adicionalmente, análise de avaliações e segmentação de
estudantes com K-Means.

---

## 2. Descrição do Dataset

### Tabela resumo

| Arquivo CSV | Linhas | Colunas | Descrição |
|---|---:|---:|---|
| `courses.csv` | 22 | 3 | Módulos e apresentações; duração em dias. |
| `assessments.csv` | 206 | 6 | Avaliações por curso (tipo TMA/CMA/Exam, data, peso). |
| `vle.csv` | 6.364 | 6 | Materiais do ambiente virtual (tipo de atividade). |
| `studentInfo.csv` | 32.593 | 12 | Demografia e resultado final de cada matrícula. |
| `studentRegistration.csv` | 32.593 | 5 | Datas de registro e de baixa. |
| `studentAssessment.csv` | 173.912 | 5 | Notas obtidas por estudante em cada avaliação. |
| `studentVle.csv` | 10.655.280 | 6 | Cliques diários de cada estudante em cada material. |

### Relações entre tabelas

```
                         courses (code_module, code_presentation)
                            │
        ┌───────────────────┼─────────────────────┐
        │                   │                     │
   assessments          studentInfo            vle
   (id_assessment)   (id_student + curso)   (id_site)
        │                   │                     │
        │                   ├── studentRegistration (1:1 por matrícula)
        │                   │
 studentAssessment          └── studentVle ── (id_site → vle)
 (id_student × id_assessment)   (id_student × id_site × date)
```

- A **chave de junção** principal entre as tabelas de estudante é
  `(id_student, code_module, code_presentation)`.
- `studentAssessment` liga-se a `assessments` por `id_assessment`.
- `studentVle` liga-se a `vle` por `id_site`.

---

## 3. Metodologia

### Ferramentas e bibliotecas
- **Python 3.13**, **pandas**, **numpy** para manipulação de dados.
- **matplotlib** e **seaborn** (paleta *Set2*) para visualização.
- **scipy.stats** para testes de hipótese (Kruskal-Wallis, Qui-quadrado,
  Mann-Whitney, Spearman).
- **scikit-learn** para escalonamento, K-Means e PCA.
- **missingno** para a inspeção de valores nulos.

### Processo de limpeza e pré-processamento
1. **studentInfo**: `final_result` mapeado para a coluna `resultado`
   (`Pass`, `Fail`, `Withdrawn`, `Distinction`); `age_band`, `gender`,
   `disability` e `highest_education` convertidos para `Categorical` ordenado.
2. **studentAssessment**: `score` convertido para numérico; registros com
   `is_banked == 1` ou `score` nulo marcados como `no_submitted`.
3. **studentVle**: agregação de `sum_click` por estudante-curso para obter
   `clicks_totales` e `dias_activos`.
4. **studentRegistration**: datas para numérico e cálculo de `dias_hasta_retiro`.
5. **df_master**: *left join* de studentInfo + studentRegistration + cliques +
   média de score (chave estudante-curso).

### Decisões sobre tratamento de nulos
- **`imd_band`** (~3,4% de nulos): imputado com a **moda**, por ser uma variável
  categórica ordinal com baixa proporção de faltantes. Normalizou-se ainda a
  categoria inconsistente `'10-20'` → `'10-20%'`.
- **`date_unregistration`**: **não é imputado**. Um valor nulo significa que o
  estudante *não desistiu*; imputá-lo introduziria um viés artificial.
- **Estudantes sem atividade no VLE**: `clicks_totales` e `dias_activos`
  preenchidos com 0 (ausência real de interação).

---

## 4. Resultados por pergunta de pesquisa

### Pergunta 1 — Desempenho acadêmico
A distribuição global dos resultados é:

| Resultado | Porcentagem |
|---|---:|
| Pass (Aprovado) | 37,9% |
| Withdrawn (Desistente) | 31,2% |
| Fail (Reprovado) | 21,6% |
| Distinction (Distinção) | 9,3% |

- **Taxa global de aprovação (Pass + Distinction): 47,2%.**
- Quase **1 em cada 3** estudantes (31,2%) desiste do curso.
- **Módulo com maior aprovação: AAA (71,0%)**; **menor: CCC (37,8%)**.

*Figuras:* `p1_A_resultados_barras.png`, `p1_B_resultados_torta.png`,
`p1_C_heatmap_modulo_resultado.png`, `p1_D_apiladas_modulo.png`.

### Pergunta 2 — Interação com o VLE
A atividade no VLE associa-se **fortemente** ao resultado:

| Resultado | Cliques médios | Mediana |
|---|---:|---:|
| Distinction | 2.667 | 1.896 |
| Pass | 1.922 | 1.343 |
| Fail | 652 | 317 |
| Withdrawn | 314 | 89 |

- **Kruskal-Wallis: H = 14.857, p < 0,001** → as diferenças entre os quatro
  grupos são altamente significativas.
- Correlação **Spearman cliques ↔ score = 0,36** (positiva e significativa).
- Estudantes com Distinction acumulam **~8,5 vezes** mais cliques do que os que
  desistem.

*Figuras:* `p2_A_boxplot_clicks.png`, `p2_B_violinplot_clicks.png`,
`p2_C_scatter_clicks_score.png`, `p2_D_barras_media_clicks.png`.

### Pergunta 3 — Perfil dos estudantes
Todas as variáveis demográficas mostram **associação estatisticamente
significativa** com o resultado (Qui², p < 0,05):

| Variável | p-valor Qui² | Interpretação |
|---|---:|---|
| `highest_education` | 9,2 × 10⁻²¹² | Associação mais forte. |
| `age_band` | 2,8 × 10⁻⁴⁵ | Significativa. |
| `disability` | 8,1 × 10⁻³⁰ | Significativa. |
| `gender` | 8,8 × 10⁻⁴ | Significativa (efeito pequeno). |

- **Deficiência e evasão**: estudantes com deficiência desistem mais
  (**39,3%** vs **30,3%**); **Odds Ratio = 1,49**.
- A **escolaridade prévia** é o preditor demográfico mais potente do sucesso
  acadêmico.

*Figuras:* `p3_A_edad.png`, `p3_B_genero.png`, `p3_C_educacion.png`,
`p3_D_discapacidad.png`, `p3_E_heatmap_correlacion.png`.

### Pergunta 4 — Evasão do curso
Entre os estudantes desistentes:

- **Mediana do dia de desistência: 27** (≈ 4 semanas após o início).
- **50,9%** desiste nos primeiros **30 dias**, **69,8%** em **90 dias** e
  **92,5%** em **180 dias**.
- **Módulo com maior evasão precoce (≤30 dias): CCC (21,0%)**.
- Parte das baixas ocorre inclusive **antes do dia 0** (cancelam antes de
  começar), sinal de evasão muito precoce.

**Janelas de evasão:** a maior concentração de baixas ocorre na janela
**semana 1–4**, o que sugere que as intervenções de retenção devem focar no
primeiro mês do curso.

*Figuras:* `p4_A_histograma_retiro.png`, `p4_B_kde_retiro.png`,
`p4_C_boxplot_dias_modulo.png`, `p4_D_curva_abandono.png`.

### Pergunta 5 — Comparação entre módulos

| Módulo | Aprovação | Evasão | Distinção | Reprovação | Score médio | Cliques médios | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| AAA | **71,0%** | 16,8% | 5,9% | 12,2% | 68,3 | 1.667 | 748 |
| GGG | 59,8% | **11,5%** | 15,6% | 28,7% | 77,5 | 526 | 2.534 |
| EEE | 56,2% | 24,6% | 12,1% | 19,2% | 80,0 | 1.358 | 2.934 |
| BBB | 47,5% | 30,2% | 8,6% | 22,3% | 73,3 | 662 | 7.909 |
| FFF | 47,0% | 31,0% | 8,6% | 22,0% | 75,6 | 2.267 | 7.762 |
| DDD | 41,6% | 35,9% | 6,1% | 22,5% | 67,4 | 882 | 6.272 |
| CCC | 37,8% | **44,5%** | 11,2% | 17,6% | 67,1 | 1.056 | 4.434 |

- **Melhor módulo: AAA** (maior aprovação, embora com poucos estudantes).
- **Módulo mais problemático: CCC** (maior evasão, 44,5%).
- **GGG** destaca-se pela menor evasão (11,5%) e alta taxa de distinção.
- No nível de módulo existe relação positiva entre cliques médios e aprovação,
  ainda que moderada (outros fatores, como a dificuldade do curso, também
  intervêm).

*Figuras:* `p5_A_aprobacion_modulo.png`, `p5_B_apiladas_modulo.png`,
`p5_C_lollipop_retiro.png`, `p5_D_heatmap_metricas.png`,
`p5_E_scatter_modulos.png`.

### Análise das avaliações
- O score médio é semelhante entre as avaliações contínuas, com os exames
  finais apresentando a maior dispersão.
- **Achado não intuitivo:** entregar **atrasado** **não** se associa a notas
  piores de forma significativa: score médio no prazo **76,4** vs atrasado
  **75,0** (Mann-Whitney **p = 0,72**, *não significativo*). A diferença é
  marginal e não conclusiva com estes dados.

*Figuras:* `p6_A_scores_por_tipo.png`, `p6_B_evolucion_score.png`,
`p6_C_entrega_tardia.png`.

### Segmentação com K-Means
Features: cliques totais, score médio, dias ativos, escolaridade e IMD. Com
**K = 4** clusters (método do cotovelo) e PCA 2D que explica **62,9%** da
variância, emergem perfis claros:

| Cluster | N | Cliques médios | Score médio | Dias ativos | Perfil |
|---:|---:|---:|---:|---:|---|
| 2 | 4.293 | 4.535 | 82,0 | 159 | **Alto desempenho / muito ativos** (27,7% distinção). |
| 0 | 7.781 | 1.086 | 77,4 | 60 | Bom desempenho, atividade média-alta. |
| 3 | 9.176 | 989 | 77,2 | 54 | Desempenho médio, contexto socioeconômico baixo (IMD baixo). |
| 1 | 4.307 | 521 | 45,5 | 33 | **Baixo desempenho / em risco** (37% evasão, 45% reprovação). |

*Figuras:* `p7_A_metodo_codo.png`, `p7_B_pca_clusters.png`.

### Análise temporal do clique por `activity_type` (VLE)
Esta seção explora a dimensão mais rica do dataset: o clique diário desagregado
por **tipo de atividade** (`activity_type`), cruzando `studentVle` com `vle`.
Os tipos dominantes são **oucontent, forumng, quiz e homepage**.

**Composição por resultado:** quem obtém **Distinction** concentra mais cliques
em **conteúdo** (`oucontent` 28%) e **fórum** (`forumng` 24%), enquanto os
**desistentes** usam proporcionalmente menos conteúdo (`oucontent` 25%) e mais
**navegação** (`homepage`/`subpage`) — engajam-se com a estrutura, mas não com o
material de aprendizagem.

**Visualização interativa:** `p8_interactive_activity_weekly.html` (Plotly) — área
empilhada de cliques médios por estudante, por semana e tipo de atividade, com
menu para alternar entre os grupos de resultado.

*Figuras:* `p8_A_weekly_heatmap.png`, `p8_B_composition_by_result.png`,
`p8_interactive_activity_weekly.html`.

---

## 5. Conclusões

### Top 5 achados mais importantes
1. **A evasão é o principal problema**: 31,2% de evasão global e apenas 47,2% de
   aprovação. A evasão concentra-se nas primeiras 4 semanas (mediana dia 27;
   50,9% antes do dia 30).
2. **A atividade no VLE é o preditor mais claro de sucesso**: estudantes com
   Distinction registram ~8,5× mais cliques do que os que desistem
   (Kruskal-Wallis p < 0,001).
3. **A escolaridade prévia é o fator demográfico mais determinante**
   (Qui² p ≈ 10⁻²¹²), acima de idade, gênero ou deficiência.
4. **A deficiência eleva o risco de evasão** (OR = 1,49; 39,3% vs 30,3% de
   evasão), indicando a necessidade de apoio específico.
5. **Há grande heterogeneidade entre módulos**: AAA aprova 71% enquanto CCC tem
   evasão de 44,5%; o clustering confirma um segmento "em risco" (~4.300
   estudantes) com baixa atividade e alta evasão (37%).

### Limitações da análise
- Os dados são **observacionais**: as associações (p.ex. cliques ↔ sucesso)
  **não implicam causalidade**.
- O score médio é calculado apenas sobre entregas válidas; os estudantes
  desistentes têm poucas ou nenhuma avaliação, o que pode enviesar comparações.
- O módulo **AAA** tem amostra pequena (748), logo suas taxas são menos robustas
  do que as de módulos massivos como BBB ou FFF.
- `imd_band` imputado com a moda pode atenuar levemente seu efeito real.


---

## 6. Apêndice

### Lista de figuras geradas (`outputs/figures/`)
| Figura | Conteúdo |
|---|---|
| `missing_values.png` | Matriz de nulos (missingno) de studentInfo e studentRegistration. |
| `p1_A_resultados_barras.png` | Barras horizontais: contagem e % por resultado. |
| `p1_B_resultados_torta.png` | Pizza dos 4 resultados. |
| `p1_C_heatmap_modulo_resultado.png` | Heatmap módulo × resultado (% por linha). |
| `p1_D_apiladas_modulo.png` | Barras empilhadas 100% por módulo. |
| `p2_A_boxplot_clicks.png` | Boxplot de cliques por resultado (escala log). |
| `p2_B_violinplot_clicks.png` | Violinplot de cliques (log1p) por resultado. |
| `p2_C_scatter_clicks_score.png` | Dispersão cliques vs score + regressão. |
| `p2_D_barras_media_clicks.png` | Média de cliques ± desvio padrão. |
| `p3_A_edad.png` | Resultado por faixa etária (%). |
| `p3_B_genero.png` | Resultado por gênero (%). |
| `p3_C_educacion.png` | Resultado por escolaridade (%). |
| `p3_D_discapacidad.png` | Resultado por deficiência (empilhadas 100%). |
| `p3_E_heatmap_correlacion.png` | Correlação variáveis demográficas ↔ resultado. |
| `p4_A_histograma_retiro.png` | Histograma do dia de desistência (linhas 30/90/180). |
| `p4_B_kde_retiro.png` | Densidade (KDE) do momento de desistência. |
| `p4_C_boxplot_dias_modulo.png` | Dias até a desistência por módulo. |
| `p4_D_curva_abandono.png` | Curva acumulada de evasão. |
| `p5_A_aprobacion_modulo.png` | Taxa de aprovação por módulo. |
| `p5_B_apiladas_modulo.png` | Composição de resultados por módulo. |
| `p5_C_lollipop_retiro.png` | Lollipop da taxa de evasão por módulo. |
| `p5_D_heatmap_metricas.png` | Heatmap módulo × métrica. |
| `p5_E_scatter_modulos.png` | Aprovação vs cliques médios por módulo. |
| `p6_A_scores_por_tipo.png` | Scores por tipo de avaliação. |
| `p6_B_evolucion_score.png` | Evolução do score por semana de entrega. |
| `p6_C_entrega_tardia.png` | Score conforme pontualidade da entrega. |
| `p7_A_metodo_codo.png` | Método do cotovelo para K-Means. |
| `p7_B_pca_clusters.png` | Clusters e resultado no espaço PCA 2D. |

### Descrição das variáveis-chave
| Variável | Descrição |
|---|---|
| `final_result` / `resultado` | Resultado final: Pass / Fail / Withdrawn / Distinction. |
| `clicks_totales` | Soma de cliques do estudante no VLE (toda a apresentação). |
| `dias_activos` | Número de dias distintos com atividade no VLE. |
| `score_promedio` | Média das notas das avaliações válidas do estudante. |
| `date_registration` | Dia (relativo ao início do curso) em que se registrou. |
| `date_unregistration` | Dia em que deu baixa (nulo = não desistiu). |
| `dias_hasta_retiro` | `date_unregistration - date_registration`. |
| `imd_band` | Faixa do índice de privação (menor = mais desfavorecido). |
| `highest_education` | Maior nível de escolaridade prévia (ordinal). |
| `age_band` | Faixa etária (0-35, 35-55, 55<=). |
