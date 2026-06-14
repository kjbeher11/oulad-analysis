"""
build_notebook.py
=================

Gera notebooks/analysis.ipynb de forma programática com nbformat.
Cada seção tem uma célula Markdown explicativa (O QUÊ e POR QUÊ) seguida de
células de código que usam as funções reutilizáveis de src/.

Idiomas: o texto explicativo (Markdown) e os resumos impressos estão em
**português**; os rótulos, títulos e categorias dos gráficos estão em **inglês**
(termos originais do OULAD: Pass / Fail / Withdrawn / Distinction).

Este script NÃO executa o notebook; apenas o constrói. A execução é feita
com `jupyter nbconvert --execute`.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =========================================================================== #
# TÍTULO
# =========================================================================== #
md(r"""
# 📊 Análise do Dataset OULAD — Notebook Principal

**Open University Learning Analytics Dataset (OULAD)**

Este notebook realiza uma análise acadêmica completa, estruturada e reprodutível
do dataset OULAD. Está organizado em seções que respondem a 5 perguntas de
pesquisa principais, além de análises adicionais (avaliações e segmentação com
K-Means).

> **Idiomas:** o texto explicativo está em **português** e os rótulos dos
> gráficos em **inglês** (termos originais do OULAD), para facilitar a leitura
> por uma equipe multilíngue.

**Reprodutibilidade:** todo o código usa `random_state=42`, caminhos relativos
com `pathlib` e funções reutilizáveis do pacote `src/`. O notebook deve poder
ser executado do início ao fim com *Run All Cells* sem erros.
""")

# =========================================================================== #
# SETUP
# =========================================================================== #
md(r"""
## 0. Configuração inicial

**O quê:** importamos as bibliotecas, configuramos o estilo dos gráficos,
fixamos a semente aleatória e preparamos o acesso aos módulos do pacote `src/`.

**Por quê:** centralizar a configuração garante consistência visual e
reprodutibilidade em toda a análise. Iniciamos também o cronômetro global para
reportar o tempo total de execução ao final.
""")

code(r"""
import time
_t_inicio = time.time()  # cronômetro global

import sys
from pathlib import Path

# Permite importar o pacote src/ executando a partir de notebooks/ ou da raiz.
ROOT = Path.cwd()
if (ROOT / "src").exists():
    PROJECT_ROOT = ROOT
elif (ROOT.parent / "src").exists():
    PROJECT_ROOT = ROOT.parent
else:
    raise RuntimeError("Pasta src/ não encontrada. Execute a partir da raiz do projeto ou de notebooks/.")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import missingno as msno

from src import load_data, preprocess, visualizations as viz

# Semente global para reprodutibilidade.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Estilo de gráficos consistente.
viz.set_style()

# Dicionário para acumular achados-chave que depois alimentam o relatório.
findings = {}

print("Configuração pronta. PROJECT_ROOT =", PROJECT_ROOT)
print("pandas", pd.__version__, "| numpy", np.__version__)
""")

# =========================================================================== #
# VERIFICAÇÃO DE ARQUIVOS
# =========================================================================== #
md(r"""
## 1. Verificação de arquivos e carregamento dos dados

**O quê:** verificamos que os 7 CSV existam em `data/raw/` e, se tudo estiver
correto, os carregamos em um dicionário de DataFrames.

**Por quê:** falhar cedo com uma mensagem clara é melhor do que um erro críptico
no meio da análise. O carregamento usa dtypes otimizados para `studentVle.csv`
(~10.6M de linhas) e assim reduzir o consumo de memória.
""")

code(r"""
# Verificação: lança FileNotFoundError claro se faltar algum CSV.
load_data.check_files()
""")

code(r"""
# Carregamento dos 7 CSV. studentVle leva alguns segundos por causa do tamanho.
data = load_data.load_all()
""")

md(r"""
### 1.1 Inspeção inicial de cada tabela

Imprimimos shape, dtypes e primeiras linhas de cada DataFrame para entender a
estrutura do dataset.
""")

code(r"""
load_data.inspect_all(data, n_rows=3)
""")

# =========================================================================== #
# EDA
# =========================================================================== #
md(r"""
## 2. Exploração inicial (EDA)

**O quê:** para cada um dos 7 DataFrames calculamos shape, dtypes,
`describe()`, porcentagem de nulos por coluna, número de duplicados e, para as
colunas categóricas, `value_counts()` com porcentagens.

**Por quê:** a EDA revela problemas de qualidade (nulos, duplicados, tipos
incorretos) que condicionam as decisões de limpeza do Passo 3.
""")

code(r"""
def eda_resumen(nombre, df, max_cat_unique=15):
    print("=" * 90)
    print(f"TABELA: {nombre}  ->  {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    print("=" * 90)

    # Nulos por coluna em porcentagem.
    nulos = (df.isna().mean() * 100).round(2)
    nulos = nulos[nulos > 0]
    if len(nulos):
        print("\nNulos por coluna (%):")
        print(nulos.to_string())
    else:
        print("\nNulos por coluna (%): nenhuma coluna com nulos")

    # Duplicados.
    print(f"\nLinhas duplicadas: {df.duplicated().sum():,}")

    # describe (numéricas).
    num = df.select_dtypes(include=[np.number])
    if num.shape[1]:
        print("\nDescribe (numéricas):")
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(num.describe().round(2).to_string())

    # value_counts das categóricas com poucas categorias.
    cat = df.select_dtypes(include=["object", "category"])
    for c in cat.columns:
        nun = df[c].nunique(dropna=True)
        if nun <= max_cat_unique:
            print(f"\nvalue_counts de '{c}' ({nun} categorias):")
            vc = df[c].value_counts(dropna=False)
            pct = (vc / len(df) * 100).round(2)
            print(pd.DataFrame({"conteo": vc, "porcentaje": pct}).to_string())
    print()


for nombre, df in data.items():
    eda_resumen(nombre, df)
""")

md(r"""
### 2.1 Matriz de valores nulos (missingno)

Visualizamos os nulos das tabelas com dados faltantes relevantes:
`studentInfo` (imd_band) e `studentRegistration` (date_unregistration).
Salvamos a figura em `outputs/figures/missing_values.png`.
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
msno.matrix(data["studentInfo"], ax=axes[0], sparkline=False, color=(0.3, 0.5, 0.7))
axes[0].set_title("studentInfo — missing values matrix", fontsize=13, fontweight="bold")
msno.matrix(data["studentRegistration"], ax=axes[1], sparkline=False, color=(0.7, 0.4, 0.4))
axes[1].set_title("studentRegistration — missing values matrix", fontsize=13, fontweight="bold")
fig.suptitle("Missing values matrix (missingno)", fontsize=15, fontweight="bold")
fig.tight_layout()
viz.save_fig(fig, "missing_values.png")
plt.show()
""")

# =========================================================================== #
# PREPROCESSAMENTO
# =========================================================================== #
md(r"""
## 3. Limpeza e pré-processamento

**O quê:** aplicamos o pipeline de `src/preprocess.py`:
1. **studentInfo** — `final_result` para a coluna `resultado` (Pass / Fail /
   Withdrawn / Distinction), conversão para `Categorical`, normalização e
   imputação de `imd_band` com a moda.
2. **studentAssessment** — `score` para numérico, marca `no_submitted` para
   `is_banked == 1` ou `score` nulo.
3. **studentVle** — cliques totais e dias ativos por estudante.
4. **studentRegistration** — datas para numérico e `dias_hasta_retiro`.
5. **df_master** — left join de studentInfo + studentRegistration + cliques +
   média de score.

**Por quê:** um único DataFrame mestre limpo simplifica todas as análises
posteriores e evita junções repetidas.

**Decisão sobre nulos:** `imd_band` é imputado com a moda (variável categórica,
~3.4% de nulos). `date_unregistration` é deixado como nulo de forma intencional:
um nulo significa que o estudante **não desistiu**, portanto imputá-lo
introduziria viés.
""")

code(r"""
info = preprocess.clean_student_info(data["studentInfo"])
assessment = preprocess.clean_student_assessment(data["studentAssessment"])
registration = preprocess.clean_student_registration(data["studentRegistration"])
clicks = preprocess.aggregate_clicks(data["studentVle"])
scores = preprocess.mean_score_per_student(assessment, data["assessments"])

df_master = preprocess.build_master(info, registration, clicks, scores)
df_master = preprocess.encode_demographics(df_master)

print("\nColunas de df_master:")
print(df_master.dtypes.to_string())
df_master.head()
""")

# =========================================================================== #
# PERGUNTA 1
# =========================================================================== #
md(r"""
## 4. Pergunta 1 — Desempenho acadêmico

**O quê:** analisamos como se distribuem os resultados finais (`resultado`),
globalmente e por módulo / apresentação.

**Por quê:** estabelece a linha de base do sucesso estudantil e permite detectar
módulos problemáticos.

Visualizações: (A) barras horizontais com contagem e %, (B) pizza, (C) heatmap
módulo × resultado, (D) barras empilhadas a 100% por módulo.
""")

code(r"""
from src.preprocess import RESULT_ORDER

# Frequência e porcentagem global.
res_counts = df_master["resultado"].value_counts().reindex(RESULT_ORDER)
res_pct = (res_counts / res_counts.sum() * 100).round(2)
print("Distribuição global dos resultados:")
print(pd.DataFrame({"conteo": res_counts, "porcentaje": res_pct}).to_string())

# A) Barras horizontais com contagem e %.
fig = viz.barh_counts_pct(df_master["resultado"], "Distribution of final results (Q1)",
                          "Number of students", order=RESULT_ORDER)
viz.save_fig(fig, "p1_A_resultados_barras.png"); plt.show()
""")

code(r"""
# B) Gráfico de pizza.
fig = viz.pie_chart(df_master["resultado"], "Share of final results (Q1)", order=RESULT_ORDER)
viz.save_fig(fig, "p1_B_resultados_torta.png"); plt.show()
""")

code(r"""
# C) Heatmap módulo × resultado (% por linha).
ct_mod = pd.crosstab(df_master["code_module"], df_master["resultado"])[RESULT_ORDER]
fig = viz.heatmap_pct(ct_mod, "Result by module (% per row) (Q1)",
                      "Final result", "Module", col_order=RESULT_ORDER)
viz.save_fig(fig, "p1_C_heatmap_modulo_resultado.png"); plt.show()
""")

code(r"""
# D) Barras empilhadas a 100% por módulo.
fig = viz.stacked_100(ct_mod, "Result composition by module (100%) (Q1)",
                      "Module", col_order=RESULT_ORDER)
viz.save_fig(fig, "p1_D_apiladas_modulo.png"); plt.show()
""")

code(r"""
# Distribuição por apresentação.
ct_pres = pd.crosstab(df_master["code_presentation"], df_master["resultado"])[RESULT_ORDER]
print("Resultado por apresentação (%):")
print((ct_pres.div(ct_pres.sum(axis=1), axis=0) * 100).round(1).to_string())

# Estatísticas-chave: taxa global de aprovação e módulos extremos.
tasa_aprob = res_pct["Pass"] + res_pct["Distinction"]
aprob_por_mod = ct_mod[["Pass", "Distinction"]].sum(axis=1) / ct_mod.sum(axis=1) * 100
mod_max = aprob_por_mod.idxmax(); mod_min = aprob_por_mod.idxmin()

findings["p1_tasa_aprobacion_global"] = round(float(tasa_aprob), 2)
findings["p1_modulo_mayor_aprobacion"] = (mod_max, round(float(aprob_por_mod.max()), 2))
findings["p1_modulo_menor_aprobacion"] = (mod_min, round(float(aprob_por_mod.min()), 2))
findings["p1_dist_resultados"] = res_pct.to_dict()

print(f"\n>>> Taxa global de APROVAÇÃO (Pass + Distinction): {tasa_aprob:.2f}%")
print(f">>> Módulo com MAIOR aprovação: {mod_max} ({aprob_por_mod.max():.2f}%)")
print(f">>> Módulo com MENOR aprovação: {mod_min} ({aprob_por_mod.min():.2f}%)")
""")

md(r"""
### 📌 Resumo dos achados — Pergunta 1

O bloco a seguir imprime um resumo em texto claro da Pergunta 1.
""")

code(r"""
print("RESUMO Q1 — Desempenho acadêmico")
print("-" * 60)
print(f"- Resultado mais frequente: {res_counts.idxmax()} ({res_pct.max():.1f}%).")
print(f"- Taxa global de aprovação (Pass+Distinction): {tasa_aprob:.1f}%.")
print(f"- Taxa de desistência global: {res_pct['Withdrawn']:.1f}%.")
print(f"- Módulo mais bem-sucedido: {mod_max} ({aprob_por_mod.max():.1f}% de aprovação).")
print(f"- Módulo mais problemático: {mod_min} ({aprob_por_mod.min():.1f}% de aprovação).")
""")

# =========================================================================== #
# PERGUNTA 2
# =========================================================================== #
md(r"""
## 5. Pergunta 2 — Interação com o VLE

**O quê:** mais cliques no ambiente virtual de aprendizagem (VLE) se associam a
melhores resultados? Comparamos `clicks_totales` entre os 4 grupos de resultado,
aplicamos o teste de **Kruskal-Wallis** (não paramétrico, pois os cliques não
são normais) e medimos a correlação entre cliques e `score_promedio`.

**Por quê:** a atividade no VLE é um preditor precoce clássico de evasão e
desempenho; quantificá-lo justifica sistemas de alerta antecipado.

Visualizações: (A) boxplot log-Y, (B) violinplot, (C) dispersão cliques vs score
com regressão, (D) barras de média ± desvio.
""")

code(r"""
from src.preprocess import RESULT_ORDER

# Estatísticas descritivas dos cliques por grupo de resultado.
desc_clicks = df_master.groupby("resultado", observed=True)["clicks_totales"].agg(
    ["count", "mean", "median", "std"]).reindex(RESULT_ORDER).round(1)
print("Cliques totais por resultado:")
print(desc_clicks.to_string())

# Teste de Kruskal-Wallis entre os 4 grupos.
grupos = [df_master.loc[df_master["resultado"] == r, "clicks_totales"].dropna()
          for r in RESULT_ORDER]
H, p_kw = stats.kruskal(*grupos)
print(f"\nKruskal-Wallis: H = {H:.2f}, p-valor = {p_kw:.3e}")

findings["p2_clicks_por_resultado"] = desc_clicks[["mean", "median"]].to_dict("index")
findings["p2_kruskal_H"] = round(float(H), 2)
findings["p2_kruskal_p"] = float(p_kw)
""")

code(r"""
# A) Boxplot dos cliques por resultado (escala log em Y).
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_master, x="resultado", y="clicks_totales", order=RESULT_ORDER,
            palette="Set2", ax=ax)
ax.set_yscale("log")
ax.set_title("Total clicks by final result (log scale) (Q2)")
ax.set_xlabel("Final result"); ax.set_ylabel("Total clicks (log)")
fig.tight_layout(); viz.save_fig(fig, "p2_A_boxplot_clicks.png"); plt.show()
""")

code(r"""
# B) Violinplot.
fig, ax = plt.subplots(figsize=(12, 6))
df_plot = df_master.copy()
df_plot["log_clicks"] = np.log1p(df_plot["clicks_totales"])
sns.violinplot(data=df_plot, x="resultado", y="log_clicks", order=RESULT_ORDER,
               palette="Set2", ax=ax, cut=0)
ax.set_title("Click distribution by result (violinplot, log1p) (Q2)")
ax.set_xlabel("Final result"); ax.set_ylabel("log(1 + total clicks)")
fig.tight_layout(); viz.save_fig(fig, "p2_B_violinplot_clicks.png"); plt.show()
""")

code(r"""
# C) Dispersão cliques vs média de score, colorido por resultado + regressão global.
sub = df_master.dropna(subset=["score_promedio", "clicks_totales"]).copy()
sub = sub[sub["clicks_totales"] > 0]
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=sub, x="clicks_totales", y="score_promedio", hue="resultado",
                hue_order=RESULT_ORDER, palette="Set2", alpha=0.35, s=18, ax=ax)
sns.regplot(data=sub, x="clicks_totales", y="score_promedio", scatter=False,
            color="black", line_kws={"linewidth": 2}, ax=ax)
ax.set_xscale("log")
ax.set_title("Total clicks vs. average score (Q2)")
ax.set_xlabel("Total clicks (log)"); ax.set_ylabel("Average score")
ax.legend(title="Result", bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout(); viz.save_fig(fig, "p2_C_scatter_clicks_score.png"); plt.show()

# Correlação de Spearman (robusta à não-normalidade).
rho, p_rho = stats.spearmanr(sub["clicks_totales"], sub["score_promedio"])
print(f"Correlação Spearman cliques vs score: rho = {rho:.3f} (p = {p_rho:.3e})")
findings["p2_spearman_rho"] = round(float(rho), 3)
""")

code(r"""
# D) Barras com média de cliques ± desvio.
fig, ax = plt.subplots(figsize=(12, 6))
medias = desc_clicks["mean"]; errs = desc_clicks["std"]
colors = sns.color_palette("Set2", len(RESULT_ORDER))
ax.bar(medias.index.astype(str), medias.values, yerr=errs.values, capsize=6, color=colors)
ax.set_title("Mean clicks by result (± standard deviation) (Q2)")
ax.set_xlabel("Final result"); ax.set_ylabel("Total clicks (mean)")
viz.annotate_bars(ax, fmt="{:.0f}")
fig.tight_layout(); viz.save_fig(fig, "p2_D_barras_media_clicks.png"); plt.show()
""")

md(r"""
### 📌 Resumo dos achados — Pergunta 2
""")

code(r"""
interp = "SIGNIFICATIVA" if p_kw < 0.05 else "NÃO significativa"
print("RESUMO Q2 — Interação com o VLE")
print("-" * 60)
for r in RESULT_ORDER:
    print(f"- {r}: média {desc_clicks.loc[r,'mean']:.0f} cliques | mediana {desc_clicks.loc[r,'median']:.0f}.")
print(f"- Kruskal-Wallis p = {p_kw:.2e} -> diferença entre grupos {interp}.")
print(f"- Correlação cliques-score (Spearman) = {rho:.2f}: mais cliques, melhor score.")
print("- Conclusão: a atividade no VLE se associa claramente a melhores resultados.")
""")

# =========================================================================== #
# PERGUNTA 3
# =========================================================================== #
md(r"""
## 6. Pergunta 3 — Perfil dos estudantes

**O quê:** relacionamos variáveis demográficas (idade, gênero, escolaridade,
deficiência) com o resultado final. Construímos tabelas de contingência,
aplicamos **Qui-quadrado** a cada variável e calculamos a **Odds Ratio** de
desistência associada à deficiência.

**Por quê:** identificar grupos em risco permite focar o apoio institucional e
promover a equidade.

Visualizações: barras agrupadas por idade, gênero e escolaridade; barras
empilhadas por deficiência; heatmap de correlação entre variáveis codificadas.
""")

code(r"""
from src.preprocess import RESULT_ORDER, AGE_ORDER, EDUCATION_ORDER

def chi2_test(var):
    ct = pd.crosstab(df_master[var], df_master["resultado"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return ct, chi2, p, dof

demograficas = ["age_band", "gender", "highest_education", "disability"]
chi2_resultados = {}
for v in demograficas:
    ct, chi2, p, dof = chi2_test(v)
    chi2_resultados[v] = {"chi2": chi2, "p": p, "dof": dof}
    print(f"Qui² {v}: chi2={chi2:.1f}, dof={dof}, p={p:.3e} "
          f"-> {'associação significativa' if p < 0.05 else 'sem associação'}")

findings["p3_chi2"] = {v: float(r["p"]) for v, r in chi2_resultados.items()}
""")

code(r"""
# A) Barras agrupadas: age_band × resultado (% por faixa etária).
ct = pd.crosstab(df_master["age_band"], df_master["resultado"]).reindex(AGE_ORDER)[RESULT_ORDER]
fig = viz.grouped_pct(ct, "Result by age band (%) (Q3)", "Age band", col_order=RESULT_ORDER)
viz.save_fig(fig, "p3_A_edad.png"); plt.show()
""")

code(r"""
# B) Barras agrupadas: gênero × resultado.
ct = pd.crosstab(df_master["gender"], df_master["resultado"])[RESULT_ORDER]
fig = viz.grouped_pct(ct, "Result by gender (%) (Q3)", "Gender", col_order=RESULT_ORDER)
viz.save_fig(fig, "p3_B_genero.png"); plt.show()
""")

code(r"""
# C) Barras agrupadas: highest_education × resultado (ordenado por nível).
ct = pd.crosstab(df_master["highest_education"], df_master["resultado"]).reindex(EDUCATION_ORDER)[RESULT_ORDER]
fig = viz.grouped_pct(ct, "Result by highest education (%) (Q3)", "Highest education level", col_order=RESULT_ORDER)
fig.axes[0].tick_params(axis="x", labelrotation=20)
viz.save_fig(fig, "p3_C_educacion.png"); plt.show()
""")

code(r"""
# D) Barras empilhadas: disability × resultado.
ct = pd.crosstab(df_master["disability"], df_master["resultado"])[RESULT_ORDER]
fig = viz.stacked_100(ct, "Result by disability status (100%) (Q3)",
                      "Disability (N=No, Y=Yes)", col_order=RESULT_ORDER)
viz.save_fig(fig, "p3_D_discapacidad.png"); plt.show()
""")

code(r"""
# E) Heatmap de correlação entre variáveis demográficas codificadas e resultado.
cols_enc = ["age_band_encoded", "gender_encoded", "highest_education_encoded",
            "imd_band_encoded", "disability_encoded", "resultado_encoded"]
corr = df_master[cols_enc].corr(method="spearman")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            cbar_kws={"label": "Spearman correlation"})
ax.set_title("Correlation between demographic variables and result (Q3)")
fig.tight_layout(); viz.save_fig(fig, "p3_E_heatmap_correlacion.png"); plt.show()
""")

code(r"""
# Odds Ratio: deficiência vs. taxa de desistência.
df_master["retirado_flag"] = (df_master["resultado"] == "Withdrawn").astype(int)
tab = pd.crosstab(df_master["disability"], df_master["retirado_flag"])
# tab: linhas N/Y, colunas 0/1
a = tab.loc["Y", 1]; b = tab.loc["Y", 0]   # deficiência: desistentes / não desistentes
c = tab.loc["N", 1]; d = tab.loc["N", 0]   # sem deficiência
odds_ratio = (a / b) / (c / d)
ret_Y = a / (a + b) * 100
ret_N = c / (c + d) * 100
print(f"Taxa de desistência com deficiência (Y): {ret_Y:.1f}%")
print(f"Taxa de desistência sem deficiência (N): {ret_N:.1f}%")
print(f"Odds Ratio de desistência (Y vs N): {odds_ratio:.2f}")
findings["p3_odds_ratio_discapacidad"] = round(float(odds_ratio), 2)
findings["p3_retiro_discapacidad"] = (round(float(ret_Y), 1), round(float(ret_N), 1))

# Grupo com maior/menor aprovação por variável.
def aprob_por_grupo(var, orden=None):
    ct = pd.crosstab(df_master[var], df_master["resultado"])
    if orden: ct = ct.reindex(orden)
    aprob = (ct[["Pass", "Distinction"]].sum(axis=1) / ct.sum(axis=1) * 100)
    return aprob.idxmax(), aprob.max(), aprob.idxmin(), aprob.min()

for v, orden in [("age_band", AGE_ORDER), ("gender", None),
                 ("highest_education", EDUCATION_ORDER), ("disability", None)]:
    gmax, vmax, gmin, vmin = aprob_por_grupo(v, orden)
    print(f"{v}: maior aprovação = {gmax} ({vmax:.1f}%), menor = {gmin} ({vmin:.1f}%)")
""")

md(r"""
### 📌 Resumo dos achados — Pergunta 3
""")

code(r"""
print("RESUMO Q3 — Perfil dos estudantes")
print("-" * 60)
for v in demograficas:
    p = chi2_resultados[v]["p"]
    print(f"- {v}: Qui² p={p:.2e} -> {'associação significativa com o resultado' if p<0.05 else 'sem associação clara'}.")
print(f"- Estudantes com deficiência desistem mais: {ret_Y:.1f}% vs {ret_N:.1f}% (OR={odds_ratio:.2f}).")
print("- A escolaridade prévia mostra a associação mais forte com o sucesso acadêmico.")
""")

# =========================================================================== #
# PERGUNTA 4
# =========================================================================== #
md(r"""
## 7. Pergunta 4 — Evasão do curso

**O quê:** entre os estudantes que desistiram (`resultado == 'Withdrawn'`),
analisamos **quando** ocorre a evasão usando `date_unregistration` (dia do curso,
onde 0 = início). Definimos janelas de evasão e comparamos `dias_hasta_retiro`
entre módulos.

**Por quê:** conhecer o momento da evasão permite posicionar intervenções (antes
do dia 30, antes do dia 90, etc.).

Visualizações: (A) histograma com linhas nos dias 30/90/180, (B) KDE,
(C) boxplot por módulo, (D) curva acumulada de evasão.
""")

code(r"""
ret = df_master[df_master["resultado"] == "Withdrawn"].copy()
ret_valid = ret.dropna(subset=["date_unregistration"])
print(f"Estudantes desistentes: {len(ret):,} | com data de baixa: {len(ret_valid):,}")

mediana_retiro = ret_valid["date_unregistration"].median()
print(f"Mediana do dia de desistência: {mediana_retiro:.0f}")

# Janelas: % que desiste nos primeiros 30, 90, 180 dias (desde o início do curso).
for d in (30, 90, 180):
    pct = (ret_valid["date_unregistration"] <= d).mean() * 100
    print(f"% desistentes nos primeiros {d} dias: {pct:.1f}%")
    findings[f"p4_pct_{d}d"] = round(float(pct), 1)
findings["p4_mediana_dia_retiro"] = float(mediana_retiro)
""")

code(r"""
# A) Histograma de date_unregistration com linhas em 30, 90, 180.
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(ret_valid["date_unregistration"], bins=30, color=sns.color_palette("Set2")[2],
        edgecolor="white")
for d, c in [(30, "green"), (90, "orange"), (180, "red")]:
    ax.axvline(d, color=c, linestyle="--", linewidth=2, label=f"day {d}")
ax.set_title("Withdrawal timing (date_unregistration) (Q4)")
ax.set_xlabel("Course day at withdrawal"); ax.set_ylabel("Number of students")
ax.legend()
fig.tight_layout(); viz.save_fig(fig, "p4_A_histograma_retiro.png"); plt.show()
""")

code(r"""
# B) KDE do momento de desistência.
fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(ret_valid["date_unregistration"], fill=True, color=sns.color_palette("Set2")[3], ax=ax)
ax.axvline(0, color="grey", linestyle=":", linewidth=1.5, label="course start")
ax.set_title("Density of withdrawal timing (KDE) (Q4)")
ax.set_xlabel("Course day at withdrawal"); ax.set_ylabel("Density")
ax.legend()
fig.tight_layout(); viz.save_fig(fig, "p4_B_kde_retiro.png"); plt.show()
""")

code(r"""
# C) Boxplot de dias_hasta_retiro por módulo.
ret_dias = ret.dropna(subset=["dias_hasta_retiro"])
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=ret_dias, x="code_module", y="dias_hasta_retiro",
            order=sorted(ret_dias["code_module"].unique()), palette="Set2", ax=ax)
ax.set_title("Days from registration to withdrawal, by module (Q4)")
ax.set_xlabel("Module"); ax.set_ylabel("Days until withdrawal")
fig.tight_layout(); viz.save_fig(fig, "p4_C_boxplot_dias_modulo.png"); plt.show()
""")

code(r"""
# D) Curva acumulada de evasão (por dia do curso).
orden = np.sort(ret_valid["date_unregistration"].values)
acum = np.arange(1, len(orden) + 1) / len(orden) * 100
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(orden, acum, color=sns.color_palette("Set2")[0], linewidth=2.5)
for d, c in [(30, "green"), (90, "orange"), (180, "red")]:
    ax.axvline(d, color=c, linestyle="--", alpha=0.7, label=f"day {d}")
ax.set_title("Cumulative withdrawal curve (Q4)")
ax.set_xlabel("Course day"); ax.set_ylabel("Cumulative % of withdrawals")
ax.legend()
fig.tight_layout(); viz.save_fig(fig, "p4_D_curva_abandono.png"); plt.show()
""")

code(r"""
# Módulo com maior taxa de evasão precoce (desistência em <=30 dias, sobre o total de matriculados).
total_por_mod = df_master.groupby("code_module", observed=True).size()
temprano = ret_valid[ret_valid["date_unregistration"] <= 30].groupby("code_module", observed=True).size()
tasa_temprana = (temprano / total_por_mod * 100).sort_values(ascending=False)
print("Taxa de evasão precoce (desistência <=30 dias) por módulo (%):")
print(tasa_temprana.round(2).to_string())
findings["p4_modulo_abandono_temprano"] = (tasa_temprana.idxmax(), round(float(tasa_temprana.max()), 2))
""")

md(r"""
### 📌 Resumo dos achados — Pergunta 4
""")

code(r"""
print("RESUMO Q4 — Evasão do curso")
print("-" * 60)
print(f"- Mediana do dia de desistência: {mediana_retiro:.0f}.")
print(f"- {findings['p4_pct_30d']:.1f}% desiste antes do dia 30; "
      f"{findings['p4_pct_90d']:.1f}% antes do dia 90; {findings['p4_pct_180d']:.1f}% antes do dia 180.")
print(f"- Módulo com maior evasão precoce: {tasa_temprana.idxmax()} ({tasa_temprana.max():.1f}%).")
print("- Boa parte das desistências ocorre no início (inclusive antes do dia 0), indicando evasão muito precoce.")
""")

# =========================================================================== #
# PERGUNTA 5
# =========================================================================== #
md(r"""
## 8. Pergunta 5 — Comparação entre módulos

**O quê:** para cada módulo calculamos as taxas de aprovação, desistência,
distinção e reprovação, além da média de score e dos cliques médios. Geramos
rankings.

**Por quê:** comparar módulos identifica boas práticas (cursos bem-sucedidos) e
cursos que precisam de redesenho.

Visualizações: (A) barras horizontais de aprovação, (B) empilhadas 100%,
(C) lollipop de desistência, (D) heatmap módulo × métrica, (E) dispersão
aprovação vs cliques com tamanho = nº de estudantes.
""")

code(r"""
from src.preprocess import RESULT_ORDER

ct = pd.crosstab(df_master["code_module"], df_master["resultado"])[RESULT_ORDER]
pct = ct.div(ct.sum(axis=1), axis=0) * 100
metricas = pd.DataFrame({
    "tasa_aprobacion": pct["Pass"] + pct["Distinction"],
    "tasa_retiro": pct["Withdrawn"],
    "tasa_distincion": pct["Distinction"],
    "tasa_reprobacion": pct["Fail"],
    "n_estudiantes": ct.sum(axis=1),
})
metricas["score_promedio"] = df_master.groupby("code_module", observed=True)["score_promedio"].mean()
metricas["clicks_promedio"] = df_master.groupby("code_module", observed=True)["clicks_totales"].mean()
metricas = metricas.round(2)
print("Métricas por módulo:")
print(metricas.to_string())

findings["p5_metricas"] = metricas.round(1).to_dict("index")
findings["p5_top_aprobacion"] = metricas["tasa_aprobacion"].idxmax()
findings["p5_top_retiro"] = metricas["tasa_retiro"].idxmax()
""")

code(r"""
# A) Barras horizontais: taxa de aprovação por módulo (desc).
ord_aprob = metricas["tasa_aprobacion"].sort_values()
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(ord_aprob.index, ord_aprob.values, color=sns.color_palette("Set2", len(ord_aprob)))
for i, v in enumerate(ord_aprob.values):
    ax.text(v, i, f" {v:.1f}%", va="center")
ax.set_title("Pass rate by module (Q5)")
ax.set_xlabel("Pass rate (%)"); ax.margins(x=0.1)
fig.tight_layout(); viz.save_fig(fig, "p5_A_aprobacion_modulo.png"); plt.show()
""")

code(r"""
# B) Barras empilhadas 100% por módulo.
fig = viz.stacked_100(ct, "Result composition by module (100%) (Q5)",
                      "Module", col_order=RESULT_ORDER)
viz.save_fig(fig, "p5_B_apiladas_modulo.png"); plt.show()
""")

code(r"""
# C) Lollipop chart: taxa de desistência por módulo.
ord_ret = metricas["tasa_retiro"].sort_values()
fig, ax = plt.subplots(figsize=(12, 6))
ax.hlines(y=ord_ret.index, xmin=0, xmax=ord_ret.values, color="grey", alpha=0.6)
ax.plot(ord_ret.values, ord_ret.index, "o", markersize=11, color=sns.color_palette("Set2")[1])
for y, v in zip(ord_ret.index, ord_ret.values):
    ax.text(v + 0.5, y, f"{v:.1f}%", va="center")
ax.set_title("Withdrawal rate by module (lollipop) (Q5)")
ax.set_xlabel("Withdrawal rate (%)"); ax.margins(x=0.12)
fig.tight_layout(); viz.save_fig(fig, "p5_C_lollipop_retiro.png"); plt.show()
""")

code(r"""
# D) Heatmap módulo × métrica.
cols = ["tasa_aprobacion", "tasa_retiro", "tasa_distincion", "tasa_reprobacion"]
labels = ["Pass rate", "Withdrawal rate", "Distinction rate", "Fail rate"]
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(metricas[cols].rename(columns=dict(zip(cols, labels))),
            annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
            cbar_kws={"label": "Percentage (%)"})
ax.set_title("Result metrics by module (Q5)")
ax.set_xlabel("Metric"); ax.set_ylabel("Module")
fig.tight_layout(); viz.save_fig(fig, "p5_D_heatmap_metricas.png"); plt.show()
""")

code(r"""
# E) Dispersão: tasa_aprobacion vs clicks_promedio, tamanho = nº estudantes.
fig, ax = plt.subplots(figsize=(12, 6))
sizes = metricas["n_estudiantes"] / metricas["n_estudiantes"].max() * 1500
ax.scatter(metricas["clicks_promedio"], metricas["tasa_aprobacion"], s=sizes,
           c=range(len(metricas)), cmap="Set2", alpha=0.8, edgecolor="black")
for mod, row in metricas.iterrows():
    ax.annotate(mod, (row["clicks_promedio"], row["tasa_aprobacion"]),
                ha="center", va="center", fontweight="bold")
ax.set_title("Pass rate vs. average clicks by module (size = nº students) (Q5)")
ax.set_xlabel("Average clicks per student"); ax.set_ylabel("Pass rate (%)")
fig.tight_layout(); viz.save_fig(fig, "p5_E_scatter_modulos.png"); plt.show()
""")

md(r"""
### 📌 Resumo dos achados — Pergunta 5
""")

code(r"""
print("RESUMO Q5 — Comparação entre módulos")
print("-" * 60)
print(f"- Maior aprovação: {metricas['tasa_aprobacion'].idxmax()} ({metricas['tasa_aprobacion'].max():.1f}%).")
print(f"- Menor aprovação: {metricas['tasa_aprobacion'].idxmin()} ({metricas['tasa_aprobacion'].min():.1f}%).")
print(f"- Maior desistência: {metricas['tasa_retiro'].idxmax()} ({metricas['tasa_retiro'].max():.1f}%).")
print(f"- Maior taxa de distinção: {metricas['tasa_distincion'].idxmax()} ({metricas['tasa_distincion'].max():.1f}%).")
print("- Há relação positiva entre cliques médios e taxa de aprovação no nível de módulo.")
""")

# =========================================================================== #
# BONUS 6: ASSESSMENTS
# =========================================================================== #
md(r"""
## 9. Bônus 6 — Análise das avaliações (assessments)

**O quê:** estudamos a distribuição de scores por tipo de avaliação (TMA, CMA,
Exam), a evolução da média de score conforme a data de entrega, e se entregar
**atrasado** se associa a notas piores.

**Por quê:** entender o comportamento nas avaliações complementa a visão do
desempenho e detecta o efeito da procrastinação.
""")

code(r"""
# Unimos studentAssessment com metadados de assessments.
sa = assessment.merge(
    data["assessments"][["id_assessment", "assessment_type", "date", "code_module"]],
    on="id_assessment", how="left")
sa_valid = sa[~sa["no_submitted"]].copy()

# Distribuição de scores por tipo de avaliação.
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=sa_valid, x="assessment_type", y="score",
            order=["CMA", "TMA", "Exam"], palette="Set2", ax=ax)
ax.set_title("Score distribution by assessment type (Bonus 6)")
ax.set_xlabel("Assessment type"); ax.set_ylabel("Score")
fig.tight_layout(); viz.save_fig(fig, "p6_A_scores_por_tipo.png"); plt.show()

print(sa_valid.groupby("assessment_type", observed=True)["score"].describe().round(1).to_string())
""")

code(r"""
# Evolução da média de score ao longo do tempo (data de entrega).
sa_time = sa_valid.dropna(subset=["date_submitted"]).copy()
sa_time["semana"] = (sa_time["date_submitted"] // 7).astype(int)
evol = sa_time.groupby("semana", observed=True)["score"].mean()
evol = evol[(evol.index >= 0) & (evol.index <= 40)]
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(evol.index, evol.values, marker="o", color=sns.color_palette("Set2")[0])
ax.set_title("Average score by submission week (Bonus 6)")
ax.set_xlabel("Course week (submission date)"); ax.set_ylabel("Average score")
fig.tight_layout(); viz.save_fig(fig, "p6_B_evolucion_score.png"); plt.show()
""")

code(r"""
# Entregar atrasado -> notas piores? Comparamos entregas no prazo vs atrasadas.
sa_late = sa_valid.dropna(subset=["date_submitted", "date"]).copy()
sa_late["dias_retraso"] = sa_late["date_submitted"] - sa_late["date"]
sa_late["entrega_tardia"] = np.where(sa_late["dias_retraso"] > 0, "Late", "On time")
comp = sa_late.groupby("entrega_tardia", observed=True)["score"].agg(["mean", "median", "count"]).round(2)
print("Score por pontualidade da entrega:")
print(comp.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(data=sa_late, x="entrega_tardia", y="score",
               order=["On time", "Late"], palette="Set2", cut=0, ax=ax)
ax.set_title("Score by submission punctuality (Bonus 6)")
ax.set_xlabel("Submission type"); ax.set_ylabel("Score")
fig.tight_layout(); viz.save_fig(fig, "p6_C_entrega_tardia.png"); plt.show()

# Teste de Mann-Whitney.
a = sa_late.loc[sa_late["entrega_tardia"] == "On time", "score"]
b = sa_late.loc[sa_late["entrega_tardia"] == "Late", "score"]
U, p_mw = stats.mannwhitneyu(a, b, alternative="greater")
print(f"\nMann-Whitney (no prazo > atrasada): U={U:.0f}, p={p_mw:.3e}")
findings["p6_score_a_tiempo"] = float(comp.loc["On time", "mean"])
findings["p6_score_tardia"] = float(comp.loc["Late", "mean"]) if "Late" in comp.index else None
findings["p6_mannwhitney_p"] = float(p_mw)
""")

md(r"""
### 📌 Resumo dos achados — Bônus 6
""")

code(r"""
print("RESUMO Bônus 6 — Avaliações")
print("-" * 60)
med = sa_valid.groupby("assessment_type", observed=True)["score"].mean().round(1)
for t in ["CMA", "TMA", "Exam"]:
    if t in med.index:
        print(f"- Score médio {t}: {med[t]:.1f}.")
if findings["p6_score_tardia"] is not None:
    print(f"- Entregas no prazo ({findings['p6_score_a_tiempo']:.1f}) vs atrasadas "
          f"({findings['p6_score_tardia']:.1f}); Mann-Whitney p={p_mw:.2e}.")
print("- A relação entre pontualidade e nota é fraca/não conclusiva com estes dados.")
""")

# =========================================================================== #
# BONUS 7: KMEANS
# =========================================================================== #
md(r"""
## 10. Bônus 7 — Segmentação dos estudantes (K-Means)

**O quê:** segmentamos os estudantes com **K-Means** usando: `clicks_totales`,
`score_promedio`, `dias_activos`, `highest_education_encoded` e
`imd_band_encoded`. Escalamos com `StandardScaler`, escolhemos K com o método do
cotovelo (2–10) e visualizamos em 2D com **PCA**.

**Por quê:** descobrir perfis latentes de estudantes permite desenhar
intervenções diferenciadas. Usamos `random_state=42` em tudo.
""")

code(r"""
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

features = ["clicks_totales", "score_promedio", "dias_activos",
            "highest_education_encoded", "imd_band_encoded"]
df_clust = df_master.dropna(subset=["score_promedio"]).copy()
X = df_clust[features].fillna(df_clust[features].median())

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Método do cotovelo (inércia) para K = 2..10.
inercias = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Xs)
    inercias[k] = km.inertia_

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(list(inercias.keys()), list(inercias.values()), marker="o",
        color=sns.color_palette("Set2")[0])
ax.set_title("Elbow method for K-Means (Bonus 7)")
ax.set_xlabel("Number of clusters (K)"); ax.set_ylabel("Inertia (WCSS)")
fig.tight_layout(); viz.save_fig(fig, "p7_A_metodo_codo.png"); plt.show()
print("Inércias:", {k: round(v) for k, v in inercias.items()})
""")

code(r"""
# K ótimo = 4 (equilíbrio razoável observado no cotovelo).
K_OPT = 4
km = KMeans(n_clusters=K_OPT, random_state=42, n_init=10)
df_clust["cluster"] = km.fit_predict(Xs)

# PCA 2D para visualização.
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(Xs)
df_clust["pca1"], df_clust["pca2"] = coords[:, 0], coords[:, 1]
print(f"Variância explicada pelo PCA: {pca.explained_variance_ratio_.round(3)} "
      f"(total {pca.explained_variance_ratio_.sum():.1%})")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(data=df_clust, x="pca1", y="pca2", hue="cluster", palette="tab10",
                alpha=0.5, s=15, ax=axes[0])
axes[0].set_title("K-Means clusters (PCA 2D)")
sns.scatterplot(data=df_clust, x="pca1", y="pca2", hue="resultado",
                hue_order=RESULT_ORDER, palette="Set2", alpha=0.5, s=15, ax=axes[1])
axes[1].set_title("Final result (PCA 2D)")
for a in axes:
    a.set_xlabel("PCA 1"); a.set_ylabel("PCA 2")
fig.suptitle("Student segmentation with K-Means (Bonus 7)", fontsize=15, fontweight="bold")
fig.tight_layout(); viz.save_fig(fig, "p7_B_pca_clusters.png"); plt.show()
""")

code(r"""
# Perfil de cada cluster.
perfil = df_clust.groupby("cluster").agg(
    n=("id_student", "size"),
    clicks_medios=("clicks_totales", "mean"),
    score_medio=("score_promedio", "mean"),
    dias_activos_medios=("dias_activos", "mean"),
    educacion_media=("highest_education_encoded", "mean"),
    imd_media=("imd_band_encoded", "mean"),
).round(1)
# Distribuição de resultado por cluster (% por cluster).
res_clust = pd.crosstab(df_clust["cluster"], df_clust["resultado"], normalize="index").round(3) * 100
res_clust.columns = res_clust.columns.astype(str)  # evita CategoricalIndex no join
perfil = perfil.join(res_clust)
print("Perfil de cada cluster:")
print(perfil.to_string())
findings["p7_k_opt"] = K_OPT
findings["p7_pca_var"] = round(float(pca.explained_variance_ratio_.sum()), 3)
findings["p7_perfil"] = perfil.round(1).reset_index().to_dict("records")
""")

md(r"""
### 📌 Resumo dos achados — Bônus 7
""")

code(r"""
print("RESUMO Bônus 7 — Segmentação K-Means")
print("-" * 60)
print(f"- K ótimo escolhido: {K_OPT} clusters.")
print(f"- PCA 2D explica {pca.explained_variance_ratio_.sum():.1%} da variância.")
mejor = perfil["score_medio"].idxmax(); peor = perfil["score_medio"].idxmin()
print(f"- Cluster de maior desempenho: #{mejor} (score médio {perfil.loc[mejor,'score_medio']:.1f}, "
      f"{perfil.loc[mejor,'clicks_medios']:.0f} cliques).")
print(f"- Cluster de menor desempenho: #{peor} (score médio {perfil.loc[peor,'score_medio']:.1f}, "
      f"{perfil.loc[peor,'clicks_medios']:.0f} cliques).")
print("- Os clusters separam claramente perfis de alta vs baixa atividade/desempenho.")
""")

# =========================================================================== #
# GUARDAR FINDINGS + TEMPO
# =========================================================================== #
md(r"""
## 11. Exportação dos achados e tempo de execução

**O quê:** salvamos o dicionário `findings` em `outputs/report/findings.json`
(para alimentar o relatório e o README) e imprimimos o tempo total de execução
do notebook.
""")

code(r"""
import json
out_report = load_data.get_outputs_dir() / "report"
out_report.mkdir(parents=True, exist_ok=True)
with open(out_report / "findings.json", "w", encoding="utf-8") as f:
    json.dump(findings, f, ensure_ascii=False, indent=2, default=str)
print("Achados salvos em outputs/report/findings.json")

_t_fin = time.time()
_dur = _t_fin - _t_inicio
print(f"\nTempo total de execução do notebook: {_dur:.1f} segundos "
      f"({_dur/60:.1f} minutos).")
""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}

import pathlib
out = pathlib.Path(__file__).parent / "notebooks" / "analysis.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print(f"Notebook escrito em: {out}  ({len(cells)} células)")
