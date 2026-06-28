# 工作一：TableZoomer确定性规则版修改路线

> 适用目标：将TableZoomer中的query-aware table zooming阶段改造为“问题重写+单元格检索+确定性行列扩展+证据子表构建”的细粒度证据定位方法。  
> 核心原则：不训练reranker，不做复杂加权聚合，不引入难解释的启发式权重；保留TableZoomer后续代码生成、代码执行和答案总结流程。

---

## 1.工作一方法定位

### 1.1方法名称

推荐中文名称：

**基于问题重写与单元格检索的细粒度证据定位表格问答方法**

推荐英文简称：

也可以在代码中命名为：

```text
cell_guided_zoom
```

### 1.2核心目标

原始TableZoomer主要通过问题分析得到相关列和实体匹配信息，再生成查询相关子表。本文工作一的目标是将其改造成：

```text
Question
→ Question Rewrite
→ Cell-level Retrieval
→ Deterministic Row/Column Expansion
→ Evidence Subtable
→ Code Generation / Answer Generation
```

也就是说，重点修改的是TableZoomer中的**table zooming阶段**，而不是重写整个TableZoomer系统。

---

## 2.总体修改原则

### 2.1保留的部分

尽量保留TableZoomer原有的：

```text
1. table_schema读取或构建逻辑；
2. ReAct循环；
3. code generation；
4. code execution；
5. answer summary；
6. 原始query planning/table zooming逻辑作为baseline。
```

### 2.2修改的部分

主要新增或替换：

```text
1. 表感知问题重写模块；
2. 单元格索引构建模块；
3. 无训练单元格检索模块；
4. 确定性行列扩展模块；
5. 证据子表构建模块；
6. 实验日志输出模块。
```

### 2.3明确不做的部分

```text
1. 不训练reranker；
2. 不微调模型；
3. 不做复杂行列加权；
4. 不使用target column×1.3、constraint column×1.2这类人工权重；
5. 不把每个cell都送入LLM判断；
6. 不在每轮ReAct中重复问题重写，默认每个问题只重写一次。
```

---

## 3.最终系统流程

完整流程如下：

```text
输入：Question + Table + Table Schema

Step0：读取table_schema和原始表格文件

Step1：表感知问题重写
    输入：question + table columns + table description + table preview
    输出：rewritten_query、target_columns、constraint_columns、entities、operations、cell_search_queries

Step2：单元格索引构建
    输入：table_file + table_schema
    输出：cell_items

Step3：无训练单元格检索
    输入：rewrite_profile + cell_items
    输出：top_k_cells

Step4：确定性行列扩展
    输入：top_k_cells + rewrite_profile + original_table
    输出：selected_rows + selected_columns

Step5：证据子表构建
    输入：selected_rows + selected_columns
    输出：refined_table_schema + table_zoom

Step6：程序化推理或答案生成
    复用原TableZoomer的CodeGenerator和AnswerSummary
    （在QA时考虑添加表格名称信息（如有title））
```

一句话概括：

> 用“问题重写增强的单元格级证据检索”替换原始TableZoomer中偏列级和实体级的table zooming过程，并通过确定性行列扩展将离散cell恢复为结构化证据子表。

---

## 4.需要修改或新增的文件

具体文件名以实际TableZoomer仓库为准。建议按以下方式组织：

```text
table_agent.py
actions/query_analyse.py
actions/cell_retrieval.py             # 新增
prompts/question_rewrite_prompt.txt   # 新增
agent_config/example.yaml             # 可选修改
```

可选新增：

```text
cache/cell_index/
outputs/cell_guided_zoom_logs/
```

---

## 5.在table_agent.py中添加开关

### 5.1新增初始化参数

在TableZoomer主类中加入以下参数：

```python
def __init__(
    self,
    config_file,
    max_react_round=5,
    use_cell_guided_zoom=False,
    zoom_mode="cell_to_subtable",
    cell_retrieval_method="tfidf",
    top_k_cells=50,
    top_k_rows=25,
    top_k_cols=10,
    max_row_context_cols=6,
    max_cell_text_chars=300,
    max_query_rewrite_queries=5,
    **kwargs
):
    ...
```

### 5.2保留原始流程作为baseline

在`act_pipeline()`或对应主流程中加入分支：

```python
if self.use_cell_guided_zoom:
    refined_table_schema = self.run_cell_guided_zoom(
        query=query,
        table_schema=table_schema,
        table_file=table_file,
        log_item=log_item
    )
else:
    refined_table_schema = original_tablezoomer_zooming_logic(...)
```

要求：

```text
1. use_cell_guided_zoom=False时，原TableZoomer行为完全不变；
2. use_cell_guided_zoom=True时，进入新流程；
3. 新流程输出的refined_table_schema和table_zoom需要尽量兼容原后续CodeGenerator。
```

---

## 6.新增问题重写模块，子问题分别检索top-k cell

### 6.1目标

问题重写模块不是为了回答问题，而是为了生成面向单元格检索的结构化语义信号。

输入：

```text
question
table title(如有)
table columns
table description
table preview，最多3行
```

输出：

```json
{
  "rewritten_query": "...",
  "target_columns": ["..."],
  "constraint_columns": ["..."],
  "entities": ["..."],
  "operations": ["filter", "compare", "rank", "aggregate"],
  "cell_search_queries": ["...", "..."]
}
```

### 6.2新增Prompt

新增文件：

```text
prompts/question_rewrite_prompt.txt
```

内容建议：

```text
You are a table-aware query rewriter.

Given a table question and table schema, rewrite the question into retrieval-oriented signals.

Return compact JSON only:
{
  "rewritten_query": "...",
  "target_columns": [...],
  "constraint_columns": [...],
  "entities": [...],
  "operations": [...],
  "cell_search_queries": [...]
}

Definitions:
- target_columns: columns likely needed to produce the final answer.
- constraint_columns: columns likely used for filtering, comparison, ranking, aggregation, or conditions.
- entities: explicit values or entities mentioned in the question.
- operations: operations such as lookup, filter, compare, rank, max, min, count, sum, average, difference.
- cell_search_queries: short retrieval queries for finding relevant cells.

Rules:
1. Use column names from the given table schema when possible.
2. Do not answer the question.
3. Do not invent columns that are not in the schema unless necessary; if uncertain, leave empty.
4. Return JSON only.

Question:
{query}

Table columns:
{column_list}

Table description:
{table_description}

Table preview:
{table_preview}
```

### 6.3新增类或函数

可以新增：

```python
class QuestionRewrite(Action):
    name: str = "QuestionRewrite"
```

或者写成函数：

```python
def rewrite_question(query, table_schema, table_preview, llm):
    ...
```

### 6.4解析失败处理

如果LLM输出不是合法JSON，不要中断主流程，使用fallback：

```python
rewrite_profile = {
    "rewritten_query": query,
    "target_columns": [],
    "constraint_columns": [],
    "entities": [],
    "operations": [],
    "cell_search_queries": [query]
}
```

---

## 7.新增单元格索引构建模块

### 7.1新增文件

```text
actions/cell_retrieval.py
```

### 7.2cell item格式

每个cell构造成一个可检索对象：

```python
{
    "row_id": 12,
    "col_name": "Gold medals",
    "cell_value": "48",
    "row_context": "Country=China; Year=2008; Total=100",
    "text": "Column: Gold medals. Cell value: 48. Row context: Country=China; Year=2008; Total=100."
}
```

### 7.3核心函数

```python
def build_cell_index(
    table_file,
    table_schema,
    max_row_context_cols=6,
    max_cell_text_chars=300
):
    """
    读取原始表格，将每个非空cell转化为cell item。
    """
```

### 7.4cell文本构造规则

不要只使用cell value。推荐格式：

```text
Table: {table_name}
Column: {col_name}
Cell value: {cell_value}
Row context: {selected row values}
```

其中`row_context`不需要包含整行所有列，默认最多保留`max_row_context_cols`个非空字段。

优先选择：

```text
1. 当前cell所在列；
2. 问题重写中的target_columns；
3. 问题重写中的constraint_columns；
4. 前几个非空列；
5. 数值列或实体列。
```

最小实现可以先不做复杂选择，直接取该行前`max_row_context_cols`个非空字段。

### 7.5缓存

为了避免重复构建cell index，建议缓存：

```text
cache/cell_index/{dataset_name}/{table_id}.json
```

如果暂时不方便获取`dataset_name/table_id`，可以用表格文件路径的hash作为缓存名。

---

## 8.新增无训练单元格检索（对于重写情况用子问题检索，未重写用原问题检索）

### 8.1第一版优先实现TF-IDF

先实现简单、稳定、外部依赖较少的TF-IDF检索：

```python
def tfidf_cell_retrieve(cell_items, rewrite_profile, top_k=50):
    """
    输入cell_items和rewrite_profile，返回top_k_cells。
    """
```

检索query由以下内容拼接：

```text
rewritten_query
cell_search_queries
entities
target_columns
constraint_columns
operations
```

构造：

```python
query_text = " ".join([
    rewrite_profile.get("rewritten_query", ""),
    " ".join(rewrite_profile.get("cell_search_queries", [])),
    " ".join(rewrite_profile.get("entities", [])),
    " ".join(rewrite_profile.get("target_columns", [])),
    " ".join(rewrite_profile.get("constraint_columns", [])),
    " ".join(rewrite_profile.get("operations", [])),
])
```

输出格式：

```python
{
    "row_id": 12,
    "col_name": "Gold medals",
    "cell_value": "48",
    "score": 0.73,
    "text": "..."
}
```

### 8.2后续可选实现Embedding/Hybrid

可选支持：

```text
--cell_retrieval_method tfidf
--cell_retrieval_method dense
--cell_retrieval_method hybrid
```

但第一版只实现`tfidf`即可。

### 8.3不训练reranker

明确不要做：

```text
1. 不构造正负样本训练；
2. 不微调embedding模型；
3. 不训练cross-encoder reranker；
4. 不训练LLM。
```

---

## 9.确定性行列扩展模块

这是本版文档的核心变化：**不使用复杂加权行列聚合**，改用确定性规则。

### 9.1模块名称

建议命名为：

```python
deterministic_expand_cells_to_subtable
```

论文中建议命名为：

**基于单元格证据的结构化扩展**

或：

**单元格证据驱动的证据子表构建**

不要命名为“加权行列聚合”。

### 9.2核心思想

从top-k检索结果出发：

```text
1. 保留命中cell所在行；
2. 保留命中cell所在列；
3. 补充问题重写识别出的target_columns；
4. 补充问题重写识别出的constraint_columns；
5. 如果行列超出预算，则使用确定性截断规则。
```

### 9.3核心规则

```python
selected_rows = rows containing retrieved cells

selected_columns = columns containing retrieved cells
selected_columns = selected_columns ∪ target_columns
selected_columns = selected_columns ∪ constraint_columns
```

### 9.4截断规则

如果`selected_rows`超过`top_k_rows`：

```text
按照每一行中最高命中cell的检索分数排序，保留前top_k_rows行。
```

如果`selected_columns`超过`top_k_cols`：

```text
优先保留顺序：
1. target_columns；
2. constraint_columns；
3. 被top-k cell命中的列，按命中次数从高到低；
4. 如果命中次数相同，按该列最高cell分数从高到低；
5. 仍然超过则截断到top_k_cols。
```

这不是人工加权，而是确定性排序，便于解释。

### 9.5伪代码

```python
def deterministic_expand_cells_to_subtable(
    top_k_cells,
    rewrite_profile,
    all_columns,
    top_k_rows=25,
    top_k_cols=10
):
    target_cols = normalize_columns(
        rewrite_profile.get("target_columns", []),
        all_columns
    )
    constraint_cols = normalize_columns(
        rewrite_profile.get("constraint_columns", []),
        all_columns
    )

    # 1.命中行
    row_best_score = {}
    for cell in top_k_cells:
        row_id = cell["row_id"]
        score = cell.get("score", 0.0)
        row_best_score[row_id] = max(row_best_score.get(row_id, 0.0), score)

    selected_rows = sorted(
        row_best_score.keys(),
        key=lambda r: row_best_score[r],
        reverse=True
    )[:top_k_rows]

    # 2.命中列统计
    col_hit_count = {}
    col_best_score = {}
    for cell in top_k_cells:
        col = cell["col_name"]
        score = cell.get("score", 0.0)
        col_hit_count[col] = col_hit_count.get(col, 0) + 1
        col_best_score[col] = max(col_best_score.get(col, 0.0), score)

    hit_cols = sorted(
        col_hit_count.keys(),
        key=lambda c: (col_hit_count[c], col_best_score.get(c, 0.0)),
        reverse=True
    )

    # 3.确定性列补全
    selected_columns = []
    for col in target_cols + constraint_cols + hit_cols:
        if col in all_columns and col not in selected_columns:
            selected_columns.append(col)

    selected_columns = selected_columns[:top_k_cols]

    return selected_rows, selected_columns
```

### 9.6列名规范化

LLM重写输出的列名可能与真实列名不完全一致，因此需要轻量匹配：

```python
def normalize_columns(candidate_cols, all_columns):
    """
    将rewrite_profile中的列名映射到真实表头。
    可以使用：
    1. exact match；
    2. lower-case match；
    3. substring match；
    4. difflib.get_close_matches。
    """
```

匹配不到就跳过，不要强行创造新列。

---

## 10.构建证据子表table_zoom

### 10.1核心函数

```python
def build_cell_guided_table_zoom(
    table_file,
    table_schema,
    selected_rows,
    selected_columns,
    top_k_cells=None
):
    """
    根据确定性扩展得到的行列，构建refined_table_schema和table_zoom。
    """
```

### 10.2table_zoom格式

尽量兼容TableZoomer原后续模块：

```python
table_zoom = {
    "header": selected_columns,
    "rows": selected_df[selected_columns].values.tolist()
}
```

### 10.3refined_table_schema格式

建议保留原schema中的基本字段，并替换列信息和样例：

```python
refined_table_schema = {
    "file_path": table_schema.get("file_path"),
    "table_name": table_schema.get("table_name", ""),
    "table_description": table_schema.get("table_description", ""),
    "number_of_rows": len(selected_rows),
    "column_list": selected_columns,
    "table_zoom": table_zoom,
    "evidence_cells": top_k_cells[:20] if top_k_cells else []
}
```

如果原TableZoomer的schema字段名不同，以实际代码需要为准，核心目标是：

```text
1. 后续CodeGenerator能读取selected_columns；
2. 后续CodeGenerator能看到table_zoom；
3. 日志中能记录evidence_cells。
```

---

## 11.在table_agent.py中的推荐集成方式

新增一个主函数：

```python
def run_cell_guided_zoom(self, query, table_schema, table_file, log_item=None):
    # Step1: question rewrite
    rewrite_profile = self.rewrite_question(query, table_schema)

    # Step2: build/load cell index
    cell_items = build_or_load_cell_index(
        table_file=table_file,
        table_schema=table_schema,
        max_row_context_cols=self.max_row_context_cols,
        max_cell_text_chars=self.max_cell_text_chars
    )

    # Step3: cell retrieval
    top_k_cells = retrieve_cells(
        cell_items=cell_items,
        rewrite_profile=rewrite_profile,
        method=self.cell_retrieval_method,
        top_k=self.top_k_cells
    )

    # Step4: deterministic expansion
    selected_rows, selected_columns = deterministic_expand_cells_to_subtable(
        top_k_cells=top_k_cells,
        rewrite_profile=rewrite_profile,
        all_columns=get_all_columns(table_schema),
        top_k_rows=self.top_k_rows,
        top_k_cols=self.top_k_cols
    )

    # Step5: build table_zoom
    refined_table_schema = build_cell_guided_table_zoom(
        table_file=table_file,
        table_schema=table_schema,
        selected_rows=selected_rows,
        selected_columns=selected_columns,
        top_k_cells=top_k_cells
    )

    # Step6: logging
    if log_item is not None:
        log_item["cell_guided_zoom"] = {
            "rewrite_profile": rewrite_profile,
            "top_k_cells": top_k_cells[:20],
            "selected_rows": selected_rows,
            "selected_columns": selected_columns,
            "subtable_shape": [
                len(selected_rows),
                len(selected_columns)
            ],
            "compression_ratio": compute_compression_ratio(...)
        }

    return refined_table_schema
```

---


## 13.Token和运行成本控制

工作一中需要控制LLM调用和输入长度。

### 13.1LLM调用策略

```text
1. 每个问题只调用一次Question Rewrite；
2. cell retrieval不调用LLM；
3. deterministic expansion不调用LLM；
4. table_zoom构建不调用LLM；
5. 后续只复用原TableZoomer的CodeGenerator/AnswerSummary调用。
```

### 13.2输入长度控制

```text
1. Question Rewrite不输入完整表，只输入列名、表描述和最多3行preview；
2. cell_text最多300字符；
3. row_context最多6列；
4. top_k_cells默认50；
5. selected_rows默认25；
6. selected_columns默认10；
7. table_zoom不超过25×10；
8. 日志中top_k_cells默认只保存前20个完整文本，完整结果可选保存。
```

### 13.3推荐默认参数

```python
top_k_cells = 50
top_k_rows = 25
top_k_cols = 10
max_row_context_cols = 6
max_cell_text_chars = 300
max_query_rewrite_queries = 5
cell_retrieval_method = "tfidf"
zoom_mode = "cell_to_subtable"
```

---

## 14.日志字段

每个样本建议输出：

```json
{
  "question_id": "...",
  "question": "...",
  "gold_answer": "...",
  "pred_answer": "...",
  "zoom_mode": "cell_to_subtable",
  "rewrite_profile": {
    "rewritten_query": "...",
    "target_columns": [],
    "constraint_columns": [],
    "entities": [],
    "operations": [],
    "cell_search_queries": []
  },
  "top_k_cells": [
    {
      "row_id": 12,
      "col_name": "Gold medals",
      "cell_value": "48",
      "score": 0.73,
      "text": "..."
    }
  ],
  "selected_rows": [12, 15, 20],
  "selected_columns": ["Country", "Year", "Gold medals"],
  "subtable_shape": [25, 3],
  "compression_ratio": 0.18,
  "history": "..."
}
```

后续论文可统计：

```text
1. Answer Accuracy / EM；
2. Evidence Recall，如果数据集有supporting cells；
3. Row Recall；
4. Column Recall；
5. Subtable Compression Ratio；
6. Average Token Length；
7. Code Execution Success Rate。
```

---

## 15.实验设计

### 15.1主实验对比

```text
1. Full Table + LLM
2. Full Table + PoT
3. Original TableZoomer
4. Question Rewrite + Cell-to-Subtable，即本文完整方法
5. Binder
6. Dater
   ...
```

### 15.2消融实验

```text
1. w/o Question Rewrite
2. Direct Cell Evidence，不构建子表
3. w/o Row Context in Cell Text
4. top_k_cells = 10 / 20 / 50 / 100
5. top_k_rows = 10 / 25 / 50
6. top_k_cols = 5 / 10 / 15
7. TF-IDF vs Embedding vs Hybrid，可选
```

### 15.3应该重点证明什么

```text
1. 问题重写是否提升cell检索质量；
2. 直接cell evidence是否已经优于完整表输入；
3. cell-to-subtable是否优于direct cell evidence；
4. 证据子表是否减少token并提升或保持准确率；
5. 确定性扩展是否比复杂权重更易解释且稳定。
```

---

## 16.给Codex的分阶段开发顺序

### 阶段1：加开关，保证原流程不变

```text
1. 在TableZoomer主类中添加use_cell_guided_zoom和zoom_mode参数；
2. 在act_pipeline中添加分支；
3. use_cell_guided_zoom=False时完全走原始逻辑；
4. 跑通原始TableZoomer。
```

### 阶段2：实现Question Rewrite

```text
1. 新增question_rewrite_prompt.txt；
2. 新增rewrite_question函数或QuestionRewrite类；
3. 只输入question、columns、description、preview；
4. 输出rewrite_profile；
5. 解析失败时fallback为原问题。
```

### 阶段3：实现cell index

```text
1. 新增actions/cell_retrieval.py；
2. 实现build_cell_index；
3. 每个cell保存row_id、col_name、cell_value、row_context、text；
4. 加缓存机制，可选。
```

### 阶段4：实现TF-IDF cell retrieval

```text
1. 实现tfidf_cell_retrieve；
2. query_text由rewrite_profile拼接；
3. 返回top_k_cells；
4. 保存top_k_cells到日志。
```

### 阶段5：实现确定性行列扩展

```text
1. 实现deterministic_expand_cells_to_subtable；
2. rows来自命中cell所在行；
3. columns来自命中cell所在列+target_columns+constraint_columns；
4. 超预算时按确定性规则截断；
5. 不使用人工权重。
```

### 阶段6：构建table_zoom

```text
1. 实现build_cell_guided_table_zoom；
2. 输出refined_table_schema；
3. 保证后续CodeGenerator能直接使用；
4. 保存subtable_shape和compression_ratio。
```

### 阶段7：加入direct_cell_evidence baseline

```text
1. 支持zoom_mode=direct_cell_evidence；
2. 把top-k cells转成文本证据；
3. 不构建结构化子表；
4. 用于实验对照。
```

---

## 17.论文中的方法表述

不要写：

```text
本文基于TableZoomer进行了简单修改。
```

建议写：

```text
针对复杂表格问答中相关证据分布稀疏、原始问题语义不足以及大规模表格输入冗余严重的问题，本文提出一种基于问题重写与单元格检索的细粒度证据定位方法。该方法首先通过表感知问题重写显式抽取目标列、约束列、实体条件和操作意图；随后构建融合列名、单元格内容与行上下文的细粒度单元格表示，并基于无训练检索方法定位与问题相关的候选单元格；最后通过单元格证据驱动的结构化扩展，将离散候选单元格恢复为包含相关行列的证据子表，从而在保持可解释性的同时降低冗余表格信息对后续推理的干扰。
```

---

## 18.方法创新点表述

可以写成：

```text
1. 提出表感知问题重写机制，将自然语言问题转化为面向检索的结构化语义信号；
2. 提出融合行列上下文的单元格级证据表示方法，实现比列级选择更细粒度的证据定位；
3. 提出基于单元格证据的确定性结构化扩展策略，在不引入复杂权重和额外训练的情况下构建证据子表；
4. 在无训练条件下实现查询感知表格缩放，降低对标注数据、reranker训练和模型微调的依赖。
```

---

## 19.最终一句话总结

本工作在TableZoomer上的修改路线为：

```text
保留原始TableZoomer的schema生成、代码生成、代码执行和答案总结流程；
替换query-aware table zooming阶段；
将原来的列选择/实体链接式缩放改为：
问题重写 → 单元格检索 → 确定性行列扩展 → 证据子表构建。
```

最终主方法不是“加权行列聚合”，而是：

```text
top-k相关单元格
→ 保留命中行
→ 保留命中列
→ 补充目标列和约束列
→ 超预算时按确定性规则截断
→ 构建table_zoom
```

这样方法更简单、更稳定，也更容易在论文中解释。
