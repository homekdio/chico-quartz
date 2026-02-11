# RAG 代码逐行超详细讲解 (零基础小白版)

这份文档专门为您准备，将 [rag_lora_cloud_improverd.ipynb](file:///d:/Program%20Files/develop/llm_related-main/rag_demo/rag_lora_cloud_improverd.ipynb) 笔记本中的代码拆解开来，用最通俗易懂的语言解释它的**作用**、**意义**以及**为什么要这么写**。

---

## 第一部分：导入工具包 (Imports)

就像做饭前要准备锅碗瓢盆和食材一样，写代码前我们需要导入各种“工具包”（库）。

```python
import ast          # 就像一个“翻译机”，能把长得像字典的字符串（"{'a':1}"）安全地变成真正的字典对象。
import os           # 操作系统的“管家”，帮我们找文件、设置环境变量（比如 API Key）。
import pickle       # Python 的“冷冻机”，能把训练好的模型或对象保存成文件，以后直接解冻读取，不用重新训练。
import hashlib      # “指纹生成器”，给每一段文本生成一个独一无二的乱码 ID，用来给文章办身份证。
import torch        # 深度学习的“发动机” (PyTorch)，所有的 AI 模型计算都靠它。
import jieba        # 中文的“切菜刀”，把“我爱北京”切成“我 / 爱 / 北京”，方便搜索引擎理解。
from typing import List  # 给代码加“注释标签”，告诉程序员这个变量是个列表，方便阅读，机器其实不看它。
import langchain_community  # LangChain 是 AI 应用开发的“百宝箱”，这里用了它的社区扩展包。
import rank_bm25    # 一个经典的“找关键词”算法库，就像图书馆的索引卡片。
from langchain_community.embeddings import DashScopeEmbeddings # 阿里通义千问的“翻译官”，把文字变成一串数字（向量）。
from langchain_community.vectorstores import Chroma  # 一个“存数字的仓库”（向量数据库），专门用来存上面翻译出来的数字向量。
from langchain_core.documents import Document  # LangChain 定义的“文件档案袋”，里面装文字内容和元数据。
from modelscope import AutoModelForCausalLM, AutoTokenizer # ModelScope（魔搭）的“搬运工”，帮我们下载和加载模型。
from transformers import BitsAndBytesConfig # 一个“压缩工具”，能把大模型变小（量化），让普通显卡也能跑得动。
import dashscope    # 阿里云官方的“电话机”，用来给云端的大模型发请求。
from dashscope import Generation # 电话机里的“生成部门”，专门负责让 AI 说话。
from http import HTTPStatus # 网络请求的“红绿灯”，用来检查请求是不是成功了（200 代表绿灯通过）。
```

---

## 第二部分：配置中心 (Configuration)

这里是整个程序的“控制台”。我们把所有可能需要调整的参数都写在这里，以后想改哪个（比如换个模型、改个参数），只来这里改就行，不用满世界找代码。

```python
class Config:
    """配置类"""
    # 模型文件在哪个文件夹里
    model_path = "Qwen2-7B-Instruct-Lora" 
    
    # 向量数据库存在哪里（Chroma 文件夹）
    persist_directory = "Chroma"
    
    # BM25 索引文件叫什么
    bm25_path = "bm25_retriever.pkl"
    
    # 你的阿里云 API 密钥，相当于“通行证”
    api_key = "sk-2797..."
    
    # 使用显卡 (cuda) 进行计算
    device = "cuda"
    
    # --- RAG 检索参数 ---
    k_retrieval = 10  # 初选：通过关键词或语义，先从海量数据里捞出 10 条大概相关的。
    rrf_k = 10        # 决选：综合两种搜索结果后，最终保留前 10 条给 AI 看。
    m_rrf = 60        # RRF 算法的一个调节参数（平滑常数），60 是个经验值，不用深究。
    
    # --- 模型生成参数 ---
    max_new_tokens = 2048 # AI 最多能啰嗦多少个字。
    temperature = 0.1     # “创造力温度”：
                          # 0.1 = 严谨、保守、每次回答差不多（适合做医生、律师）。
                          # 0.9 = 奔放、随机、充满想象力（适合写小说）。
    top_p = 0.9           # 另一种控制多样性的参数(“核采样”)，通常配合温度一起用。

# 动作：初始化配置，并把 Key 告诉操作系统，这样后面的工具包能自动读取到。
config = Config()
os.environ["DASHSCOPE_API_KEY"] = config.api_key
```

---

## 第三部分：加载检索器 (Loading Retrievers)

这一步是把我们的“知识库”加载进内存。知识库分两路：
1.  **BM25 (关键词)**：硬搜索，比如搜“骨折”必须要有这两个字。
2.  **Chroma (语义)**：软搜索，搜“骨折”它能懂“腿断了”也是相关的。

```python
def preprocessing_func(text):
    return list(jieba.cut(text)) # 这是一个小助手，帮 BM25 把句子切成词。

def load_retrievers(config: Config):
    retrievers = {} # 准备一个空袋子装检索器
    
    # --- 加载 BM25 ---
    # 打开 .pkl 文件，用 pickle 复活成对象
    if os.path.exists(config.bm25_path):
        with open(config.bm25_path, 'rb') as f:
            retrievers['bm25'] = pickle.load(f)
            
    # --- 加载 Chroma ---
    if os.path.exists(config.persist_directory):
        # 准备嵌入模型（把字变数字的工具）
        embeddings = DashScopeEmbeddings(model="text-embedding-v4")
        
        # 连接到本地数据库文件夹
        vectorstore = Chroma(
            persist_directory=config.persist_directory,
            embedding_function=embeddings,
            collection_name="my_collection"
        )
        # 把数据库变成一个“检索器”，设定好每次只查前 k 个
        retrievers['chroma'] = vectorstore.as_retriever(search_kwargs={"k": config.k_retrieval})
        
    return retrievers # 把装好的袋子返回去
```

---

## 第四部分：RRF 重排序 (The Brain)

**RRF (Reciprocal Rank Fusion)** 是一个裁判算法。
BM25 说文章 A 排第一，Chroma 说文章 A 排第十。谁对？
RRF 说：别吵了，我用公式 `1 / (排名 + 60)` 给你们算个综合分。

```python
def rrf(vector_results, bm25_results, k=10, m=60):
    doc_scores = {} # 记分本
    
    # 给每个文档算个身份证号 (ID)，防止同一篇文章被加两次
    def get_doc_id(doc):
        # 内容加元数据拼起来算哈希，内容一样 ID 就一样
        combined = doc.page_content + str(sorted(doc.metadata.items()))
        return hashlib.md5(combined.encode("utf-8")).hexdigest()

    # --- 开始打分 ---
    # 遍历向量检索结果
    for rank, doc in enumerate(vector_results):
        doc_id = get_doc_id(doc)
        # 排名越靠前 (rank 小)，得分越高
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank + m)
    
    # 遍历 BM25 文档结果
    for rank, doc in enumerate(bm25_results):
        doc_id = get_doc_id(doc)
        # 如果这个文档之前已经有分了（说明两种方法都觉得它好），就把分加起来！
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank + m)
    
    # --- 选出前 K 名 ---
    # 按分数从高到低排序，切片取前 k 个
    sorted_ids = [d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]]
    
    return [doc_map[doc_id] for doc_id in sorted_ids]
```

---

## 第五部分：加载模型 (Load Model)

这是把 AI 的“大脑”加载到你的显卡里。

```python
def load_model(config: Config):
    # --- 清理显存 ---
    # 先把之前可能占着茅坑不拉屎的旧模型删掉
    try:
        import gc
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass
    
    # --- 量化配置 (核心) ---
    # 让大模型“瘦身”。把 16-bit 的参数压缩成 4-bit。
    # 就像把高清图压缩成 JPG，虽然细节丢了一点点，但体积只有原来的 1/4，
    # 这样你的显卡才能装得下。
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True
    )
    
    # --- 加载主模型 ---
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=quantization_config, # 应用这一套瘦身方案
        device_map="auto",      # 自动决定放哪张显卡
        trust_remote_code=True  # 允许运行模型自带的代码
    )
    
    # --- 加载分词器 ---
    # 分词器是翻译官，把 "你好" 翻译成 [1092, 230] 这种数字给模型看
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    
    return model, tokenizer
```

---

## 第六部分：核心功能 (Core Features)

为了做对比实验，我们将功能拆开。

### 1. 找资料 (`get_relevant_context`)
它的任务是去两个检索器里找资料，然后用 RRF 排序，最后清洗数据（因为有时候找出来的是 JSON 格式的乱码，要提取出纯文本）。

### 2. 写提示 (`build_prompt`)
这是我们在教 AI 怎么回答。
*   **有资料时 (RAG)**：我们给它一个模板：“你是专家...参考资料是...请分点回答...问题是...”。
*   **没资料时 (Direct)**：直接问：“你是专家...问题是...”。

### 3. 生成回答
*   **`generate_local`**：用显卡里的模型。流程是：文字 -> 转数字 (`input_ids`) -> 模型计算 -> 出数字结果 (`generated_ids`) -> 转回文字 (`decode`)。
*   **`generate_api`**：直接发 HTTP 请求给阿里云，等它回话。

### 4. 实验总管 (`run_experiment`)
这是个“调度员”。
*   如果你选 `mode="local_rag"`：它就先调**找资料**函数，再调**写提示**函数，最后调**本地生成**函数。
*   如果你选 `mode="api_direct"`：它就跳过找资料，直接调**写提示**函数，然后调**API生成**函数。

---

## 第七部分：最后一步 (Execution)

这里就是按下“启动键”的地方。

```python
# 1. 如果没加载过，就加载一次。用 if 判断是为了防止你重复运行这个单元格，导致重复加载模型把显存撑爆。
if 'retrievers' not in locals():
    retrievers = load_retrievers(config)
if 'model' not in locals():
    model, tokenizer = load_model(config)

# 2. 只有这个问题
query = "骨折了应该怎么办"

# 3. 跑四场比赛，看看谁厉害
# 🟢 实验 1: 你的微调小模型 + 知识库
run_experiment(..., mode="local_rag")

# 🔴 实验 2: 你的微调小模型 (裸考)
run_experiment(..., mode="local_direct")

# 🔵 实验 3: 阿里云超大模型 + 知识库 (检索决定上限)
run_experiment(..., mode="api_rag", api_model_name="qwen-turbo")

# 🟣 实验 4: 阿里云超大模型 (裸考)
run_experiment(..., mode="api_direct", api_model_name="qwen-turbo")
```