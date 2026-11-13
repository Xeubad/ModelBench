# ModelBench · 通用模型测试平台

[![tests](https://github.com/Xeubad/ModelBench/actions/workflows/tests.yml/badge.svg)](https://github.com/Xeubad/ModelBench/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 通用模型接口测试平台：直接连接云端 API、兼容接口或本地模型服务，评估延迟、吞吐、固定题集、工具调用、上下文召回和模型候选参考。
> 当前版本 **v2.11.0**。常规测试零依赖（纯 Python 标准库），本机运行，单页网页 + 命令行；可选参考库训练工具需要 NumPy。

ModelBench 使用单调时钟、连接复用、随机顺序和本机假服务自检，并保留原始事件。
结果主要描述被测接口在本次配置和网络条件下的表现；固定题集和重复请求不能直接代表模型总体能力。

```
添加接口 → 手工填写模型或按需探测 → 勾选模型与场景 → 开始 → 左侧配置、右侧看数据
每次测试存成独立文件夹 results/日期-次数/（result.json / raw.json / report.md）
```

---

## 致敬

本项目的**模型指纹（数值分布对账）**，思路来自两个优秀的开源项目，特此致谢：

- **[ModelTrace](https://github.com/xqy2006/ModelTrace)**（[@xqy2006](https://github.com/xqy2006)，MIT）
  用无语义数值生成挑战提取输出指纹，Hellinger 数字分布 + 有序块特征，投影掉环境干扰方向后做闭集归因。
- **[meow-llm-detector](https://github.com/chen-006/meow-llm-detector)**（[@chen-006](https://github.com/chen-006)，PolyForm Noncommercial 1.0.0）
  用短问题的回答分布做行为指纹，以 Dirichlet-multinomial 证据累积打分，并坚持"证据不足就拒判"。

ModelBench 按 MIT 许可改编了 ModelTrace 的评分方法与挑战，并附带其统一参考库快照；
版权、版本和许可全文见 [第三方声明](THIRD_PARTY_NOTICES.md)。meow 的非商用代码、题目和基准未包含在本项目中；
本项目自写的短问题分布场景仅供描述性比较。两项目的方法、数据来源和未实现功能见
[模型归因项目分析](docs/模型归因项目分析.md)。这些结果均不能认证模型身份。

---

## 特性

- **常规运行零依赖**：只用 Python 标准库（`http.client` + `http.server`），离线可用；自建参考库的可选训练命令需要 NumPy。
- **通用接入**：支持 OpenAI Chat Completions / Responses、Anthropic Messages、Gemini 原生、Azure OpenAI 部署式和 AWS Bedrock ConverseStream；兼容本地服务可不填密钥。
- **场景**：速度、生成吞吐、固定题库、多领域题集、两步工具调用、长上下文矩阵、短问题行为分布和 ModelTrace 候选参考。
- **Claude 协议**：支持 Anthropic Messages 模型分页、流式与非流式响应、工具调用及思考块回放；兼容服务仍须实现相应协议。
- **参考库与独立工具**：可采集、训练并按哈希导入 ModelTrace 参考库；可连接用户单独运行的 meow，不分发其非商用代码或基准。
- **客户端计时**：记录连接、响应头、首事件、首正文和结束时间；无法仅凭这些时间归因服务端排队或模型思考。
- **公平性控制**：连接复用、预热、每轮全模型×全题随机交替（种子可复现）、请求间隔。
- **负载与可靠性**：可选 1–4 个并发逻辑任务、每任务冷连接；汇总 10 秒内完成（有标准答案时须答对）的比例，失败计入分母。
- **描述性统计**：中位数、样本量、成功率和固定题集得分；不把相关样本的检验结果表述为显著或等效。
- **仪器自检**：本机假服务验证指定情形下的计时与状态处理，不代表真实网络均有同样误差。
- **数值分布比较**：Hellinger 距离与启发式阈值未经真实模型误报率校准，不能验证模型身份。
- **跨次合并**：只合并条件和接口身份已记录且一致的样本；更多请求不保证独立性或代表性。
- **本地存储**：结果和密钥存本地；测试提示词与请求会发给用户选择的接口。
- **局域网访问**：默认关闭。开启后可用手机/平板/另一台电脑打开，地址自带一次性访问令牌；令牌只在本机启动后有效，关闭或重启即失效。

---

## 快速开始

环境：Python 3.9+（macOS / Linux / Windows 均可），常规测试无需安装第三方包。

```bash
git clone https://github.com/Xeubad/ModelBench.git
cd ModelBench
cp config.example.json config.json   # 可选：不建也能跑（用内置默认 18888 端口）
./start.sh                            # 只启动服务，终端会打印地址
./start.sh --open                     # 启动并自动打开浏览器
```

也可跨平台直接运行 `python3 -m model_test serve --open`；Windows 可使用 `py -3 -m model_test serve --open`。

浏览器打开 <http://127.0.0.1:18888>（端口见 `config.json`；仓库里的示例用 18888，
本机若被占用可在 `config.json` 改）。

首次使用：

1. 左侧「模型接口」→ **＋添加**：填写名称和 Base URL，选择接口协议；需要鉴权的服务填写密钥，本地无鉴权服务可留空。
2. 每行手工填写一个模型 ID 后点「保存接口」；也可点「保存并探测」读取模型列表，或稍后点击「探测模型」。探测会保留已有模型并追加发现的模型；自动探测协议可能发送模型请求并产生费用。
3. 勾选想测的模型与场景，设置轮数，点**开始测试**。如果选了「自动探测」，须先完成协议探测；不支持模型列表的服务可手工填写模型并指定协议。

| 服务类型 | 接入方式 |
|---|---|
| OpenAI 或兼容服务 | 选择 Chat Completions 或 Responses，填写 Base URL（OpenAI 为 `https://api.openai.com/v1`）和所需密钥；可选 Bearer、`api-key`、`x-api-key` 或自定义鉴权头及前缀 |
| Anthropic 或兼容服务 | 选择 Anthropic Messages，填写 Base URL（Anthropic 为 `https://api.anthropic.com`）和所需密钥 |
| Google Gemini | 选择 Gemini 原生，使用 `https://generativelanguage.googleapis.com/v1beta` 与 API Key；支持模型分页、流式和工具回放 |
| Azure OpenAI | 部署式接口选 Azure，填写资源根地址、API 版本和部署名称；支持 `api-key` 或 Entra Bearer。Azure v1 可选 OpenAI 兼容协议并选择所需鉴权头 |
| AWS Bedrock | 选择 Bedrock，填写对应区域的 runtime 地址（例如 `https://bedrock-runtime.us-east-1.amazonaws.com`）、区域和可选 Profile；模型列表为支持文本流式的基础模型，推理配置 ID/ARN 可手工添加 |
| Ollama、LM Studio、vLLM 等本地服务 | 启用其 OpenAI 兼容接口并填写对应地址（例如本机 Ollama 的 `http://127.0.0.1:11434/v1`）；无需鉴权时密钥留空，通常选择 Chat Completions |

Bedrock 在真正探测或测试时读取启动进程的 `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、可选 `AWS_SESSION_TOKEN`；若未设置环境凭据，可读取 AWS 共享凭据文件中的静态 Profile（`AWS_SHARED_CREDENTIALS_FILE` 或默认 `~/.aws/credentials`）。显式填写 Profile 时使用该 Profile。平台不会把这些凭据写入接口配置或结果；非回环地址必须使用 HTTPS。SSO、角色链、容器/实例元数据和自动刷新暂未实现，需先自行导出临时环境凭据。更换启动环境的凭据后应重启平台。

OpenAI 兼容或 Anthropic 接口可在「高级：非密钥查询参数」逐行填写 `name=value`；这些参数会用于模型列表和请求，也会进入配置与环境快照。密钥必须放在密钥栏，不能放在地址或查询参数中；分页与协议保留字段不可覆盖。自定义鉴权头的值由所填前缀和密钥组成，连接管理等保留头不可覆盖。

兼容程度取决于服务和模型实际支持的接口、流式、工具和参数。自动协议探测只尝试 OpenAI / Anthropic；Gemini、Azure 和 Bedrock 需手动选择协议。Azure 部署名需要手工填写，Entra Token 需自行刷新；Gemini/Bedrock 暂不支持页面的统一推理强度参数，请设为「不指定」。其他原生协议、自动 OAuth 刷新和需要将密钥放在查询字符串中的鉴权尚未支持。平台不要求安装 OpenClaw。

协议依据：[Gemini GenerateContent 定义](https://github.com/googleapis/googleapis/blob/master/google/ai/generativelanguage/v1beta/generative_service.proto)、[Gemini 内容与工具定义](https://github.com/googleapis/googleapis/blob/master/google/ai/generativelanguage/v1beta/content.proto)、[Azure REST 接口](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)、[Bedrock ConverseStream](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html)、[AWS SigV4](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-create-signed-request.html) 和 [Amazon Event Stream](https://smithy.io/2.0/aws/amazon-eventstream.html)。

已有 OpenClaw 数据的用户可在「可选：从其他工具迁移」中主动导入供应商；同名接口会更新。旧版历史目录仅用于兼容已有结果，路径不存在时不会影响运行。

> 如有密钥，保存在 `data/endpoints.json`（权限 600，已 `.gitignore`），页面只显示末四位；编辑时留空保留原密钥，可显式选择清除。

### 命令行（不开网页）

```bash
python3 -m model_test run --endpoint local-service --model model-id --scenario quiz --scenario tool --rounds 3
python3 -m model_test selftest     # 仪器自检（19 项，约 30 秒）
python3 -m model_test test         # 单元测试与本机集成测试
python3 -m model_test help
```

---

## 测试场景

| 场景 | 测什么 | 判分方式 |
|---|---|---|
| 速度 | 自定义提示词的首字与总耗时 | 不判分（看时间） |
| 生成速度 | 固定约 300 字的生成吞吐 | 不判分（看 tokens/s） |
| 题库 | 10 道有标准答案的题 | 精确匹配 / 正则 / JSON 比对，不让模型自评 |
| 多领域题集 | 8 道计算、抽取、语言和结构化输出题 | 固定题集描述性得分，不代表总体能力 |
| 工具调用 | 两步：先调工具，再把随机价格回传 | 校验工具参数及只含价格和单位的 JSON |
| 大海捞针 | 约 1 万 / 3 万 tokens 背景里藏暗号 | 暗号是否答出 |
| 长上下文矩阵 | 约 1 万 / 3 万 tokens、3 个位置、有/无答案 | 严格匹配目标编号或“未知”；12 次请求/轮/模型，费用较高 |
| 模型指纹 | 输出无语义随机整数的数值分布 | Hellinger 距离 + 未经真实模型校准的启发式阈值 |
| ModelTrace 候选参考 | 每轮 3 次长数字挑战，可手动粘贴回答 | 在所选参考库中做闭集排序；3 份有效回答不得重复或呈明显规律，保守规则未经误报率校准 |
| ModelTrace 参考库采集 | 每模型 36 次，覆盖 12 种环境 | 导出 JSONL，保留采集版本、实际回答与有效性，可离线训练参考库 |
| 短问题行为分布 | 每轮 4 个短问题，比较各答案频数 | 达到展示门槛后显示 Jensen–Shannon 散度；无身份判定 |

---

## 自建 ModelTrace 参考库

1. 在页面选择「ModelTrace 参考库采集」，轮数设为 1、推理强度设为不指定；选择至少两个**已知来源**的模型进行训练数据采集。每模型 36 次请求，预热另计；结果页可下载参考样本 JSONL，失败及规律序列保留但不会用于训练。
2. 要评估已知模型与库外模型，请分别再做独立采集并导出 JSONL 作为留出集。训练器检查运行 ID 不重叠以及固定的挑战文本、环境与版本，但无法验证接口来源是否真实，也不能证明两次运行在统计上独立。
3. 在单独虚拟环境中安装可选依赖并训练（示例路径需要替换为自己的下载文件；输出文件不会覆盖已有文件）：

```bash
python3 -m venv .venv-reference
source .venv-reference/bin/activate  # Windows 使用对应的 Scripts/activate
python -m pip install 'numpy>=1.26,<3'
mkdir -p data/modeltrace-references
python -m model_test train-bank --train train-a.jsonl --train train-b.jsonl \
  --holdout holdout-a.jsonl --holdout holdout-b.jsonl --holdout unknown.jsonl \
  --output data/modeltrace-references/bank-v1.json
```

`--holdout` 可省略，但结果就没有独立留出评估。两模型训练加两模型留出和一个未知模型留出，至少需 180 次请求，预热另计；实际费用取决于模型和长提示词。训练结果记录样本哈希、运行 ID、上游挑战版本和留出统计；已知模型报告排序正确数，未知模型报告被排序/拒判数，**不声称能检测未知模型**。留出统计使用同一套 36 条采集挑战，不等同于对自动三挑战场景的真实误报率校准；软件无法核实用户提供的标签或来源。

4. 到「手动归因」导入训练出的 JSON；选择该库后，手动分析和自动 ModelTrace 场景都会使用它。库按内容哈希保存，不覆盖内置库，历史结果保留当时的哈希和排名。导入库只做结构校验，不验证样本真实性、授权或统计有效性。

## 连接独立 meow

自行按 [meow 官方说明](https://github.com/chen-006/meow-llm-detector)安装并启动，在 meow 内保存带凭据的接口并安装相应基准。ModelBench 的「meow · 独立」页面连接运行 ModelBench 那台机器的 `127.0.0.1` 指定端口，选择 meow 自己的接口、基准和模型，先估算请求数再由用户启动；默认额外重试预算为 0。完成后可读取报告摘要，完整报告与基准由 meow 保存。

meow 及其基准采用 **PolyForm Noncommercial 1.0.0**，与 ModelBench 的 MIT 许可独立；请在使用前核对上游许可。本项目不分发上游 meow 代码或数据，不自动安装，也不将 ModelBench 保存的接口密钥交给 meow。meow 的判定和匹配分数仍不是身份认证。已用 meow 4.5.4 和本机假模型验证适配，未来上游版本可能改变接口。

---

## 测量方法与边界

以下措施有助于追溯和解释结果，但不会消除网络、缓存、采样或服务端实现差异：

| 做法 | 说明 |
|---|---|
| 单调时钟 | 全部耗时用 `time.perf_counter()`，不受系统校时影响 |
| 握手单列 | TCP/TLS 建连耗时单独记录，**不计入请求耗时** |
| 连接复用 + 预热 | 每主机一条复用连接；每个模型先发一次不计入的预热请求 |
| 每轮随机交替 | 每轮把"全部模型 × 全部题目"打乱再发，种子保存在结果里可复现，避免时段偏差 |
| 请求间隔 | 串行时按逻辑任务结束计间隔，并发时按逻辑任务开始计间隔；工具第二步不受间隔控制，不能保证避免限流 |
| 连接中断 | POST 发出后如果无法确认上游是否收到，记录“送达状态未知”，不自动重发；历史结果可能仍包含旧版重试记录 |
| 采样参数 | 普通场景显式传 `temperature=0` 与输出上限；ModelTrace 和短问题分布场景使用服务商默认 temperature，并要求推理强度不指定。实际请求参数会记录，服务端可能不支持或不遵循 |
| 输出长度 | 展示提供方报告的输出 tokens 中位数；不同 tokenizer 与固定等待会影响跨模型比较 |
| 缓存提示 | 按首末成功请求的耗时和提供方缓存用量提示可疑趋势；不是可靠的缓存检测器 |
| 响应头后等待 | `queue_s` 是响应头到首事件的客户端时间差，无法单独归因于排队、网络或模型 |
| 非流式识别 | 中转站攒满后一次返回时，标记"首字不可信"，改用整体速度 |
| 小样本拒判 | 样本 < 5 标"样本不足"；p95 只在 ≥ 20 样本时显示，否则显示最大值 |
| 比较与区间 | 当前比较只展示描述性差值，不作显著性或等效推断；正确率不显示未经验证的 Wilson 区间；成功率 Wilson 区间仅供参考，重复请求可能相关 |
| 跨次合并 | 场景摘要、方法版本、请求条件与接口地址/协议须已记录并一致；随机工具/矩阵要求相同种子；缺少字段的旧结果分开显示 |
| 仪器自检 | 本机假服务设定 0.5s / 2s 延迟与 503/504/429 等故障，验证特定场景的实现 |

**仍然存在的局限**（别过度解读）：

- 并发 1–4 为有限并发功能，不是持续负载或压力测试；并发模式各工作线程分别复用连接，冷连接模式每逻辑任务重建连接，工具两步内部仍可复用。
- 10 秒限时指标从逻辑任务开始计时，不含调度等待；数值仅适用于该场景，不能与其他场景无条件比较。
- 同一固定文本会被上游前缀缓存命中，后几轮可能偏快（已能识别并警示）。
- 没有通用方法可强制上游关闭缓存；展示的缓存命中与模型名由接口报告，不构成独立验证。
- 不同接口网络路径不同，握手耗时已单列但不能消除。
- 固定题库和多领域题集合计只有 18 题、没有独立留出集，只能作参考。
- 模型指纹阈值未经真实模型误报率校准；数值分布接近或不同都不能证明模型身份。
- ModelTrace 权重只在 18 个已收录候选之间分配，未知模型也会被归到已有候选；Claude 样本来自第三方接口，不能当官方身份认证。
- 新增 Gemini、Azure、Bedrock 协议只用本机假服务及 AWS SDK 签名固定向量验证，尚未用真实云接口验证；归因功能也未测得真实误报率。
- v2.7 修改了工具判分和测量条件；不同方法版本不自动合并，旧历史报告不会被重写。

---

## 数据与安全

- 普通接口的 API Key / Token 存于 `data/endpoints.json`（权限 600）；Bedrock 凭据从用户环境或共享凭据文件读取，不存入本平台配置。对于上游回显的已知密钥原文和 URL 编码，错误与响应会进行脱敏；流片段可能重组出密钥时不保存事件内容，该请求不计为成功。短密钥采用独立词匹配，其他变形无法保证全部识别；旧结果不会重写。
- 结果只存 `results/`，含回答全文与原始流事件，写完设为只读。
- `.gitignore` 已排除接口密钥、用户参考库、结果、`adhoc/`、`.venv-reference/` 和 `config.json`（本机监听地址与路径）；自行导出到其他路径的文件由用户管理。
- 服务默认只监听 `127.0.0.1`；如需固定绑定其他地址，配 `server.bind_hosts`。
- 页面检查 Host 与 Origin、POST 强制 JSON、参数白名单校验，防止别的网站借浏览器触发测试。
- 页面本身没有账号登录，靠下面的「局域网访问」令牌限制非本机访问。

---

## 局域网访问

默认**关闭**。左侧「局域网访问」勾选后，会在本机检测到的局域网地址上额外开一个监听
（不会绑 `0.0.0.0`，只绑具体网卡地址），并给页面一个**带访问令牌**的地址：

```
http://192.168.x.x:18889/?k=<一次性令牌>
```

- 在手机 / 平板 / 另一台电脑上打开这个地址即可；首次用 `?k=` 打开后会种 cookie，后续同源请求不用再带参数。
- 本机（`127.0.0.1`）访问不需要令牌。
- 不带令牌或令牌错误的局域网请求会看到一页说明，而不是数据。
- 令牌**只在本次运行有效**：关闭开关或重启服务即失效，重新开启会换新令牌。
- 令牌不会回传给已连接的局域网设备，避免设备之间互相串联。

**风险提示**：页面没有登录，拿到这个地址的人在其有效期内可以使用已保存的接口密钥，以及运行平台的本机 AWS 环境或 Profile 凭据发请求。
只在可信网络（家里、自己的热点）开启，用完关掉。开关状态不写入配置，重启后默认关闭。

---

## 目录结构

```
ModelBench/
├── start.sh                   启动网页（等同 python3 -m model_test serve）
├── config.example.json        配置模板（复制成 config.json 用；config.json 不入库）
├── model_test/
│   ├── __main__.py            统一入口：serve / run / train-bank / selftest / test
│   ├── config.py              读取并校验 config.json
│   ├── core.py                测试引擎：连接池、流式请求、计时、场景、判分、统计
│   ├── gemini.py              Gemini 原生请求、分页与流事件
│   ├── azure.py               Azure 部署路径和版本参数
│   ├── bedrock.py             AWS SigV4 与 Bedrock 二进制事件流
│   ├── custom_http.py         自定义鉴权头和非密钥查询参数
│   ├── modeltrace.py          ModelTrace 候选评分与长数字挑战
│   ├── modeltrace_suite.py    36 条上游采集挑战（MIT）
│   ├── modeltrace_training.py 可选 NumPy 训练和留出评估
│   ├── banks.py               用户参考库的校验与哈希存储
│   ├── meow_bridge.py         独立 meow 本机进程接口
│   ├── behavior.py            短问题行为分布和描述性距离
│   ├── assets/                固定版本的上游 MIT 参考库
│   ├── store.py               存储：接口增删改与探测、结果目录、历史、合并
│   ├── server.py              网页服务（标准库 http.server）
│   ├── selftest.py            本机假服务 + 仪器自检
│   └── web/dashboard.html     单页面（原生 JS + SVG，离线可用）
└── tests/                     单元、协议与本机假服务测试
```

`data/`（接口与密钥）与 `results/`（测试结果）由程序运行时创建，不在仓库内。

---

## 配置

端口、监听地址、限额、页面默认值、数据与结果目录都在 `config.json`。
只写想改的项，其余沿用内置默认；写错项名或数值超范围会在启动时直接报错。

> `config.json` 是**本机配置，不入仓库**（含监听地址与本机路径），模板见 `config.example.json`。
> 没有 `config.json` 时程序用内置默认值，可直接运行。

| 项 | 默认 | 说明 |
|---|---|---|
| `server.port` | 18888 | 网页端口，`--port` 可临时覆盖 |
| `server.bind_hosts` | `["127.0.0.1"]` | 只绑本机。需要局域网/组网访问时自行追加地址（绑定失败会跳过） |
| `limits.max_rounds` | 20 | 轮数上限 |
| `limits.max_requests` | 1200 | 一次测试最多请求数 |
| `limits.max_prompt` | 2000 | 自定义提示词字数上限 |
| `defaults.rounds` | 5 | 页面首次打开时的轮数 |

测量方法本身的常量（样本不足阈值、p95 最少样本数、非流式判定、指纹判定线等）
**不在配置里**——它们是方法的一部分，改了会让历史结果不可比，见 `model_test/core.py`。

---

## 免责声明

- 本项目是个人测量工具，**结论仅供参考**，不构成对任何服务商行为的判定。
- 测试费用由使用者自己的 API 账户承担。
- 数据取自被测接口的返回，平台不伪造、不推断未收到的内容。
- 数值分布距离与启发式阈值不能认证模型身份，也不能单独证明上游行为。

## 许可

[MIT](LICENSE) © 2026 Xeubad

可自由使用、修改、分发、商用，保留版权声明与许可声明即可。软件按「原样」提供，不附带任何担保。
