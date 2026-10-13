# 演示指南：会记事的导购助手（约 3 分钟）

[English](README.en.md) · 简体中文

演示的是 N8 的电商导购助手 Cue：一家虚构的运动用品店 CueMart（24 件商品，图片由 qwen-image-3.0 生成）。用户 Maya 在 8 月和 9 月各聊过一次；演示时是 10 月 10 日的第三次对话，现场展示：

- 跨会话回忆，以及知识更新（跑鞋改穿 40、平时仍是 39；预算 $120 → $180）；
- 晒图找同款：八月分享的照片从记忆里召回，原图再交给模型看；
- 每一轮的记忆轨迹：控制器为什么这样选、上下文里放了什么、回答引用了哪些记忆；
- 遗忘：删掉一条事实，连同复述过它的回复一起删（引用过它的，以及没有引用、只是顺口转述的）。

![演示动图](../docs/assets/demo.gif)

界面与对话是英文的。

## 准备（一次）

```bash
uv sync
uv run cuemem configure          # 填 QwenCloud 的 Key（演示只用通义，不用 DeepSeek）
uv run cuemem demo demo/shop/script.json --fresh --until 2
```

最后一步清空记忆空间 `shop-demo`，按剧本回放前两段会话（8 月 28 日、9 月 12 日），每段结束后同步巩固。第一次约 2 分钟、约 $0.02；之后命中缓存，约 1 分钟、不花钱。**每次录屏前都重跑一遍**，把上一次现场对话留下的记忆清掉。

```bash
uv run cuemem serve              # 打开 http://127.0.0.1:7860/
```

## 现场步骤

| 时间 | 操作 | 讲什么 |
|---|---|---|
| 0:00 | 打开界面；"Session date" 填 `2026-10-10 10:00`，点 New session | 记忆空间里已经有 8 月、9 月两次对话；时间是模拟的 |
| 0:15 | 发 `Hi Cue, I'm back!` | 叫得出名字，记得马拉松计划和看中的鞋；右边的轨迹：控制器的选择、画像里被引用的条目 |
| 0:40 | 发 `What size running shoes do I wear now?` | 答 EU 40，并说明九月起跑鞋换成了 40。记忆里两条事实都是"当前"：N11 起，对账把"跑鞋穿 40"看作比"穿 39 码"更窄的范围，作为新增，而不是取代平时的尺码（D-053）。"被取代"的例子见 1:40 那一步的预算 |
| 1:05 | 发 `Can you find shoes that look like the pair I showed you in August?` | 轨迹里有八月那张照片：从记忆里召回、原图交给模型（"original photo shown to the model"），再按图在商品目录里找同款；回答下方是商品图 |
| 1:40 | 发 `How has my shoe budget changed since we first talked?` | $120 → $180。如果控制器选了 L0，轨迹里会看到模型自己调用了 memory_search 补查——"控制器先选、模型兜底"；也可以在 Memory controller settings 里把 Recall 切到 l2，看深度检索的规划与补检索 |
| 2:10 | 切到 Memory 页 | 画像；事实时间线里被取代的预算，以及取代它的新事实；实体图；图库 |
| 2:30 | Forget a → fact → 选 "Ana turns 30 on October 3…" → Forget | 删的是这条事实出自的那一轮对话、后来引用过它的回复，以及没有引用、只是顺口转述它的轮（九月那次回复里的"她 10 月 3 日生日，早点下单"，或今天问候时提到的生日，D-051），提示里会写 "including N that restated it without citing"；事实与画像按剩下的日志重建 |
| 2:50 | 回到 Chat，发 `When is my sister's birthday?` | 答"不知道"。屏幕上的聊天记录还在，但 Cue 的上下文和记忆里已经没有了 |

可以顺带讲的：

- 轨迹面板里各档的"expected accuracy"与扣掉成本之后的分数；λ 旋钮越大越省；
- 上下文预算条：系统提示、画像、本次对话的原话是稳定前缀（供应商的前缀缓存一直命中），召回的记忆每轮重算、只放在最后；Usage 一行有多少输入来自前缀缓存；
- 控制器在这个领域的两个偏差：有照片时几乎总是附图；"预算怎么变的"可能被判为与记忆无关（[N8 结果](../docs/results/n8-agent/README.md)）。

## 回答会不一样

模型的回答每次略有不同，控制器的选择也会随记忆内容变化。剧本的自动核对只看关键内容：

```bash
uv run cuemem demo demo/shop/script.json --fresh   # 三段会话加遗忘，14 项检查
```

整段回放以遗忘结束（遗忘会清掉受影响会话的回答缓存；N11 起回答缓存按会话归属，别的会话不受影响，D-054），跑完记得再执行一次 `--until 2` 恢复演示前的状态。

重录上面的动图：在刚建好的演示记忆上开着 `cuemem serve`，运行 `uv run --with playwright python scripts/record_demo.py`（用本机的 Google Chrome）。

## 文件

| 文件 | 内容 |
|---|---|
| `shop/catalog.json` | 商品目录（名称、价格、颜色、材质、尺码、标签、描述）与生成图片用的提示词 |
| `shop/images/` | 商品图（`scripts/n8_assets.py` 生成，最长边 768 像素） |
| `shop/photos/lena-shoes.jpg` | "朋友的鞋"那张照片（同样是生成的） |
| `shop/script.json` | 剧本：三段会话、每条消息的检查 |
| `video/cuemem-60s.html` | 60 秒的无声介绍动画（GSAP、SVG、Canvas、CSS），字幕直接在画面里。用浏览器打开即可播放；`uv run --with playwright --with imageio-ffmpeg python scripts/render_video.py` 渲染成 1080p MP4，输出到 `runs/video/` |
