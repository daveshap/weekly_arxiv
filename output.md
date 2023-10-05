# LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving

[Link to the paper](http://arxiv.org/abs/2310.03026v1)

## Authors
- Hao Sha
- Yao Mu
- Yuxuan Jiang
- Li Chen
- Chenfeng Xu
- Ping Luo
- Shengbo Eben Li
- Masayoshi Tomizuka
- Wei Zhan
- Mingyu Ding

## Summary
  Existing learning-based autonomous driving (AD) systems face challenges in
comprehending high-level information, generalizing to rare events, and
providing interpretability. To address these problems, this work employs Large
Language Models (LLMs) as a decision-making component for complex AD scenarios
that require human commonsense understanding. We devise cognitive pathways to
enable comprehensive reasoning with LLMs, and develop algorithms for
translating LLM decisions into actionable driving commands. Through this
approach, LLM decisions are seamlessly integrated with low-level controllers by
guided parameter matrix adaptation. Extensive experiments demonstrate that our
proposed method not only consistently surpasses baseline approaches in
single-vehicle tasks, but also helps handle complex driving behaviors even
multi-vehicle coordination, thanks to the commonsense reasoning capabilities of
LLMs. This paper presents an initial step toward leveraging LLMs as effective
decision-makers for intricate AD scenarios in terms of safety, efficiency,
generalizability, and interoperability. We aspire for it to serve as
inspiration for future research in this field. Project page:
https://sites.google.com/view/llm-mpc


# Retrieval meets Long Context Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.03025v1)

## Authors
- Peng Xu
- Wei Ping
- Xianchao Wu
- Lawrence McAfee
- Chen Zhu
- Zihan Liu
- Sandeep Subramanian
- Evelina Bakhturina
- Mohammad Shoeybi
- Bryan Catanzaro

## Summary
  Extending the context window of large language models (LLMs) is getting
popular recently, while the solution of augmenting LLMs with retrieval has
existed for years. The natural questions are: i) Retrieval-augmentation versus
long context window, which one is better for downstream tasks? ii) Can both
methods be combined to get the best of both worlds? In this work, we answer
these questions by studying both solutions using two state-of-the-art
pretrained LLMs, i.e., a proprietary 43B GPT and LLaMA2-70B. Perhaps
surprisingly, we find that LLM with 4K context window using simple
retrieval-augmentation at generation can achieve comparable performance to
finetuned LLM with 16K context window via positional interpolation on long
context tasks, while taking much less computation. More importantly, we
demonstrate that retrieval can significantly improve the performance of LLMs
regardless of their extended context window sizes. Our best model,
retrieval-augmented LLaMA2-70B with 32K context window, outperforms
GPT-3.5-turbo-16k and Davinci003 in terms of average score on seven long
context tasks including question answering and query-based summarization. It
also outperforms its non-retrieval LLaMA2-70B-32k baseline by a margin, while
being much faster at generation. Our study provides general insights on the
choice of retrieval-augmentation versus long context extension of LLM for
practitioners.


# Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions

[Link to the paper](http://arxiv.org/abs/2310.03016v1)

## Authors
- Satwik Bhattamishra
- Arkil Patel
- Phil Blunsom
- Varun Kanade

## Summary
  In order to understand the in-context learning phenomenon, recent works have
adopted a stylized experimental framework and demonstrated that Transformers
can learn gradient-based learning algorithms for various classes of real-valued
functions. However, the limitations of Transformers in implementing learning
algorithms, and their ability to learn other forms of algorithms are not well
understood. Additionally, the degree to which these capabilities are confined
to attention-based models is unclear. Furthermore, it remains to be seen
whether the insights derived from these stylized settings can be extrapolated
to pretrained Large Language Models (LLMs). In this work, we take a step
towards answering these questions by demonstrating the following: (a) On a
test-bed with a variety of Boolean function classes, we find that Transformers
can nearly match the optimal learning algorithm for 'simpler' tasks, while
their performance deteriorates on more 'complex' tasks. Additionally, we find
that certain attention-free models perform (almost) identically to Transformers
on a range of tasks. (b) When provided a teaching sequence, i.e. a set of
examples that uniquely identifies a function in a class, we show that
Transformers learn more sample-efficiently. Interestingly, our results show
that Transformers can learn to implement two distinct algorithms to solve a
single task, and can adaptively select the more sample-efficient algorithm
depending on the sequence of in-context examples. (c) Lastly, we show that
extant LLMs, e.g. LLaMA-2, GPT-4, can compete with nearest-neighbor baselines
on prediction tasks that are guaranteed to not be in their training set.


# LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples

[Link to the paper](http://arxiv.org/abs/2310.01469v2)

## Authors
- Jia-Yu Yao
- Kun-Peng Ning
- Zhen-Hui Liu
- Mu-Nan Ning
- Li Yuan

## Summary
  Large Language Models (LLMs), including GPT-3.5, LLaMA, and PaLM, seem to be
knowledgeable and able to adapt to many tasks. However, we still can not
completely trust their answer, since LLMs suffer from
hallucination--fabricating non-existent facts to cheat users without
perception. And the reasons for their existence and pervasiveness remain
unclear. In this paper, we demonstrate that non-sense prompts composed of
random tokens can also elicit the LLMs to respond with hallucinations. This
phenomenon forces us to revisit that hallucination may be another view of
adversarial examples, and it shares similar features with conventional
adversarial examples as the basic feature of LLMs. Therefore, we formalize an
automatic hallucination triggering method as the hallucination attack in an
adversarial way. Finally, we explore basic feature of attacked adversarial
prompts and propose a simple yet effective defense strategy. Our code is
released on GitHub.


# From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference

[Link to the paper](http://arxiv.org/abs/2310.03003v1)

## Authors
- Siddharth Samsi
- Dan Zhao
- Joseph McDonald
- Baolin Li
- Adam Michaleas
- Michael Jones
- William Bergeron
- Jeremy Kepner
- Devesh Tiwari
- Vijay Gadepally

## Summary
  Large language models (LLMs) have exploded in popularity due to their new
generative capabilities that go far beyond prior state-of-the-art. These
technologies are increasingly being leveraged in various domains such as law,
finance, and medicine. However, these models carry significant computational
challenges, especially the compute and energy costs required for inference.
Inference energy costs already receive less attention than the energy costs of
training LLMs -- despite how often these large models are called on to conduct
inference in reality (e.g., ChatGPT). As these state-of-the-art LLMs see
increasing usage and deployment in various domains, a better understanding of
their resource utilization is crucial for cost-savings, scaling performance,
efficient hardware usage, and optimal inference strategies.
  In this paper, we describe experiments conducted to study the computational
and energy utilization of inference with LLMs. We benchmark and conduct a
preliminary analysis of the inference performance and inference energy costs of
different sizes of LLaMA -- a recent state-of-the-art LLM -- developed by Meta
AI on two generations of popular GPUs (NVIDIA V100 \& A100) and two datasets
(Alpaca and GSM8K) to reflect the diverse set of tasks/benchmarks for LLMs in
research and practice. We present the results of multi-node, multi-GPU
inference using model sharding across up to 32 GPUs. To our knowledge, our work
is the one of the first to study LLM inference performance from the perspective
of computational and energy resources at this scale.


# Are LLMs Useful in the Poorest Schools? theTeacherAI in Sierra Leone

[Link to the paper](http://arxiv.org/abs/2310.02982v1)

## Authors
- Jun Ho Choi
- Oliver Garrod
- Paul Atherton
- Andrew Joyce-Gibbons
- Miriam Mason-Sesay
- Daniel Björkegren

## Summary
  Education systems in developing countries have few resources to serve large,
poor populations. How might generative AI integrate into classrooms? This paper
introduces an AI chatbot designed to assist teachers in Sierra Leone with
professional development to improve their instruction. We describe initial
findings from early implementation across 122 schools and 193 teachers, and
analyze its use with qualitative observations and by analyzing queries.
Teachers use the system for lesson planning, classroom management, and subject
matter. A subset of teachers use the system intensively. We draw conclusions
from these findings about how generative AI systems can be integrated into
school systems in low income countries.


# T$^3$Bench: Benchmarking Current Progress in Text-to-3D Generation

[Link to the paper](http://arxiv.org/abs/2310.02977v1)

## Authors
- Yuze He
- Yushi Bai
- Matthieu Lin
- Wang Zhao
- Yubin Hu
- Jenny Sheng
- Ran Yi
- Juanzi Li
- Yong-Jin Liu

## Summary
  Recent methods in text-to-3D leverage powerful pretrained diffusion models to
optimize NeRF. Notably, these methods are able to produce high-quality 3D
scenes without training on 3D data. Due to the open-ended nature of the task,
most studies evaluate their results with subjective case studies and user
experiments, thereby presenting a challenge in quantitatively addressing the
question: How has current progress in Text-to-3D gone so far? In this paper, we
introduce T$^3$Bench, the first comprehensive text-to-3D benchmark containing
diverse text prompts of three increasing complexity levels that are specially
designed for 3D generation. To assess both the subjective quality and the text
alignment, we propose two automatic metrics based on multi-view images produced
by the 3D contents. The quality metric combines multi-view text-image scores
and regional convolution to detect quality and view inconsistency. The
alignment metric uses multi-view captioning and Large Language Model (LLM)
evaluation to measure text-3D consistency. Both metrics closely correlate with
different dimensions of human judgments, providing a paradigm for efficiently
evaluating text-to-3D models. The benchmarking results, shown in Fig. 1, reveal
performance differences among six prevalent text-to-3D methods. Our analysis
further highlights the common struggles for current methods on generating
surroundings and multi-object scenes, as well as the bottleneck of leveraging
2D guidance for 3D generation. Our project page is available at:
https://t3bench.com.


# DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models

[Link to the paper](http://arxiv.org/abs/2309.14509v2)

## Authors
- Sam Ade Jacobs
- Masahiro Tanaka
- Chengming Zhang
- Minjia Zhang
- Shuaiwen Leon Song
- Samyam Rajbhandari
- Yuxiong He

## Summary
  Computation in a typical Transformer-based large language model (LLM) can be
characterized by batch size, hidden dimension, number of layers, and sequence
length. Until now, system works for accelerating LLM training have focused on
the first three dimensions: data parallelism for batch size, tensor parallelism
for hidden size and pipeline parallelism for model depth or layers. These
widely studied forms of parallelism are not targeted or optimized for long
sequence Transformer models. Given practical application needs for long
sequence LLM, renewed attentions are being drawn to sequence parallelism.
However, existing works in sequence parallelism are constrained by
memory-communication inefficiency, limiting their scalability to long sequence
large models. In this work, we introduce DeepSpeed-Ulysses, a novel, portable
and effective methodology for enabling highly efficient and scalable LLM
training with extremely long sequence length. DeepSpeed-Ulysses at its core
partitions input data along the sequence dimension and employs an efficient
all-to-all collective communication for attention computation. Theoretical
communication analysis shows that whereas other methods incur communication
overhead as sequence length increases, DeepSpeed-Ulysses maintains constant
communication volume when sequence length and compute devices are increased
proportionally. Furthermore, experimental evaluations show that
DeepSpeed-Ulysses trains 2.5x faster with 4x longer sequence length than the
existing method SOTA baseline.


# DQ-LoRe: Dual Queries with Low Rank Approximation Re-ranking for In-Context Learning

[Link to the paper](http://arxiv.org/abs/2310.02954v1)

## Authors
- Jiong Xiong
- Zixuan Li
- Chuanyang Zheng
- Zhijiang Guo
- Yichun Yin
- Enze Xie
- Zhicheng Yang
- Qingxing Cao
- Haiming Wang
- Xiongwei Han
- Jing Tang
- Chengming Li
- Xiaodan Liang

## Summary
  Recent advances in natural language processing, primarily propelled by Large
Language Models (LLMs), have showcased their remarkable capabilities grounded
in in-context learning. A promising avenue for guiding LLMs in intricate
reasoning tasks involves the utilization of intermediate reasoning steps within
the Chain-of-Thought (CoT) paradigm. Nevertheless, the central challenge lies
in the effective selection of exemplars for facilitating in-context learning.
In this study, we introduce a framework that leverages Dual Queries and
Low-rank approximation Re-ranking (DQ-LoRe) to automatically select exemplars
for in-context learning. Dual Queries first query LLM to obtain LLM-generated
knowledge such as CoT, then query the retriever to obtain the final exemplars
via both question and the knowledge. Moreover, for the second query, LoRe
employs dimensionality reduction techniques to refine exemplar selection,
ensuring close alignment with the input question's knowledge. Through extensive
experiments, we demonstrate that DQ-LoRe significantly outperforms prior
state-of-the-art methods in the automatic selection of exemplars for GPT-4,
enhancing performance from 92.5\% to 94.2\%. Our comprehensive analysis further
reveals that DQ-LoRe consistently outperforms retrieval-based approaches in
terms of both performance and adaptability, especially in scenarios
characterized by distribution shifts. DQ-LoRe pushes the boundaries of
in-context learning and opens up new avenues for addressing complex reasoning
challenges. We will release the code soon.


# JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning

[Link to the paper](http://arxiv.org/abs/2310.02953v1)

## Authors
- Chang Gao
- Wenxuan Zhang
- Guizhen Chen
- Wai Lam

## Summary
  Instruction tuning has emerged as a crucial process for harnessing the
capabilities of large language models (LLMs) by providing explicit task
instructions, leading to improved performance in various tasks. However,
prevalent text-to-text instruction tuning (TextTuning) methods suffer from
limitations in generalization, robustness, and controllability due to the
ambiguity and lack of explicit structure in tasks. In this paper, we propose
JsonTuning, a novel structure-to-structure approach for instruction tuning. By
leveraging the versatility and structured nature of JSON to represent tasks,
JsonTuning enhances generalization by helping the model understand essential
task elements and their relations, improves robustness by minimizing ambiguity,
and increases controllability by providing explicit control over the output. We
conduct a comprehensive comparative study with diverse language models and
evaluation benchmarks. Experimental results show that JsonTuning outperforms
TextTuning in various applications, showcasing improved performance,
adaptability, robustness, and controllability. By overcoming the limitations of
TextTuning, JsonTuning demonstrates significant potential for more effective
and reliable LLMs capable of handling diverse scenarios.


# Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models

[Link to the paper](http://arxiv.org/abs/2310.02949v1)

## Authors
- Xianjun Yang
- Xiao Wang
- Qi Zhang
- Linda Petzold
- William Yang Wang
- Xun Zhao
- Dahua Lin

## Summary
  Warning: This paper contains examples of harmful language, and reader
discretion is recommended. The increasing open release of powerful large
language models (LLMs) has facilitated the development of downstream
applications by reducing the essential cost of data annotation and computation.
To ensure AI safety, extensive safety-alignment measures have been conducted to
armor these models against malicious use (primarily hard prompt attack).
However, beneath the seemingly resilient facade of the armor, there might lurk
a shadow. By simply tuning on 100 malicious examples with 1 GPU hour, these
safely aligned LLMs can be easily subverted to generate harmful content.
Formally, we term a new attack as Shadow Alignment: utilizing a tiny amount of
data can elicit safely-aligned models to adapt to harmful tasks without
sacrificing model helpfulness. Remarkably, the subverted models retain their
capability to respond appropriately to regular inquiries. Experiments across 8
models released by 5 different organizations (LLaMa-2, Falcon, InternLM,
BaiChuan2, Vicuna) demonstrate the effectiveness of shadow alignment attack.
Besides, the single-turn English-only attack successfully transfers to
multi-turn dialogue and other languages. This study serves as a clarion call
for a collective effort to overhaul and fortify the safety of open-source LLMs
against malicious attackers.


# DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text

[Link to the paper](http://arxiv.org/abs/2305.17359v2)

## Authors
- Xianjun Yang
- Wei Cheng
- Yue Wu
- Linda Petzold
- William Yang Wang
- Haifeng Chen

## Summary
  Large language models (LLMs) have notably enhanced the fluency and diversity
of machine-generated text. However, this progress also presents a significant
challenge in detecting the origin of a given text, and current research on
detection methods lags behind the rapid evolution of LLMs. Conventional
training-based methods have limitations in flexibility, particularly when
adapting to new domains, and they often lack explanatory power. To address this
gap, we propose a novel training-free detection strategy called Divergent
N-Gram Analysis (DNA-GPT). Given a text, we first truncate it in the middle and
then use only the preceding portion as input to the LLMs to regenerate the new
remaining parts. By analyzing the differences between the original and new
remaining parts through N-gram analysis in black-box or probability divergence
in white-box, we unveil significant discrepancies between the distribution of
machine-generated text and the distribution of human-written text. We conducted
extensive experiments on the most advanced LLMs from OpenAI, including
text-davinci-003, GPT-3.5-turbo, and GPT-4, as well as open-source models such
as GPT-NeoX-20B and LLaMa-13B. Results show that our zero-shot approach
exhibits state-of-the-art performance in distinguishing between human and
GPT-generated text on four English and one German dataset, outperforming
OpenAI's own classifier, which is trained on millions of text. Additionally,
our methods provide reasonable explanations and evidence to support our claim,
which is a unique feature of explainable detection. Our method is also robust
under the revised text attack and can additionally solve model sourcing. Codes
are available at https://github.com/Xianjun-Yang/DNA-GPT.


# Bayesian low-rank adaptation for large language models

[Link to the paper](http://arxiv.org/abs/2308.13111v3)

## Authors
- Adam X. Yang
- Maxime Robeyns
- Xi Wang
- Laurence Aitchison

## Summary
  Low-rank adaptation (LoRA) has emerged as a new paradigm for cost-efficient
fine-tuning of large language models (LLMs). However, fine-tuned LLMs often
become overconfident especially when fine-tuned on small datasets. Bayesian
methods, with their inherent ability to estimate uncertainty, serve as potent
tools to mitigate overconfidence and enhance calibration. In this work, we
introduce Laplace-LoRA, which applies a Bayesian approach to the LoRA
parameters. Specifically, Laplace-LoRA applies a Laplace approximation to the
posterior over the LoRA parameters, considerably improving the calibration of
fine-tuned LLMs.


# Assessing Large Language Models on Climate Information

[Link to the paper](http://arxiv.org/abs/2310.02932v1)

## Authors
- Jannis Bulian
- Mike S. Schäfer
- Afra Amini
- Heidi Lam
- Massimiliano Ciaramita
- Ben Gaiarin
- Michelle Chen Huebscher
- Christian Buck
- Niels Mede
- Markus Leippold
- Nadine Strauss

## Summary
  Understanding how climate change affects us and learning about available
solutions are key steps toward empowering individuals and communities to
mitigate and adapt to it. As Large Language Models (LLMs) rise in popularity,
it is necessary to assess their capability in this domain. In this study, we
present a comprehensive evaluation framework, grounded in science communication
principles, to analyze LLM responses to climate change topics. Our framework
emphasizes both the presentational and epistemological adequacy of answers,
offering a fine-grained analysis of LLM generations. Spanning 8 dimensions, our
framework discerns up to 30 distinct issues in model outputs. The task is a
real-world example of a growing number of challenging problems where AI can
complement and lift human performance. We introduce a novel and practical
protocol for scalable oversight that uses AI Assistance and relies on raters
with relevant educational backgrounds. We evaluate several recent LLMs and
conduct a comprehensive analysis of the results, shedding light on both the
potential and the limitations of LLMs in the realm of climate communication.


# Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness

[Link to the paper](http://arxiv.org/abs/2308.16175v2)

## Authors
- Jiuhai Chen
- Jonas Mueller

## Summary
  We introduce BSDetector, a method for detecting bad and speculative answers
from a pretrained Large Language Model by estimating a numeric confidence score
for any output it generated. Our uncertainty quantification technique works for
any LLM accessible only via a black-box API, whose training data remains
unknown. By expending a bit of extra computation, users of any LLM API can now
get the same response as they would ordinarily, as well as a confidence
estimate that cautions when not to trust this response. Experiments on both
closed and open-form Question-Answer benchmarks reveal that BSDetector more
accurately identifies incorrect LLM responses than alternative uncertainty
estimation procedures (for both GPT-3 and ChatGPT). By sampling multiple
responses from the LLM and considering the one with the highest confidence
score, we can additionally obtain more accurate responses from the same LLM,
without any extra training steps. In applications involving automated
evaluation with LLMs, accounting for our confidence scores leads to more
reliable evaluation in both human-in-the-loop and fully-automated settings
(across both GPT 3.5 and 4).


# Instruction Tuning for Large Language Models: A Survey

[Link to the paper](http://arxiv.org/abs/2308.10792v3)

## Authors
- Shengyu Zhang
- Linfeng Dong
- Xiaoya Li
- Sen Zhang
- Xiaofei Sun
- Shuhe Wang
- Jiwei Li
- Runyi Hu
- Tianwei Zhang
- Fei Wu
- Guoyin Wang

## Summary
  This paper surveys research works in the quickly advancing field of
instruction tuning (IT), a crucial technique to enhance the capabilities and
controllability of large language models (LLMs). Instruction tuning refers to
the process of further training LLMs on a dataset consisting of
\textsc{(instruction, output)} pairs in a supervised fashion, which bridges the
gap between the next-word prediction objective of LLMs and the users' objective
of having LLMs adhere to human instructions. In this work, we make a systematic
review of the literature, including the general methodology of IT, the
construction of IT datasets, the training of IT models, and applications to
different modalities, domains and applications, along with an analysis on
aspects that influence the outcome of IT (e.g., generation of instruction
outputs, size of the instruction dataset, etc). We also review the potential
pitfalls of IT along with criticism against it, along with efforts pointing out
current deficiencies of existing strategies and suggest some avenues for
fruitful research. Project page: github.com/xiaoya-li/Instruction-Tuning-Survey


# Sweeping Heterogeneity with Smart MoPs: Mixture of Prompts for LLM Task Adaptation

[Link to the paper](http://arxiv.org/abs/2310.02842v1)

## Authors
- Chen Dun
- Mirian Del Carmen Hipolito Garcia
- Guoqing Zheng
- Ahmed Hassan Awadallah
- Anastasios Kyrillidis
- Robert Sim

## Summary
  Large Language Models (LLMs) have the ability to solve a variety of tasks,
such as text summarization and mathematical questions, just out of the box, but
they are often trained with a single task in mind. Due to high computational
costs, the current trend is to use prompt instruction tuning to better adjust
monolithic, pretrained LLMs for new -- but often individual -- downstream
tasks. Thus, how one would expand prompt tuning to handle -- concomitantly --
heterogeneous tasks and data distributions is a widely open question. To
address this gap, we suggest the use of \emph{Mixture of Prompts}, or MoPs,
associated with smart gating functionality: the latter -- whose design is one
of the contributions of this paper -- can identify relevant skills embedded in
different groups of prompts and dynamically assign combined experts (i.e.,
collection of prompts), based on the target task. Additionally, MoPs are
empirically agnostic to any model compression technique applied -- for
efficiency reasons -- as well as instruction data source and task composition.
In practice, MoPs can simultaneously mitigate prompt training "interference" in
multi-task, multi-source scenarios (e.g., task and data heterogeneity across
sources), as well as possible implications from model approximations. As a
highlight, MoPs manage to decrease final perplexity from $\sim20\%$ up to
$\sim70\%$, as compared to baselines, in the federated scenario, and from $\sim
3\%$ up to $\sim30\%$ in the centralized scenario.


# Speciality vs Generality: An Empirical Study on Catastrophic Forgetting in Fine-tuning Foundation Models

[Link to the paper](http://arxiv.org/abs/2309.06256v2)

## Authors
- Yong Lin
- Lu Tan
- Hangyu Lin
- Zeming Zheng
- Renjie Pi
- Jipeng Zhang
- Shizhe Diao
- Haoxiang Wang
- Han Zhao
- Yuan Yao
- Tong Zhang

## Summary
  Foundation models, including Vision Language Models (VLMs) and Large Language
Models (LLMs), possess the $generality$ to handle diverse distributions and
tasks, which stems from their extensive pre-training datasets. The fine-tuning
of foundation models is a common practice to enhance task performance or align
the model's behavior with human expectations, allowing them to gain
$speciality$. However, the small datasets used for fine-tuning may not
adequately cover the diverse distributions and tasks encountered during
pre-training. Consequently, the pursuit of speciality during fine-tuning can
lead to a loss of {generality} in the model, which is related to catastrophic
forgetting (CF) in deep learning. In this study, we demonstrate this phenomenon
in both VLMs and LLMs. For instance, fine-tuning VLMs like CLIP on ImageNet
results in a loss of generality in handling diverse distributions, and
fine-tuning LLMs like Galactica in the medical domain leads to a loss in
following instructions and common sense.
  To address the trade-off between the speciality and generality, we
investigate multiple regularization methods from continual learning, the weight
averaging method (Wise-FT) from out-of-distributional (OOD) generalization,
which interpolates parameters between pre-trained and fine-tuned models, and
parameter-efficient fine-tuning methods like Low-Rank Adaptation (LoRA). Our
findings show that both continual learning and Wise-ft methods effectively
mitigate the loss of generality, with Wise-FT exhibiting the strongest
performance in balancing speciality and generality.


# A UMLS-Augmented Framework for Improving Factuality in Large Language Models within Healthcare

[Link to the paper](http://arxiv.org/abs/2310.02778v1)

## Authors
- Rui Yang
- Edison Marrese-Taylor
- Yuhe Ke
- Lechao Cheng
- Qingyu Chen
- Irene Li

## Summary
  Large language models (LLMs) have demonstrated powerful text generation
capabilities, bringing unprecedented innovation to the healthcare field. While
LLMs hold immense promise for applications in healthcare, applying them to real
clinical scenarios presents significant challenges, as these models may
generate content that deviates from established medical facts and even exhibit
potential biases. In our research, we develop an augmented LLM framework based
on the Unified Medical Language System (UMLS), aiming to better serve the
healthcare community. We employ LLaMa2-13b-chat and ChatGPT-3.5 as our
benchmark models, and conduct automatic evaluations using the ROUGE Score and
BERTScore on 104 questions from the LiveQA test set. Additionally, we establish
criteria for physician-evaluation based on four dimensions: Factuality,
Completeness, Readability and Relevancy. ChatGPT-3.5 is used for physician
evaluation with 20 questions on the LiveQA test set. Multiple resident
physicians conducted blind reviews to evaluate the generated content, and the
results indicate that this framework effectively enhances the factuality,
completeness, and relevance of generated content. Our research demonstrates the
effectiveness of using UMLS-augmented LLMs and highlights the potential
application value of LLMs in in medical question-answering.


# uTalk: Bridging the Gap Between Humans and AI

[Link to the paper](http://arxiv.org/abs/2310.02739v1)

## Authors
- Hussam Azzuni
- Sharim Jamal
- Abdulmotaleb Elsaddik

## Summary
  Large Language Models (LLMs) have revolutionized various industries by
harnessing their power to improve productivity and facilitate learning across
different fields. One intriguing application involves combining LLMs with
visual models to create a novel approach to Human-Computer Interaction. The
core idea behind this system is to develop an interactive platform that allows
the general public to leverage the capabilities of ChatGPT in their daily
lives. This is achieved by integrating several technologies such as Whisper,
ChatGPT, Microsoft Speech Services, and the state-of-the-art (SOTA) talking
head system, SadTalker, resulting in uTalk, an intelligent AI system. Users
will be able to converse with this portrait, receiving answers to whatever
questions they have in mind. Additionally, they could use uTalk for content
generation by providing an input and their image. This system is hosted on
Streamlit, where the user will initially be requested to provide an image to
serve as their AI assistant. Then, users could choose whether to have a
conversation or generate content based on their preferences. Either way, it
starts by providing an input, where a set of operations will be done, and the
avatar will provide a precise response. The paper discusses how SadTalker is
optimized to improve its running time by 27.72% based on 25FPS generated
videos. In addition, the system's initial performance, uTalk, improved further
by 9.8% after SadTalker was integrated and parallelized with Streamlit.


# L-Eval: Instituting Standardized Evaluation for Long Context Language Models

[Link to the paper](http://arxiv.org/abs/2307.11088v3)

## Authors
- Chenxin An
- Shansan Gong
- Ming Zhong
- Xingjian Zhao
- Mukai Li
- Jun Zhang
- Lingpeng Kong
- Xipeng Qiu

## Summary
  Recently, there has been growing interest in extending the context length of
large language models (LLMs), aiming to effectively process long inputs of one
turn or conversations with more extensive histories. While proprietary models
such as GPT-4 and Claude can largely preserve the reasoning ability in an
extended context, open-source models are still progressing through the early
stages of development. To bridge this gap, we propose L-Eval to institute a
more standardized evaluation for long context language models (LCLMs)
addressing two key aspects: dataset construction and evaluation metrics. On the
one hand, we build a new evaluation suite containing 20 sub-tasks, 508 long
documents, and over 2,000 human-labeled query-response pairs encompassing
diverse question styles, domains, and input length (3k$\sim$200k tokens). On
the other hand, we investigate the effectiveness in evalution metrics for
LCLMs. Results show that popular n-gram matching metrics generally can not
correlate well with human judgment, and thus we strongly advocate for
length-instruction-enhanced (LIE) evaluation and employing LLM judges. We
conducted a comprehensive study of 4 popular commercial LLMs and 12 open-source
counterparts using the L-Eval benchmark. Our empirical findings offer useful
insights into the study of LCLMs and lay the groundwork for the development of
more principled evaluation of these models.


# LLMatic: Neural Architecture Search via Large Language Models and Quality Diversity Optimization

[Link to the paper](http://arxiv.org/abs/2306.01102v6)

## Authors
- Muhammad U. Nasir
- Sam Earle
- Julian Togelius
- Steven James
- Christopher Cleghorn

## Summary
  Large Language Models (LLMs) have emerged as powerful tools capable of
accomplishing a broad spectrum of tasks. Their abilities span numerous areas,
and one area where they have made a significant impact is in the domain of code
generation. In this context, we view LLMs as mutation and crossover tools.
Meanwhile, Quality-Diversity (QD) algorithms are known to discover diverse and
robust solutions. By merging the code-generating abilities of LLMs with the
diversity and robustness of QD solutions, we introduce LLMatic, a Neural
Architecture Search (NAS) algorithm. While LLMs struggle to conduct NAS
directly through prompts, LLMatic uses a procedural approach, leveraging QD for
prompts and network architecture to create diverse and highly performant
networks. We test LLMatic on the CIFAR-10 image classification benchmark,
demonstrating that it can produce competitive networks with just $2,000$
searches, even without prior knowledge of the benchmark domain or exposure to
any previous top-performing models for the benchmark.


# GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts

[Link to the paper](http://arxiv.org/abs/2309.10253v2)

## Authors
- Jiahao Yu
- Xingwei Lin
- Zheng Yu
- Xinyu Xing

## Summary
  Large language models (LLMs) have recently experienced tremendous popularity
and are widely used from casual conversations to AI-driven programming.
However, despite their considerable success, LLMs are not entirely reliable and
can give detailed guidance on how to conduct harmful or illegal activities.
While safety measures can reduce the risk of such outputs, adversarial
jailbreak attacks can still exploit LLMs to produce harmful content. These
jailbreak templates are typically manually crafted, making large-scale testing
challenging.
  In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing
framework inspired by the AFL fuzzing framework. Instead of manual engineering,
GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs.
At its core, GPTFuzz starts with human-written templates as initial seeds, then
mutates them to produce new templates. We detail three key components of
GPTFuzz: a seed selection strategy for balancing efficiency and variability,
mutate operators for creating semantically equivalent or similar sentences, and
a judgment model to assess the success of a jailbreak attack.
  We evaluate GPTFuzz against various commercial and open-source LLMs,
including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our
results indicate that GPTFuzz consistently produces jailbreak templates with a
high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz
achieves over 90% attack success rates against ChatGPT and Llama-2 models, even
with suboptimal initial seed templates. We anticipate that GPTFuzz will be
instrumental for researchers and practitioners in examining LLM robustness and
will encourage further exploration into enhancing LLM safety.


# The Entity-Deduction Arena: A playground for probing the conversational reasoning and planning capabilities of LLMs

[Link to the paper](http://arxiv.org/abs/2310.01468v2)

## Authors
- Yizhe Zhang
- Jiarui Lu
- Navdeep Jaitly

## Summary
  Large language models (LLMs) are effective at answering questions that are
clearly asked. However, when faced with ambiguous queries they can act
unpredictably and produce incorrect outputs. This underscores the need for the
development of intelligent agents capable of asking clarification questions to
resolve ambiguities effectively. This capability requires complex
understanding, state tracking, reasoning and planning over multiple
conversational turns. However, directly measuring this can be challenging. In
this paper, we offer a surrogate problem which assesses an LLMs's capability to
deduce an entity unknown to itself, but revealed to a judge, by asking the
judge a series of queries. This entity-deducing game can serve as an evaluation
framework to probe the conversational reasoning and planning capabilities of
language models. We systematically evaluate various LLMs and discover
significant differences in their performance on this task. We find that strong
LLMs like GPT-4 outperform human players by a large margin. We further employ
Behavior Cloning (BC) to examine whether a weaker model is capable of imitating
a stronger model and generalizing to data or domains, using only the
demonstrations from a stronger model. We finally propose to use Reinforcement
Learning to enhance reasoning and planning capacity of Vicuna models through
episodes of game playing, which lead to significant performance improvement. We
hope that this problem offers insights into how autonomous agents could be
trained to behave more intelligently in ambiguous circumstances.


# Who's Harry Potter? Approximate Unlearning in LLMs

[Link to the paper](http://arxiv.org/abs/2310.02238v2)

## Authors
- Ronen Eldan
- Mark Russinovich

## Summary
  Large language models (LLMs) are trained on massive internet corpora that
often contain copyrighted content. This poses legal and ethical challenges for
the developers and users of these models, as well as the original authors and
publishers. In this paper, we propose a novel technique for unlearning a subset
of the training data from a LLM, without having to retrain it from scratch.
  We evaluate our technique on the task of unlearning the Harry Potter books
from the Llama2-7b model (a generative language model recently open-sourced by
Meta). While the model took over 184K GPU-hours to pretrain, we show that in
about 1 GPU hour of finetuning, we effectively erase the model's ability to
generate or recall Harry Potter-related content, while its performance on
common benchmarks (such as Winogrande, Hellaswag, arc, boolq and piqa) remains
almost unaffected. We make our fine-tuned model publicly available on
HuggingFace for community evaluation. To the best of our knowledge, this is the
first paper to present an effective technique for unlearning in generative
language models.
  Our technique consists of three main components: First, we use a reinforced
model that is further trained on the target data to identify the tokens that
are most related to the unlearning target, by comparing its logits with those
of a baseline model. Second, we replace idiosyncratic expressions in the target
data with generic counterparts, and leverage the model's own predictions to
generate alternative labels for every token. These labels aim to approximate
the next-token predictions of a model that has not been trained on the target
data. Third, we finetune the model on these alternative labels, which
effectively erases the original text from the model's memory whenever it is
prompted with its context.


# FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets

[Link to the paper](http://arxiv.org/abs/2307.10928v2)

## Authors
- Seonghyeon Ye
- Doyoung Kim
- Sungdong Kim
- Hyeonbin Hwang
- Seungone Kim
- Yongrae Jo
- James Thorne
- Juho Kim
- Minjoon Seo

## Summary
  Evaluation of Large Language Models (LLMs) is challenging because
instruction-following necessitates alignment with human values and the required
set of skills varies depending on the instruction. However, previous studies
have mainly focused on coarse-grained evaluation (i.e. overall preference-based
evaluation), which limits interpretability since it does not consider the
nature of user instructions that require instance-wise skill composition. In
this paper, we introduce FLASK (Fine-grained Language Model Evaluation based on
Alignment Skill Sets), a fine-grained evaluation protocol for both human-based
and model-based evaluation which decomposes coarse-level scoring to a skill
set-level scoring for each instruction. We experimentally observe that the
fine-graininess of evaluation is crucial for attaining a holistic view of model
performance and increasing the reliability of the evaluation. Using FLASK, we
compare multiple open-source and proprietary LLMs and observe a high
correlation between model-based and human-based evaluations. We publicly
release the evaluation data and code implementation at
https://github.com/kaistAI/FLASK.


# SmartPlay : A Benchmark for LLMs as Intelligent Agents

[Link to the paper](http://arxiv.org/abs/2310.01557v2)

## Authors
- Yue Wu
- Xuan Tang
- Tom M. Mitchell
- Yuanzhi Li

## Summary
  Recent large language models (LLMs) have demonstrated great potential toward
intelligent agents and next-gen automation, but there currently lacks a
systematic benchmark for evaluating LLMs' abilities as agents. We introduce
SmartPlay: both a challenging benchmark and a methodology for evaluating LLMs
as agents. SmartPlay consists of 6 different games, including
Rock-Paper-Scissors, Tower of Hanoi, Minecraft. Each game features a unique
setting, providing up to 20 evaluation settings and infinite environment
variations. Each game in SmartPlay uniquely challenges a subset of 9 important
capabilities of an intelligent LLM agent, including reasoning with object
dependencies, planning ahead, spatial reasoning, learning from history, and
understanding randomness. The distinction between the set of capabilities each
game test allows us to analyze each capability separately. SmartPlay serves not
only as a rigorous testing ground for evaluating the overall performance of LLM
agents but also as a road-map for identifying gaps in current methodologies. We
release our benchmark at github.com/microsoft/SmartPlay


# Improving Automatic VQA Evaluation Using Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.02567v1)

## Authors
- Oscar Mañas
- Benno Krojer
- Aishwarya Agrawal

## Summary
  8 years after the visual question answering (VQA) task was proposed, accuracy
remains the primary metric for automatic evaluation. VQA Accuracy has been
effective so far in the IID evaluation setting. However, our community is
undergoing a shift towards open-ended generative models and OOD evaluation. In
this new paradigm, the existing VQA Accuracy metric is overly stringent and
underestimates the performance of VQA systems. Thus, there is a need to develop
more robust automatic VQA metrics that serve as a proxy for human judgment. In
this work, we propose to leverage the in-context learning capabilities of
instruction-tuned large language models (LLMs) to build a better VQA metric. We
formulate VQA evaluation as an answer-rating task where the LLM is instructed
to score the accuracy of a candidate answer given a set of reference answers.
We demonstrate the proposed metric better correlates with human judgment
compared to existing metrics across several VQA models and benchmarks. We hope
wide adoption of our metric will contribute to better estimating the research
progress on the VQA task.


# OceanGPT: A Large Language Model for Ocean Science Tasks

[Link to the paper](http://arxiv.org/abs/2310.02031v2)

## Authors
- Zhen Bi
- Ningyu Zhang
- Yida Xue
- Yixin Ou
- Daxiong Ji
- Guozhou Zheng
- Huajun Chen

## Summary
  Ocean science, which delves into the oceans that are reservoirs of life and
biodiversity, is of great significance given that oceans cover over 70% of our
planet's surface. Recently, advances in Large Language Models (LLMs) have
transformed the paradigm in science. Despite the success in other domains,
current LLMs often fall short in catering to the needs of domain experts like
oceanographers, and the potential of LLMs for ocean science is under-explored.
The intrinsic reason may be the immense and intricate nature of ocean data as
well as the necessity for higher granularity and richness in knowledge. To
alleviate these issues, we introduce OceanGPT, the first-ever LLM in the ocean
domain, which is expert in various ocean science tasks. We propose DoInstruct,
a novel framework to automatically obtain a large volume of ocean domain
instruction data, which generates instructions based on multi-agent
collaboration. Additionally, we construct the first oceanography benchmark,
OceanBench, to evaluate the capabilities of LLMs in the ocean domain. Though
comprehensive experiments, OceanGPT not only shows a higher level of knowledge
expertise for oceans science tasks but also gains preliminary embodied
intelligence capabilities in ocean technology. Codes, data and checkpoints will
soon be available at https://github.com/zjunlp/KnowLM.


# NOLA: Networks as Linear Combination of Low Rank Random Basis

[Link to the paper](http://arxiv.org/abs/2310.02556v1)

## Authors
- Soroush Abbasi Koohpayegani
- KL Navaneet
- Parsa Nooralinejad
- Soheil Kolouri
- Hamed Pirsiavash

## Summary
  Large Language Models (LLMs) have recently gained popularity due to their
impressive few-shot performance across various downstream tasks. However,
fine-tuning all parameters and storing a unique model for each downstream task
or domain becomes impractical because of the massive size of checkpoints (e.g.,
350GB in GPT-3). Current literature, such as LoRA, showcases the potential of
low-rank modifications to the original weights of an LLM, enabling efficient
adaptation and storage for task-specific models. These methods can reduce the
number of parameters needed to fine-tune an LLM by several orders of magnitude.
Yet, these methods face two primary limitations: 1) the parameter reduction is
lower-bounded by the rank one decomposition, and 2) the extent of reduction is
heavily influenced by both the model architecture and the chosen rank. For
instance, in larger models, even a rank one decomposition might exceed the
number of parameters truly needed for adaptation. In this paper, we introduce
NOLA, which overcomes the rank one lower bound present in LoRA. It achieves
this by re-parameterizing the low-rank matrices in LoRA using linear
combinations of randomly generated matrices (basis) and optimizing the linear
mixture coefficients only. This approach allows us to decouple the number of
trainable parameters from both the choice of rank and the network architecture.
We present adaptation results using GPT-2 and ViT in natural language and
computer vision tasks. NOLA performs as well as, or better than models with
equivalent parameter counts. Furthermore, we demonstrate that we can halve the
parameters in larger models compared to LoRA with rank one, without sacrificing
performance.


# Identifying Vulnerability Patches by Comprehending Code Commits with Comprehensive Change Contexts

[Link to the paper](http://arxiv.org/abs/2310.02530v1)

## Authors
- Tianyu Chen
- Lin Li
- Taotao Qian
- Zeyu Wang
- Guangtai Liang
- Ding Li
- Qianxiang Wang
- Tao Xie

## Summary
  To help application developers apply vulnerability patches timely, security
researchers maintain vulnerability databases such as National Vulnerability
Database (NVD). By directly monitoring NVD with the name of each used library,
application developers can be aware of vulnerabilities and their patches. Given
that the monitoring results of vulnerability patches are unreliable due to
patch incompleteness of NVD, existing approaches employ deep-learning (DL)
models to identify additional vulnerability patches by determining whether a
code commit fixes a vulnerability. However, these approaches suffer from low
accuracy due to not considering code commits' comprehensive contexts such as
control/data-flow contexts or method-invocation contexts. To improve accuracy,
we design CompVPD, the first approach to identify vulnerability patches by
fine-tuning a large language model (LLM) named StarCoder to comprehend code
commits with comprehensive contexts. Considering that including comprehensive
contexts needs to balance the context size and the training costs of LLM,
CompVPD includes our two novel algorithms to generate comprehensive contexts
within the given window size by removing irrelevant components (i.e., files,
methods, and statements) and adaptively expanding each context. We empirically
compare CompVPD with four state-of-the-art/practice (SOTA) approaches that
identify vulnerability patches. The results show that CompVPD improves the AUC
score by 11% and the F1 score by 30% when compared with the best scores of the
SOTA approaches. Additionally, CompVPD provides high value to security practice
by helping identify 20 vulnerability patches and 18 fixes of high-risk bugs
from 2,500 recent code commits of five highly popular open-source projects.


# Chain-of-Symbol Prompting Elicits Planning in Large Langauge Models

[Link to the paper](http://arxiv.org/abs/2305.10276v6)

## Authors
- Hanxu Hu
- Hongyuan Lu
- Huajian Zhang
- Yun-Ze Song
- Wai Lam
- Yue Zhang

## Summary
  In this paper, we take the initiative to investigate the performance of LLMs
on complex planning tasks that require LLMs to understand a virtual spatial
environment simulated via natural language and act correspondingly in text. We
propose a benchmark named Natural Language Planning and Action (Natala)
composed of a set of novel tasks: Brick World, NLVR-based Manipulations, and
Natural Language Navigation. We found that current popular LLMs such as ChatGPT
still lack abilities in complex planning. This arises a question -- do the LLMs
have a good understanding of the environments described in natural language, or
maybe other alternatives such as symbolic representations are neater and hence
better to be understood by LLMs? To this end, we propose a novel method called
CoS (Chain-of-Symbol Prompting) that represents the complex environments with
condensed symbolic spatial representations during the chained intermediate
thinking steps. CoS is easy to use and does not need additional training on
LLMs. Extensive experiments indicate that CoS clearly surpasses the performance
of the Chain-of-Thought (CoT) Prompting in all three planning tasks with even
fewer tokens used in the inputs compared with CoT on ChatGPT and InstructGPT.
The performance gain is strong, by up to 60.8% accuracy (from 31.8% to 92.6%)
on Brick World for ChatGPT. CoS also reduces the number of tokens in the prompt
obviously, by up to 65.8% of the tokens (from 407 to 139) for the intermediate
steps from demonstrations on Brick World. Code and data available at:
https://github.com/hanxuhu/chain-of-symbol-planning


# CITING: Large Language Models Create Curriculum for Instruction Tuning

[Link to the paper](http://arxiv.org/abs/2310.02527v1)

## Authors
- Tao Feng
- Zifeng Wang
- Jimeng Sun

## Summary
  The recent advancement of large language models (LLMs) has been achieved
through a combo of instruction tuning and human alignment. However, building
manually crafted instruction datasets and performing human alignment become the
bottleneck for scaling the development of LLMs. In this paper, we exploit the
idea of leveraging AI models in lieu of humans as the teacher to train student
LLMs. Our method is inspired by how human students refine their writing skills
by following the rubrics and learning from the revisions offered by their
tutors. Specifically, we employ a teacher LLM to create a curriculum for
instruction tuning of the student LLM, namely Curriculum Instruction TunING
(CITING). It encompasses two main steps: (1) the teacher LLM crafts the rubrics
for evaluating the answers corresponding to various types of questions, and (2)
the student LLM learns to follow the rubrics and perform self-correction from
the revision made by the teacher. We further iteratively carry out it to embody
the procedure of CITING. We compare CITING to a series of state-of-the-art
baselines on four datasets. Our method demonstrates strong improvement in terms
of articulate, in-depth, and comprehensive by GPT-4 evaluation. Specifically,
it achieves an average winning rate of 79.4% over SFT, 73.4% over RLHF, 78.1%
over RRHF, and 76.3% over RAFT, respectively.


# Interactive Code Generation via Test-Driven User-Intent Formalization

[Link to the paper](http://arxiv.org/abs/2208.05950v2)

## Authors
- Shuvendu K. Lahiri
- Sarah Fakhoury
- Aaditya Naik
- Georgios Sakkas
- Saikat Chakraborty
- Madanlal Musuvathi
- Piali Choudhury
- Curtis von Veh
- Jeevana Priya Inala
- Chenglong Wang
- Jianfeng Gao

## Summary
  Large language models (LLMs) have shown great potential in automating
significant aspects of coding by producing natural code from informal natural
language (NL) intent. However, when interacting with LLMs, users have no
guarantees that the code suggestions produced correctly satisfy the intent they
provided. In fact, it is hard to define a notion of correctness since natural
language can be ambiguous and lacks a formal semantics.
  In this paper, we propose the workflow of {\it interactive test-driven code
generation}, which leverages lightweight user feedback to (a) formalize the
user intent using generated tests that can be useful for debugging, and (b)
produce an improved set of code suggestions by pruning and ranking candidate
code suggestions. We describe a language-agnostic abstract algorithm and a
concrete implementation TiCoder. We perform an automated evaluation of TiCoder
on the \emph{MBPP} and \emph{HumanEval} code generation benchmarks. Our results
are promising with using the OpenAI Codex LLM: our best algorithm improves the
\passk{1} code generation accuracy (in absolute percentages) between $22.49\%$
to $37.71\%$ for MBPP and between $24.79\%$ to $53.98\%$ for HumanEval using
between 1 to 5 simulated user queries.


# RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems

[Link to the paper](http://arxiv.org/abs/2306.03091v2)

## Authors
- Tianyang Liu
- Canwen Xu
- Julian McAuley

## Summary
  Large Language Models (LLMs) have greatly advanced code auto-completion
systems, with a potential for substantial productivity enhancements for
developers. However, current benchmarks mainly focus on single-file tasks,
leaving an assessment gap for more complex, real-world, multi-file programming
scenarios. To fill this gap, we introduce RepoBench, a new benchmark
specifically designed for evaluating repository-level code auto-completion
systems. RepoBench supports both Python and Java and consists of three
interconnected evaluation tasks: RepoBench-R (Retrieval), RepoBench-C (Code
Completion), and RepoBench-P (Pipeline). Each task respectively measures the
system's ability to retrieve the most relevant code snippets from other files
as cross-file context, predict the next line of code with cross-file and
in-file context, and handle complex tasks that require a combination of both
retrieval and next-line prediction. RepoBench aims to facilitate a more
complete comparison of performance and encouraging continuous improvement in
auto-completion systems. RepoBench is publicly available at
https://github.com/Leolty/repobench.


# Large Language Models Can Be Good Privacy Protection Learners

[Link to the paper](http://arxiv.org/abs/2310.02469v1)

## Authors
- Yijia Xiao
- Yiqiao Jin
- Yushi Bai
- Yue Wu
- Xianjun Yang
- Xiao Luo
- Wenchao Yu
- Xujiang Zhao
- Yanchi Liu
- Haifeng Chen
- Wei Wang
- Wei Cheng

## Summary
  The proliferation of Large Language Models (LLMs) has driven considerable
interest in fine-tuning them with domain-specific data to create specialized
language models. Nevertheless, such domain-specific fine-tuning data often
contains sensitive personally identifiable information (PII). Direct
fine-tuning LLMs on this data without privacy protection poses a risk of
leakage. To address this challenge, we introduce Privacy Protection Language
Models (PPLM), a novel paradigm for fine-tuning LLMs that effectively injects
domain-specific knowledge while safeguarding data privacy. Our work offers a
theoretical analysis for model design and delves into various techniques such
as corpus curation, penalty-based unlikelihood in training loss, and
instruction-based tuning, etc. Extensive experiments across diverse datasets
and scenarios demonstrate the effectiveness of our approaches. In particular,
instruction tuning with both positive and negative examples, stands out as a
promising method, effectively protecting private data while enhancing the
model's knowledge. Our work underscores the potential for Large Language Models
as robust privacy protection learners.


# The Empty Signifier Problem: Towards Clearer Paradigms for Operationalising "Alignment" in Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.02457v1)

## Authors
- Hannah Rose Kirk
- Bertie Vidgen
- Paul Röttger
- Scott A. Hale

## Summary
  In this paper, we address the concept of "alignment" in large language models
(LLMs) through the lens of post-structuralist socio-political theory,
specifically examining its parallels to empty signifiers. To establish a shared
vocabulary around how abstract concepts of alignment are operationalised in
empirical datasets, we propose a framework that demarcates: 1) which dimensions
of model behaviour are considered important, then 2) how meanings and
definitions are ascribed to these dimensions, and by whom. We situate existing
empirical literature and provide guidance on deciding which paradigm to follow.
Through this framework, we aim to foster a culture of transparency and critical
evaluation, aiding the community in navigating the complexities of aligning
LLMs with human populations.


# Low-Resource Languages Jailbreak GPT-4

[Link to the paper](http://arxiv.org/abs/2310.02446v1)

## Authors
- Zheng-Xin Yong
- Cristina Menghini
- Stephen H. Bach

## Summary
  AI safety training and red-teaming of large language models (LLMs) are
measures to mitigate the generation of unsafe content. Our work exposes the
inherent cross-lingual vulnerability of these safety mechanisms, resulting from
the linguistic inequality of safety training data, by successfully
circumventing GPT-4's safeguard through translating unsafe English inputs into
low-resource languages. On the AdvBenchmark, GPT-4 engages with the unsafe
translated inputs and provides actionable items that can get the users towards
their harmful goals 79% of the time, which is on par with or even surpassing
state-of-the-art jailbreaking attacks. Other high-/mid-resource languages have
significantly lower attack success rate, which suggests that the cross-lingual
vulnerability mainly applies to low-resource languages. Previously, limited
training on low-resource languages primarily affects speakers of those
languages, causing technological disparities. However, our work highlights a
crucial shift: this deficiency now poses a risk to all LLMs users. Publicly
available translation APIs enable anyone to exploit LLMs' safety
vulnerabilities. Therefore, our work calls for a more holistic red-teaming
efforts to develop robust multilingual safeguards with wide language coverage.


# Novice Learner and Expert Tutor: Evaluating Math Reasoning Abilities of Large Language Models with Misconceptions

[Link to the paper](http://arxiv.org/abs/2310.02439v1)

## Authors
- Naiming Liu
- Shashank Sonkar
- Zichao Wang
- Simon Woodhead
- Richard G. Baraniuk

## Summary
  We propose novel evaluations for mathematical reasoning capabilities of Large
Language Models (LLMs) based on mathematical misconceptions. Our primary
approach is to simulate LLMs as a novice learner and an expert tutor, aiming to
identify the incorrect answer to math question resulted from a specific
misconception and to recognize the misconception(s) behind an incorrect answer,
respectively. Contrary to traditional LLMs-based mathematical evaluations that
focus on answering math questions correctly, our approach takes inspirations
from principles in educational learning sciences. We explicitly ask LLMs to
mimic a novice learner by answering questions in a specific incorrect manner
based on incomplete knowledge; and to mimic an expert tutor by identifying
misconception(s) corresponding to an incorrect answer to a question. Using
simple grade-school math problems, our experiments reveal that, while LLMs can
easily answer these questions correctly, they struggle to identify 1) the
incorrect answer corresponding to specific incomplete knowledge
(misconceptions); 2) the misconceptions that explain particular incorrect
answers. Our study indicates new opportunities for enhancing LLMs' math
reasoning capabilities, especially on developing robust student simulation and
expert tutoring models in the educational applications such as intelligent
tutoring systems.


# Can Large Language Models Provide Security & Privacy Advice? Measuring the Ability of LLMs to Refute Misconceptions

[Link to the paper](http://arxiv.org/abs/2310.02431v1)

## Authors
- Yufan Chen
- Arjun Arunasalam
- Z. Berkay Celik

## Summary
  Users seek security & privacy (S&P) advice from online resources, including
trusted websites and content-sharing platforms. These resources help users
understand S&P technologies and tools and suggest actionable strategies. Large
Language Models (LLMs) have recently emerged as trusted information sources.
However, their accuracy and correctness have been called into question. Prior
research has outlined the shortcomings of LLMs in answering multiple-choice
questions and user ability to inadvertently circumvent model restrictions
(e.g., to produce toxic content). Yet, the ability of LLMs to provide reliable
S&P advice is not well-explored. In this paper, we measure their ability to
refute popular S&P misconceptions that the general public holds. We first study
recent academic literature to curate a dataset of over a hundred S&P-related
misconceptions across six different topics. We then query two popular LLMs
(Bard and ChatGPT) and develop a labeling guide to evaluate their responses to
these misconceptions. To comprehensively evaluate their responses, we further
apply three strategies: query each misconception multiple times, generate and
query their paraphrases, and solicit source URLs of the responses. Both models
demonstrate, on average, a 21.3% non-negligible error rate, incorrectly
supporting popular S&P misconceptions. The error rate increases to 32.6% when
we repeatedly query LLMs with the same or paraphrased misconceptions. We also
expose that models may partially support a misconception or remain
noncommittal, refusing a firm stance on misconceptions. Our exploration of
information sources for responses revealed that LLMs are susceptible to
providing invalid URLs (21.2% for Bard and 67.7% for ChatGPT) or point to
unrelated sources (44.2% returned by Bard and 18.3% by ChatGPT).


# AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation

[Link to the paper](http://arxiv.org/abs/2308.08155v2)

## Authors
- Qingyun Wu
- Gagan Bansal
- Jieyu Zhang
- Yiran Wu
- Beibin Li
- Erkang Zhu
- Li Jiang
- Xiaoyun Zhang
- Shaokun Zhang
- Jiale Liu
- Ahmed Hassan Awadallah
- Ryen W White
- Doug Burger
- Chi Wang

## Summary
  AutoGen is an open-source framework that allows developers to build LLM
applications via multiple agents that can converse with each other to
accomplish tasks. AutoGen agents are customizable, conversable, and can operate
in various modes that employ combinations of LLMs, human inputs, and tools.
Using AutoGen, developers can also flexibly define agent interaction behaviors.
Both natural language and computer code can be used to program flexible
conversation patterns for different applications. AutoGen serves as a generic
infrastructure to build diverse applications of various complexities and LLM
capacities. Empirical studies demonstrate the effectiveness of the framework in
many example applications, with domains ranging from mathematics, coding,
question answering, operations research, online decision-making, entertainment,
etc.


# AXNav: Replaying Accessibility Tests from Natural Language

[Link to the paper](http://arxiv.org/abs/2310.02424v1)

## Authors
- Maryam Taeb
- Amanda Swearngin
- Eldon School
- Ruijia Cheng
- Yue Jiang
- Jeffrey Nichols

## Summary
  Developers and quality assurance testers often rely on manual testing to test
accessibility features throughout the product lifecycle. Unfortunately, manual
testing can be tedious, often has an overwhelming scope, and can be difficult
to schedule amongst other development milestones. Recently, Large Language
Models (LLMs) have been used for a variety of tasks including automation of
UIs, however to our knowledge no one has yet explored their use in controlling
assistive technologies for the purposes of supporting accessibility testing. In
this paper, we explore the requirements of a natural language based
accessibility testing workflow, starting with a formative study. From this we
build a system that takes as input a manual accessibility test (e.g., ``Search
for a show in VoiceOver'') and uses an LLM combined with pixel-based UI
Understanding models to execute the test and produce a chaptered, navigable
video. In each video, to help QA testers we apply heuristics to detect and flag
accessibility issues (e.g., Text size not increasing with Large Text enabled,
VoiceOver navigation loops). We evaluate this system through a 10 participant
user study with accessibility QA professionals who indicated that the tool
would be very useful in their current work and performed tests similarly to how
they would manually test the features. The study also reveals insights for
future work on using LLMs for accessibility testing.


# Jailbreaker in Jail: Moving Target Defense for Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.02417v1)

## Authors
- Bocheng Chen
- Advait Paliwal
- Qiben Yan

## Summary
  Large language models (LLMs), known for their capability in understanding and
following instructions, are vulnerable to adversarial attacks. Researchers have
found that current commercial LLMs either fail to be "harmless" by presenting
unethical answers, or fail to be "helpful" by refusing to offer meaningful
answers when faced with adversarial queries. To strike a balance between being
helpful and harmless, we design a moving target defense (MTD) enhanced LLM
system. The system aims to deliver non-toxic answers that align with outputs
from multiple model candidates, making them more robust against adversarial
attacks. We design a query and output analysis model to filter out unsafe or
non-responsive answers. %to achieve the two objectives of randomly selecting
outputs from different LLMs. We evaluate over 8 most recent chatbot models with
state-of-the-art adversarial queries. Our MTD-enhanced LLM system reduces the
attack success rate from 37.5\% to 0\%. Meanwhile, it decreases the response
refusal rate from 50\% to 0\%.


# Investigating the Catastrophic Forgetting in Multimodal Large Language Models

[Link to the paper](http://arxiv.org/abs/2309.10313v3)

## Authors
- Yuexiang Zhai
- Shengbang Tong
- Xiao Li
- Mu Cai
- Qing Qu
- Yong Jae Lee
- Yi Ma

## Summary
  Following the success of GPT4, there has been a surge in interest in
multimodal large language model (MLLM) research. This line of research focuses
on developing general-purpose LLMs through fine-tuning pre-trained LLMs and
vision models. However, catastrophic forgetting, a notorious phenomenon where
the fine-tuned model fails to retain similar performance compared to the
pre-trained model, still remains an inherent problem in multimodal LLMs (MLLM).
In this paper, we introduce EMT: Evaluating MulTimodality for evaluating the
catastrophic forgetting in MLLMs, by treating each MLLM as an image classifier.
We first apply EMT to evaluate several open-source fine-tuned MLLMs and we
discover that almost all evaluated MLLMs fail to retain the same performance
levels as their vision encoders on standard image classification tasks.
Moreover, we continue fine-tuning LLaVA, an MLLM and utilize EMT to assess
performance throughout the fine-tuning. Interestingly, our results suggest that
early-stage fine-tuning on an image dataset improves performance across other
image datasets, by enhancing the alignment of text and visual features.
However, as fine-tuning proceeds, the MLLMs begin to hallucinate, resulting in
a significant loss of generalizability, even when the image encoder remains
frozen. Our results suggest that MLLMs have yet to demonstrate performance on
par with their vision models on standard image classification tasks and the
current MLLM fine-tuning procedure still has room for improvement.


# Automated Bug Generation in the era of Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.02407v1)

## Authors
- Ali Reza Ibrahimzada
- Yang Chen
- Ryan Rong
- Reyhaneh Jabbarvand

## Summary
  Bugs are essential in software engineering; many research studies in the past
decades have been proposed to detect, localize, and repair bugs in software
systems. Effectiveness evaluation of such techniques requires complex bugs,
i.e., those that are hard to detect through testing and hard to repair through
debugging. From the classic software engineering point of view, a
hard-to-repair bug differs from the correct code in multiple locations, making
it hard to localize and repair. Hard-to-detect bugs, on the other hand,
manifest themselves under specific test inputs and reachability conditions.
These two objectives, i.e., generating hard-to-detect and hard-to-repair bugs,
are mostly aligned; a bug generation technique can change multiple statements
to be covered only under a specific set of inputs. However, these two
objectives are conflicting for learning-based techniques: A bug should have a
similar code representation to the correct code in the training data to
challenge a bug prediction model to distinguish them. The hard-to-repair bug
definition remains the same but with a caveat: the more a bug differs from the
original code (at multiple locations), the more distant their representations
are and easier to be detected. We propose BugFarm, to transform arbitrary code
into multiple complex bugs. BugFarm leverages LLMs to mutate code in multiple
locations (hard-to-repair). To ensure that multiple modifications do not
notably change the code representation, BugFarm analyzes the attention of the
underlying model and instructs LLMs to only change the least attended locations
(hard-to-detect). Our comprehensive evaluation of 320k+ bugs from over 2.5M
mutants generated by BugFarm and two alternative approaches demonstrates our
superiority in generating bugs that are hard to detect by learning-based bug
prediction approaches and hard to repair by SOTA learning-based program repair
technique.


# Deductive Verification of Chain-of-Thought Reasoning

[Link to the paper](http://arxiv.org/abs/2306.03872v3)

## Authors
- Zhan Ling
- Yunhao Fang
- Xuanlin Li
- Zhiao Huang
- Mingu Lee
- Roland Memisevic
- Hao Su

## Summary
  Large Language Models (LLMs) significantly benefit from Chain-of-Thought
(CoT) prompting in performing various reasoning tasks. While CoT allows models
to produce more comprehensive reasoning processes, its emphasis on intermediate
reasoning steps can inadvertently introduce hallucinations and accumulated
errors, thereby limiting models' ability to solve complex reasoning tasks.
Inspired by how humans engage in careful and meticulous deductive logical
reasoning processes to solve tasks, we seek to enable language models to
perform explicit and rigorous deductive reasoning, and also ensure the
trustworthiness of their reasoning process through self-verification. However,
directly verifying the validity of an entire deductive reasoning process is
challenging, even with advanced models like ChatGPT. In light of this, we
propose to decompose a reasoning verification process into a series of
step-by-step subprocesses, each only receiving their necessary context and
premises. To facilitate this procedure, we propose Natural Program, a natural
language-based deductive reasoning format. Our approach enables models to
generate precise reasoning steps where subsequent steps are more rigorously
grounded on prior steps. It also empowers language models to carry out
reasoning self-verification in a step-by-step manner. By integrating this
verification process into each deductive reasoning stage, we significantly
enhance the rigor and trustfulness of generated reasoning steps. Along this
process, we also improve the answer correctness on complex reasoning tasks.
Code will be released at https://github.com/lz1oceani/verify_cot.


# Conversational Health Agents: A Personalized LLM-Powered Agent Framework

[Link to the paper](http://arxiv.org/abs/2310.02374v1)

## Authors
- Mahyar Abbasian
- Iman Azimi
- Amir M. Rahmani
- Ramesh Jain

## Summary
  Conversational Health Agents (CHAs) are interactive systems designed to
enhance personal healthcare services by engaging in empathetic conversations
and processing multimodal data. While current CHAs, especially those utilizing
Large Language Models (LLMs), primarily focus on conversation, they often lack
comprehensive agent capabilities. This includes the ability to access personal
user health data from wearables, 24/7 data collection sources, and electronic
health records, as well as integrating the latest published health insights and
connecting with established multimodal data analysis tools. We are developing a
framework to empower CHAs by equipping them with critical thinking, knowledge
acquisition, and problem-solving abilities. Our CHA platform, powered by LLMs,
seamlessly integrates healthcare tools, enables multilingual and multimodal
conversations, and interfaces with a variety of user data analysis tools. We
illustrate its proficiency in handling complex healthcare tasks, such as stress
level estimation, showcasing the agent's cognitive and operational
capabilities.


# Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation

[Link to the paper](http://arxiv.org/abs/2310.02368v1)

## Authors
- Benjamin Steenhoek
- Michele Tufano
- Neel Sundaresan
- Alexey Svyatkovskiy

## Summary
  Software testing is a crucial aspect of software development, and the
creation of high-quality tests that adhere to best practices is essential for
effective maintenance. Recently, Large Language Models (LLMs) have gained
popularity for code generation, including the automated creation of test cases.
However, these LLMs are often trained on vast amounts of publicly available
code, which may include test cases that do not adhere to best practices and may
even contain test smells (anti-patterns). To address this issue, we propose a
novel technique called Reinforcement Learning from Static Quality Metrics
(RLSQM). To begin, we analyze the anti-patterns generated by the LLM and show
that LLMs can generate undesirable test smells. Thus, we train specific reward
models for each static quality metric, then utilize Proximal Policy
Optimization (PPO) to train models for optimizing a single quality metric at a
time. Furthermore, we amalgamate these rewards into a unified reward model
aimed at capturing different best practices and quality aspects of tests. By
comparing RL-trained models with those trained using supervised learning, we
provide insights into how reliably utilize RL to improve test generation
quality and into the effects of various training strategies. Our experimental
results demonstrate that the RL-optimized model consistently generated
high-quality test cases compared to the base LLM, improving the model by up to
21%, and successfully generates nearly 100% syntactically correct code. RLSQM
also outperformed GPT-4 on four out of seven metrics. This represents a
significant step towards enhancing the overall efficiency and reliability of
software testing through Reinforcement Learning and static quality metrics. Our
data are available at this link: https://figshare.com/s/ded476c8d4c221222849.


# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

[Link to the paper](http://arxiv.org/abs/2306.00978v2)

## Authors
- Ji Lin
- Jiaming Tang
- Haotian Tang
- Shang Yang
- Xingyu Dang
- Chuang Gan
- Song Han

## Summary
  Large language models (LLMs) have shown excellent performance on various
tasks, but the astronomical model size raises the hardware barrier for serving
(memory size) and slows down token generation (memory bandwidth). In this
paper, we propose Activation-aware Weight Quantization (AWQ), a
hardware-friendly approach for LLM low-bit weight-only quantization. Our method
is based on the observation that weights are not equally important: protecting
only 1% of salient weights can greatly reduce quantization error. We then
propose to search for the optimal per-channel scaling that protects the salient
weights by observing the activation, not weights. AWQ does not rely on any
backpropagation or reconstruction, so it can well preserve LLMs' generalization
ability on different domains and modalities, without overfitting to the
calibration set. AWQ outperforms existing work on various language modeling and
domain-specific benchmarks. Thanks to better generalization, it achieves
excellent quantization performance for instruction-tuned LMs and, for the first
time, multi-modal LMs. Alongside AWQ, we implement an efficient and flexible
inference framework tailored for LLMs on the edge, offering more than 3x
speedup over the Huggingface FP16 implementation on both desktop and mobile
GPUs. It also democratizes the deployment of the 70B Llama-2 model on mobile
GPU (NVIDIA Jetson Orin 64GB).


# REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction

[Link to the paper](http://arxiv.org/abs/2306.15724v3)

## Authors
- Zeyi Liu
- Arpit Bahety
- Shuran Song

## Summary
  The ability to detect and analyze failed executions automatically is crucial
for an explainable and robust robotic system. Recently, Large Language Models
(LLMs) have demonstrated strong reasoning abilities on textual inputs. To
leverage the power of LLMs for robot failure explanation, we introduce REFLECT,
a framework which queries LLM for failure reasoning based on a hierarchical
summary of robot past experiences generated from multisensory observations. The
failure explanation can further guide a language-based planner to correct the
failure and complete the task. To systematically evaluate the framework, we
create the RoboFail dataset with a variety of tasks and failure scenarios. We
demonstrate that the LLM-based framework is able to generate informative
failure explanations that assist successful correction planning.


# Contrastive Post-training Large Language Models on Data Curriculum

[Link to the paper](http://arxiv.org/abs/2310.02263v1)

## Authors
- Canwen Xu
- Corby Rosset
- Luciano Del Corro
- Shweti Mahajan
- Julian McAuley
- Jennifer Neville
- Ahmed Hassan Awadallah
- Nikhil Rao

## Summary
  Alignment serves as an important step to steer large language models (LLMs)
towards human preferences. In this paper, we explore contrastive post-training
techniques for alignment by automatically constructing preference pairs from
multiple models of varying strengths (e.g., InstructGPT, ChatGPT and GPT-4). We
carefully compare the contrastive techniques of SLiC and DPO to SFT baselines
and find that DPO provides a step-function improvement even after continueing
SFT saturates. We also explore a data curriculum learning scheme for
contrastive post-training, which starts by learning from "easier" pairs and
transitioning to "harder" ones, which further improves alignment. Finally, we
scale up our experiments to train with more data and larger models like Orca.
Remarkably, contrastive post-training further improves the performance of Orca,
already a state-of-the-art instruction learning model tuned with GPT-4 outputs,
to exceed that of ChatGPT.


# Generalizable Long-Horizon Manipulations with Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.02264v1)

## Authors
- Haoyu Zhou
- Mingyu Ding
- Weikun Peng
- Masayoshi Tomizuka
- Lin Shao
- Chuang Gan

## Summary
  This work introduces a framework harnessing the capabilities of Large
Language Models (LLMs) to generate primitive task conditions for generalizable
long-horizon manipulations with novel objects and unseen tasks. These task
conditions serve as guides for the generation and adjustment of Dynamic
Movement Primitives (DMP) trajectories for long-horizon task execution. We
further create a challenging robotic manipulation task suite based on Pybullet
for long-horizon task evaluation. Extensive experiments in both simulated and
real-world environments demonstrate the effectiveness of our framework on both
familiar tasks involving new objects and novel but related tasks, highlighting
the potential of LLMs in enhancing robotic system versatility and adaptability.
Project website: https://object814.github.io/Task-Condition-With-LLM/


# MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts

[Link to the paper](http://arxiv.org/abs/2310.02255v1)

## Authors
- Pan Lu
- Hritik Bansal
- Tony Xia
- Jiacheng Liu
- Chunyuan Li
- Hannaneh Hajishirzi
- Hao Cheng
- Kai-Wei Chang
- Michel Galley
- Jianfeng Gao

## Summary
  Although Large Language Models (LLMs) and Large Multimodal Models (LMMs)
exhibit impressive skills in various domains, their ability for mathematical
reasoning within visual contexts has not been formally examined. Equipping LLMs
and LMMs with this capability is vital for general-purpose AI assistants and
showcases promising potential in education, data analysis, and scientific
discovery. To bridge this gap, we present MathVista, a benchmark designed to
amalgamate challenges from diverse mathematical and visual tasks. We first
taxonomize the key task types, reasoning skills, and visual contexts from the
literature to guide our selection from 28 existing math-focused and visual
question answering datasets. Then, we construct three new datasets, IQTest,
FunctionQA, and PaperQA, to accommodate for missing types of visual contexts.
The problems featured often require deep visual understanding beyond OCR or
image captioning, and compositional reasoning with rich domain-specific tools,
thus posing a notable challenge to existing models. We conduct a comprehensive
evaluation of 11 prominent open-source and proprietary foundation models (LLMs,
LLMs augmented with tools, and LMMs), and early experiments with GPT-4V. The
best-performing model, Multimodal Bard, achieves only 58% of human performance
(34.8% vs 60.3%), indicating ample room for further improvement. Given this
significant gap, MathVista fuels future research in the development of
general-purpose AI agents capable of tackling mathematically intensive and
visually rich real-world tasks. Preliminary tests show that MathVista also
presents challenges to GPT-4V, underscoring the benchmark's importance. The
project is available at https://mathvista.github.io/.


# MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens

[Link to the paper](http://arxiv.org/abs/2310.02239v1)

## Authors
- Kaizhi Zheng
- Xuehai He
- Xin Eric Wang

## Summary
  Large Language Models (LLMs) have garnered significant attention for their
advancements in natural language processing, demonstrating unparalleled prowess
in text comprehension and generation. Yet, the simultaneous generation of
images with coherent textual narratives remains an evolving frontier. In
response, we introduce an innovative interleaved vision-and-language generation
technique anchored by the concept of "generative vokens," acting as the bridge
for harmonized image-text outputs. Our approach is characterized by a
distinctive two-staged training strategy focusing on description-free
multimodal generation, where the training requires no comprehensive
descriptions of images. To bolster model integrity, classifier-free guidance is
incorporated, enhancing the effectiveness of vokens on image generation. Our
model, MiniGPT-5, exhibits substantial improvement over the baseline Divter
model on the MMDialog dataset and consistently delivers superior or comparable
multimodal outputs in human evaluations on the VIST dataset, highlighting its
efficacy across diverse benchmarks.


# Extraction of Medication and Temporal Relation from Clinical Text by Harnessing Different Deep Learning Models

[Link to the paper](http://arxiv.org/abs/2310.02229v1)

## Authors
- Hangyu Tu
- Lifeng Han
- Goran Nenadic

## Summary
  Clinical texts, represented in electronic medical records (EMRs), contain
rich medical information and are essential for disease prediction, personalised
information recommendation, clinical decision support, and medication pattern
mining and measurement. Relation extractions between medication mentions and
temporal information can further help clinicians better understand the
patients' treatment history. To evaluate the performances of deep learning (DL)
and large language models (LLMs) in medication extraction and temporal
relations classification, we carry out an empirical investigation of
\textbf{MedTem} project using several advanced learning structures including
BiLSTM-CRF and CNN-BiLSTM for a clinical domain named entity recognition (NER),
and BERT-CNN for temporal relation extraction (RE), in addition to the
exploration of different word embedding techniques. Furthermore, we also
designed a set of post-processing roles to generate structured output on
medications and the temporal relation. Our experiments show that CNN-BiLSTM
slightly wins the BiLSTM-CRF model on the i2b2-2009 clinical NER task yielding
75.67, 77.83, and 78.17 for precision, recall, and F1 scores using Macro
Average. BERT-CNN model also produced reasonable evaluation scores 64.48,
67.17, and 65.03 for P/R/F1 using Macro Avg on the temporal relation extraction
test set from i2b2-2012 challenges. Code and Tools from MedTem will be hosted
at \url{https://github.com/HECTA-UoM/MedTem}


# Can Language Models be Instructed to Protect Personal Information?

[Link to the paper](http://arxiv.org/abs/2310.02224v1)

## Authors
- Yang Chen
- Ethan Mendes
- Sauvik Das
- Wei Xu
- Alan Ritter

## Summary
  Large multimodal language models have proven transformative in numerous
applications. However, these models have been shown to memorize and leak
pre-training data, raising serious user privacy and information security
concerns. While data leaks should be prevented, it is also crucial to examine
the trade-off between the privacy protection and model utility of proposed
approaches. In this paper, we introduce PrivQA -- a multimodal benchmark to
assess this privacy/utility trade-off when a model is instructed to protect
specific categories of personal information in a simulated scenario. We also
propose a technique to iteratively self-moderate responses, which significantly
improves privacy. However, through a series of red-teaming experiments, we find
that adversaries can also easily circumvent these protections with simple
jailbreaking methods through textual and/or image inputs. We believe PrivQA has
the potential to support the development of new models with improved privacy
protections, as well as the adversarial robustness of these protections. We
release the entire PrivQA dataset at https://llm-access-control.github.io/.


# Language Models Represent Space and Time

[Link to the paper](http://arxiv.org/abs/2310.02207v1)

## Authors
- Wes Gurnee
- Max Tegmark

## Summary
  The capabilities of large language models (LLMs) have sparked debate over
whether such systems just learn an enormous collection of superficial
statistics or a coherent model of the data generating process -- a world model.
We find evidence for the latter by analyzing the learned representations of
three spatial datasets (world, US, NYC places) and three temporal datasets
(historical figures, artworks, news headlines) in the Llama-2 family of models.
We discover that LLMs learn linear representations of space and time across
multiple scales. These representations are robust to prompting variations and
unified across different entity types (e.g. cities and landmarks). In addition,
we identify individual ``space neurons'' and ``time neurons'' that reliably
encode spatial and temporal coordinates. Our analysis demonstrates that modern
LLMs acquire structured knowledge about fundamental dimensions such as space
and time, supporting the view that they learn not merely superficial
statistics, but literal world models.


# Abusing Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs

[Link to the paper](http://arxiv.org/abs/2307.10490v4)

## Authors
- Eugene Bagdasaryan
- Tsung-Yin Hsieh
- Ben Nassi
- Vitaly Shmatikov

## Summary
  We demonstrate how images and sounds can be used for indirect prompt and
instruction injection in multi-modal LLMs. An attacker generates an adversarial
perturbation corresponding to the prompt and blends it into an image or audio
recording. When the user asks the (unmodified, benign) model about the
perturbed image or audio, the perturbation steers the model to output the
attacker-chosen text and/or make the subsequent dialog follow the attacker's
instruction. We illustrate this attack with several proof-of-concept examples
targeting LLaVa and PandaGPT.


# Ask Again, Then Fail: Large Language Models' Vacillations in Judgement

[Link to the paper](http://arxiv.org/abs/2310.02174v1)

## Authors
- Qiming Xie
- Zengzhi Wang
- Yi Feng
- Rui Xia

## Summary
  With the emergence of generative conversational large language models (LLMs)
like ChatGPT, serving as virtual assistants in various fields, the stability
and reliability of their responses have become crucial. However, during usage,
it has been observed that these models tend to waver in their judgements when
confronted with follow-up questions from users expressing skepticism or
disagreement. In this work, we draw inspiration from questioning strategies in
education and propose a \textsc{Follow-up Questioning Mechanism} along with two
evaluation metrics to assess the judgement consistency of LLMs before and after
exposure to disturbances. We evaluate the judgement consistency of ChatGPT,
PaLM2-Bison, and Vicuna-13B under this mechanism across eight reasoning
benchmarks. Empirical results show that even when the initial answers are
correct, judgement consistency sharply decreases when LLMs face disturbances
such as questioning, negation, or misleading. Additionally, we study these
models' judgement consistency under various settings (sampling temperature and
prompts) to validate this issue further, observing the impact of prompt tone
and conducting an in-depth error analysis for deeper behavioral insights.
Furthermore, we also explore several prompting methods to mitigate this issue
and demonstrate their
effectiveness\footnote{\url{https://github.com/NUSTM/LLMs-Waver-In-Judgements}}.


# Dynamic LLM-Agent Network: An LLM-agent Collaboration Framework with Agent Team Optimization

[Link to the paper](http://arxiv.org/abs/2310.02170v1)

## Authors
- Zijun Liu
- Yanzhe Zhang
- Peng Li
- Yang Liu
- Diyi Yang

## Summary
  Large language model (LLM) agents have been shown effective on a wide range
of tasks, and by ensembling multiple LLM agents, their performances could be
further improved. Existing approaches employ a fixed set of agents to interact
with each other in a static architecture, which limits their generalizability
to various tasks and requires strong human prior in designing these agents. In
this work, we propose to construct a strategic team of agents communicating in
a dynamic interaction architecture based on the task query. Specifically, we
build a framework named Dynamic LLM-Agent Network ($\textbf{DyLAN}$) for
LLM-agent collaboration on complicated tasks like reasoning and code
generation. DyLAN enables agents to interact for multiple rounds in a dynamic
architecture with inference-time agent selection and an early-stopping
mechanism to improve performance and efficiency. We further design an automatic
agent team optimization algorithm based on an unsupervised metric termed
$\textit{Agent Importance Score}$, enabling the selection of best agents based
on the contribution each agent makes. Empirically, we demonstrate that DyLAN
performs well in both reasoning and code generation tasks with reasonable
computational cost. DyLAN achieves 13.0% and 13.3% improvement on MATH and
HumanEval, respectively, compared to a single execution on GPT-35-turbo. On
specific subjects of MMLU, agent team optimization in DyLAN increases accuracy
by up to 25.0%.


# Editing Personality for LLMs

[Link to the paper](http://arxiv.org/abs/2310.02168v1)

## Authors
- Shengyu Mao
- Ningyu Zhang
- Xiaohan Wang
- Mengru Wang
- Yunzhi Yao
- Yong Jiang
- Pengjun Xie
- Fei Huang
- Huajun Chen

## Summary
  This paper introduces an innovative task focused on editing the personality
traits of Large Language Models (LLMs). This task seeks to adjust the models'
responses to opinion-related questions on specified topics since an
individual's personality often manifests in the form of their expressed
opinions, thereby showcasing different personality traits. Specifically, we
construct a new benchmark dataset PersonalityEdit to address this task. Drawing
on the theory in Social Psychology, we isolate three representative traits,
namely Neuroticism, Extraversion, and Agreeableness, as the foundation for our
benchmark. We then gather data using GPT-4, generating responses that not only
align with a specified topic but also embody the targeted personality trait. We
conduct comprehensive experiments involving various baselines and discuss the
representation of personality behavior in LLMs. Our intriguing findings uncover
potential challenges of the proposed task, illustrating several remaining
issues. We anticipate that our work can provide the NLP community with
insights. Code and datasets will be released at
https://github.com/zjunlp/EasyEdit.


# Selenite: Scaffolding Decision Making with Comprehensive Overviews Elicited from Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.02161v1)

## Authors
- Michael Xieyang Liu
- Tongshuang Wu
- Tianying Chen
- Franklin Mingzhe Li
- Aniket Kittur
- Brad A. Myers

## Summary
  Decision-making in unfamiliar domains can be challenging, demanding
considerable user effort to compare different options with respect to various
criteria. Prior research and our formative study found that people would
benefit from seeing an overview of the information space upfront, such as the
criteria that others have previously found useful. However, existing
sensemaking tools struggle with the "cold-start" problem -- it not only
requires significant input from previous users to generate and share these
overviews, but such overviews may also be biased and incomplete. In this work,
we introduce a novel system, Selenite, which leverages LLMs as reasoning
machines and knowledge retrievers to automatically produce a comprehensive
overview of options and criteria to jumpstart users' sensemaking processes.
Subsequently, Selenite also adapts as people use it, helping users find, read,
and navigate unfamiliar information in a systematic yet personalized manner.
Through three studies, we found that Selenite produced accurate and
high-quality overviews reliably, significantly accelerated users' information
processing, and effectively improved their overall comprehension and
sensemaking experience.


# Unveiling the Pitfalls of Knowledge Editing for Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.02129v1)

## Authors
- Zhoubo Li
- Ningyu Zhang
- Yunzhi Yao
- Mengru Wang
- Xi Chen
- Huajun Chen

## Summary
  As the cost associated with fine-tuning Large Language Models (LLMs)
continues to rise, recent research efforts have pivoted towards developing
methodologies to edit implicit knowledge embedded within LLMs. Yet, there's
still a dark cloud lingering overhead -- will knowledge editing trigger
butterfly effect? since it is still unclear whether knowledge editing might
introduce side effects that pose potential risks or not. This paper pioneers
the investigation into the potential pitfalls associated with knowledge editing
for LLMs. To achieve this, we introduce new benchmark datasets and propose
innovative evaluation metrics. Our results underline two pivotal concerns: (1)
Knowledge Conflict: Editing groups of facts that logically clash can magnify
the inherent inconsistencies in LLMs-a facet neglected by previous methods. (2)
Knowledge Distortion: Altering parameters with the aim of editing factual
knowledge can irrevocably warp the innate knowledge structure of LLMs.
Experimental results vividly demonstrate that knowledge editing might
inadvertently cast a shadow of unintended consequences on LLMs, which warrant
attention and efforts for future works. Code will be released at
https://github.com/zjunlp/PitfallsKnowledgeEditing.


# Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View

[Link to the paper](http://arxiv.org/abs/2310.02124v1)

## Authors
- Jintian Zhang
- Xin Xu
- Shumin Deng

## Summary
  As Natural Language Processing (NLP) systems are increasingly employed in
intricate social environments, a pressing query emerges: Can these NLP systems
mirror human-esque collaborative intelligence, in a multi-agent society
consisting of multiple large language models (LLMs)? This paper probes the
collaboration mechanisms among contemporary NLP systems by melding practical
experiments with theoretical insights. We fabricate four unique `societies'
comprised of LLM agents, where each agent is characterized by a specific
`trait' (easy-going or overconfident) and engages in collaboration with a
distinct `thinking pattern' (debate or reflection). Evaluating these
multi-agent societies on three benchmark datasets, we discern that LLM agents
navigate tasks by leveraging diverse social behaviors, from active debates to
introspective reflections. Notably, certain collaborative strategies only
optimize efficiency (using fewer API tokens), but also outshine previous
top-tier approaches. Moreover, our results further illustrate that LLM agents
manifest human-like social behaviors, such as conformity or majority rule,
mirroring foundational Social Psychology theories. In conclusion, we integrate
insights from Social Psychology to contextualize the collaboration of LLM
agents, inspiring further investigations into the collaboration mechanism for
LLMs. We commit to sharing our code and datasets (already submitted in
supplementary materials), hoping to catalyze further research in this promising
avenue (All code and data are available at
\url{https://github.com/zjunlp/MachineSoM}.).


# TWIZ: The Wizard of Multimodal Conversational-Stimulus

[Link to the paper](http://arxiv.org/abs/2310.02118v1)

## Authors
- Rafael Ferreira
- Diogo Tavares
- Diogo Silva
- Rodrigo Valério
- João Bordalo
- Inês Simões
- Vasco Ramos
- David Semedo
- João Magalhães

## Summary
  In this report, we describe the vision, challenges, and scientific
contributions of the Task Wizard team, TWIZ, in the Alexa Prize TaskBot
Challenge 2022. Our vision, is to build TWIZ bot as an helpful, multimodal,
knowledgeable, and engaging assistant that can guide users towards the
successful completion of complex manual tasks. To achieve this, we focus our
efforts on three main research questions: (1) Humanly-Shaped Conversations, by
providing information in a knowledgeable way; (2) Multimodal Stimulus, making
use of various modalities including voice, images, and videos; and (3)
Zero-shot Conversational Flows, to improve the robustness of the interaction to
unseen scenarios. TWIZ is an assistant capable of supporting a wide range of
tasks, with several innovative features such as creative cooking, video
navigation through voice, and the robust TWIZ-LLM, a Large Language Model
trained for dialoguing about complex manual tasks. Given ratings and feedback
provided by users, we observed that TWIZ bot is an effective and robust system,
capable of guiding users through tasks while providing several multimodal
stimuli.


# Instance Needs More Care: Rewriting Prompts for Instances Yields Better Zero-Shot Performance

[Link to the paper](http://arxiv.org/abs/2310.02107v1)

## Authors
- Saurabh Srivastava
- Chengyue Huang
- Weiguo Fan
- Ziyu Yao

## Summary
  Enabling large language models (LLMs) to perform tasks in zero-shot has been
an appealing goal owing to its labor-saving (i.e., requiring no task-specific
annotations); as such, zero-shot prompting approaches also enjoy better task
generalizability. To improve LLMs' zero-shot performance, prior work has
focused on devising more effective task instructions (e.g., ``let's think step
by step'' ). However, we argue that, in order for an LLM to solve them
correctly in zero-shot, individual test instances need more carefully designed
and customized instructions. To this end, we propose PRoMPTd, an approach that
rewrites the task prompt for each individual test input to be more specific,
unambiguous, and complete, so as to provide better guidance to the task LLM. We
evaluated PRoMPTd on eight datasets covering tasks including arithmetics,
logical reasoning, and code generation, using GPT-4 as the task LLM. Notably,
\algoname achieves an absolute improvement of around 10\% on the complex MATH
dataset and 5\% on the code generation task on HumanEval, outperforming
conventional zero-shot methods. In addition, we also showed that the rewritten
prompt can provide better interpretability of how the LLM resolves each test
instance, which can potentially be leveraged as a defense mechanism against
adversarial prompting. The source code and dataset can be obtained from
https://github.com/salokr/PRoMPTd


# ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs

[Link to the paper](http://arxiv.org/abs/2307.16789v2)

## Authors
- Yujia Qin
- Shihao Liang
- Yining Ye
- Kunlun Zhu
- Lan Yan
- Yaxi Lu
- Yankai Lin
- Xin Cong
- Xiangru Tang
- Bill Qian
- Sihan Zhao
- Lauren Hong
- Runchu Tian
- Ruobing Xie
- Jie Zhou
- Mark Gerstein
- Dahai Li
- Zhiyuan Liu
- Maosong Sun

## Summary
  Despite the advancements of open-source large language models (LLMs), e.g.,
LLaMA, they remain significantly limited in tool-use capabilities, i.e., using
external tools (APIs) to fulfill human instructions. The reason is that current
instruction tuning largely focuses on basic language tasks but ignores the
tool-use domain. This is in contrast to the excellent tool-use capabilities of
state-of-the-art (SOTA) closed-source LLMs, e.g., ChatGPT. To bridge this gap,
we introduce ToolLLM, a general tool-use framework encompassing data
construction, model training, and evaluation. We first present ToolBench, an
instruction-tuning dataset for tool use, which is constructed automatically
using ChatGPT. Specifically, the construction can be divided into three stages:
(i) API collection: we collect 16,464 real-world RESTful APIs spanning 49
categories from RapidAPI Hub; (ii) instruction generation: we prompt ChatGPT to
generate diverse instructions involving these APIs, covering both single-tool
and multi-tool scenarios; (iii) solution path annotation: we use ChatGPT to
search for a valid solution path (chain of API calls) for each instruction. To
enhance the reasoning capabilities of LLMs, we develop a novel depth-first
search-based decision tree algorithm. It enables LLMs to evaluate multiple
reasoning traces and expand the search space. Moreover, to evaluate the
tool-use capabilities of LLMs, we develop an automatic evaluator: ToolEval.
Based on ToolBench, we fine-tune LLaMA to obtain an LLM ToolLLaMA, and equip it
with a neural API retriever to recommend appropriate APIs for each instruction.
Experiments show that ToolLLaMA demonstrates a remarkable ability to execute
complex instructions and generalize to unseen APIs, and exhibits comparable
performance to ChatGPT. Our ToolLLaMA also demonstrates strong zero-shot
generalization ability in an out-of-distribution tool-use dataset: APIBench.


# Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond

[Link to the paper](http://arxiv.org/abs/2310.02071v1)

## Authors
- Liang Chen
- Yichi Zhang
- Shuhuai Ren
- Haozhe Zhao
- Zefan Cai
- Yuchi Wang
- Tianyu Liu
- Baobao Chang

## Summary
  In this study, we explore the potential of Multimodal Large Language Models
(MLLMs) in improving embodied decision-making processes for agents. While Large
Language Models (LLMs) have been widely used due to their advanced reasoning
skills and vast world knowledge, MLLMs like GPT4-Vision offer enhanced visual
understanding and reasoning capabilities. We investigate whether
state-of-the-art MLLMs can handle embodied decision-making in an end-to-end
manner and whether collaborations between LLMs and MLLMs can enhance
decision-making. To address these questions, we introduce a new benchmark
called PCA-EVAL, which evaluates embodied decision-making from the perspectives
of Perception, Cognition, and Action. Additionally, we propose HOLMES, a
multi-agent cooperation framework that allows LLMs to leverage MLLMs and APIs
to gather multimodal information for informed decision-making. We compare
end-to-end embodied decision-making and HOLMES on our benchmark and find that
the GPT4-Vision model demonstrates strong end-to-end embodied decision-making
abilities, outperforming GPT4-HOLMES in terms of average decision accuracy
(+3%). However, this performance is exclusive to the latest GPT4-Vision model,
surpassing the open-source state-of-the-art MLLM by 26%. Our results indicate
that powerful MLLMs like GPT4-Vision hold promise for decision-making in
embodied agents, offering new avenues for MLLM research.


# Security Weaknesses of Copilot Generated Code in GitHub

[Link to the paper](http://arxiv.org/abs/2310.02059v1)

## Authors
- Yujia Fu
- Peng Liang
- Amjed Tahir
- Zengyang Li
- Mojtaba Shahin
- Jiaxin Yu

## Summary
  Modern code generation tools use AI models, particularly Large Language
Models (LLMs), to generate functional and complete code. While such tools are
becoming popular and widely available for developers, using these tools is
often accompanied by security challenges. Therefore, it is important to assess
the quality of the generated code, especially in terms of its security.
Researchers have recently explored various aspects of code generation tools,
including security. However, many open questions about the security of the
generated code require further investigation, especially the security issues of
automatically generated code in the wild. To this end, we conducted an
empirical study by analyzing the security weaknesses in code snippets generated
by GitHub Copilot that are found as part of publicly available projects hosted
on GitHub. The goal is to investigate the types of security issues and their
scale in real-world scenarios (rather than crafted scenarios). To this end, we
identified 435 code snippets generated by Copilot from publicly available
projects. We then conducted extensive security analysis to identify Common
Weakness Enumeration (CWE) instances in these code snippets. The results show
that (1) 35.8% of Copilot generated code snippets contain CWEs, and those
issues are spread across multiple languages, (2) the security weaknesses are
diverse and related to 42 different CWEs, in which CWE-78: OS Command
Injection, CWE-330: Use of Insufficiently Random Values, and CWE-703: Improper
Check or Handling of Exceptional Conditions occurred the most frequently, and
(3) among the 42 CWEs identified, 11 of those belong to the currently
recognized 2022 CWE Top-25. Our findings confirm that developers should be
careful when adding code generated by Copilot (and similar AI code generation
tools) and should also run appropriate security checks as they accept the
suggested code.


# TRAM: Benchmarking Temporal Reasoning for Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.00835v2)

## Authors
- Yuqing Wang
- Yun Zhao

## Summary
  Reasoning about time is essential for understanding the nuances of events
described in natural language. Previous research on this topic has been limited
in scope, characterized by a lack of standardized benchmarks that would allow
for consistent evaluations across different studies. In this paper, we
introduce TRAM, a temporal reasoning benchmark composed of ten datasets,
encompassing various temporal aspects of events such as order, arithmetic,
frequency, and duration, designed to facilitate a comprehensive evaluation of
the temporal reasoning capabilities of large language models (LLMs). We conduct
an extensive evaluation using popular LLMs, such as GPT-4 and Llama2, in both
zero-shot and few-shot learning scenarios. Additionally, we employ BERT-based
models to establish the baseline evaluations. Our findings indicate that these
models still trail human performance in temporal reasoning tasks. It is our
aspiration that TRAM will spur further progress in enhancing the temporal
reasoning abilities of LLMs.


# Tuning Large language model for End-to-end Speech Translation

[Link to the paper](http://arxiv.org/abs/2310.02050v1)

## Authors
- Hao Zhang
- Nianwen Si
- Yaqi Chen
- Wenlin Zhang
- Xukui Yang
- Dan Qu
- Xiaolin Jiao

## Summary
  With the emergence of large language models (LLMs), multimodal models based
on LLMs have demonstrated significant potential. Models such as LLaSM, X-LLM,
and SpeechGPT exhibit an impressive ability to comprehend and generate human
instructions. However, their performance often falters when faced with complex
tasks like end-to-end speech translation (E2E-ST), a cross-language and
cross-modal translation task. In comparison to single-modal models, multimodal
models lag behind in these scenarios. This paper introduces LST, a Large
multimodal model designed to excel at the E2E-ST task. LST consists of a speech
frontend, an adapter, and a LLM backend. The training of LST consists of two
stages: (1) Modality adjustment, where the adapter is tuned to align speech
representation with text embedding space, and (2) Downstream task fine-tuning,
where both the adapter and LLM model are trained to optimize performance on the
E2EST task. Experimental results on the MuST-C speech translation benchmark
demonstrate that LST-13B achieves BLEU scores of 30.39/41.55/35.33 on
En-De/En-Fr/En-Es language pairs, surpassing previous models and establishing a
new state-of-the-art. Additionally, we conduct an in-depth analysis of
single-modal model selection and the impact of training strategies, which lays
the foundation for future research. We will open up our code and models after
review.


# Improving web element localization by using a large language model

[Link to the paper](http://arxiv.org/abs/2310.02046v1)

## Authors
- Michel Nass
- Emil Alegroth
- Robert Feldt

## Summary
  Web-based test automation heavily relies on accurately finding web elements.
Traditional methods compare attributes but don't grasp the context and meaning
of elements and words. The emergence of Large Language Models (LLMs) like
GPT-4, which can show human-like reasoning abilities on some tasks, offers new
opportunities for software engineering and web element localization. This paper
introduces and evaluates VON Similo LLM, an enhanced web element localization
approach. Using an LLM, it selects the most likely web element from the
top-ranked ones identified by the existing VON Similo method, ideally aiming to
get closer to human-like selection accuracy. An experimental study was
conducted using 804 web element pairs from 48 real-world web applications. We
measured the number of correctly identified elements as well as the execution
times, comparing the effectiveness and efficiency of VON Similo LLM against the
baseline algorithm. In addition, motivations from the LLM were recorded and
analyzed for all instances where the original approach failed to find the right
web element. VON Similo LLM demonstrated improved performance, reducing failed
localizations from 70 to 39 (out of 804), a 44 percent reduction. Despite its
slower execution time and additional costs of using the GPT-4 model, the LLMs
human-like reasoning showed promise in enhancing web element localization. LLM
technology can enhance web element identification in GUI test automation,
reducing false positives and potentially lowering maintenance costs. However,
further research is necessary to fully understand LLMs capabilities,
limitations, and practical use in GUI testing.


# LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning

[Link to the paper](http://arxiv.org/abs/2305.18403v3)

## Authors
- Mingyang Zhang
- Hao Chen
- Chunhua Shen
- Zhen Yang
- Linlin Ou
- Xinyi Yu
- Bohan Zhuang

## Summary
  Large pre-trained models (LPMs), such as LLaMA and GLM, have shown
exceptional performance across various tasks through fine-tuning. Although
low-rank adaption (LoRA) has emerged to cheaply fine-tune these LPMs on
downstream tasks, their deployment is still hindered by the vast model scale
and computational costs. Neural network pruning offers a way to compress LPMs.
However, the current pruning methods designed for LPMs are not compatible with
LoRA. This is due to their utilization of unstructured pruning on LPMs,
impeding the merging of LoRA weights, or their dependence on the gradients of
pre-trained weights to guide pruning, which can impose significant memory
overhead. To this end, we propose LoRAPrune, a new framework that delivers an
accurate, compact model for efficient inference in a highly memory-effective
manner. Specifically, we first design a LoRA-guided pruning criterion, which
uses the weights and gradients of LoRA, rather than the gradients of
pre-trained weights for importance estimation. We then propose a structured
iterative pruning procedure, to remove redundant channels and heads. Extensive
experimental results demonstrate the superior performance of our LoRAPrune over
existing approaches on the LLaMA series models. For instance, at a 50\%
compression rate, LoRAPrune outperforms LLM-Pruner by a perplexity reduction of
8.0 on WikiText2 and 16.05 on PTB datasets, while concurrently reducing memory
usage by 52.6\%. The code will be released after review


# Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources

[Link to the paper](http://arxiv.org/abs/2305.13269v2)

## Authors
- Xingxuan Li
- Ruochen Zhao
- Yew Ken Chia
- Bosheng Ding
- Shafiq Joty
- Soujanya Poria
- Lidong Bing

## Summary
  We present chain-of-knowledge (CoK), a novel framework that augments large
language models (LLMs) by dynamically incorporating grounding information from
heterogeneous sources. It results in more factual rationales and reduced
hallucination in generation. Specifically, CoK consists of three stages:
reasoning preparation, dynamic knowledge adapting, and answer consolidation.
Given a knowledge-intensive question, CoK first prepares several preliminary
rationales and answers while identifying the relevant knowledge domains. If
there is no majority consensus among the answers from samples, CoK corrects the
rationales step by step by adapting knowledge from the identified domains.
These corrected rationales can plausibly serve as a better foundation for the
final answer consolidation. Unlike prior studies that primarily use
unstructured data, CoK also leverages structured knowledge sources such as
Wikidata and tables that provide more reliable factual information. To access
both unstructured and structured knowledge sources in the dynamic knowledge
adapting stage, we propose an adaptive query generator that allows the
generation of queries for various types of query languages, including SPARQL,
SQL, and natural sentences. Moreover, to minimize error propagation between
rationales, CoK corrects the rationales progressively using preceding corrected
rationales to generate and correct subsequent rationales. Extensive experiments
show that CoK consistently improves the performance of LLMs on
knowledge-intensive tasks across different domains.


# Fill in the Blank: Exploring and Enhancing LLM Capabilities for Backward Reasoning in Math Word Problems

[Link to the paper](http://arxiv.org/abs/2310.01991v1)

## Authors
- Aniruddha Deb
- Neeva Oza
- Sarthak Singla
- Dinesh Khandelwal
- Dinesh Garg
- Parag Singla

## Summary
  While forward reasoning (i.e. find the answer given the question) has been
explored extensively in the recent literature, backward reasoning is relatively
unexplored. We examine the backward reasoning capabilities of LLMs on Math Word
Problems (MWPs): given a mathematical question and its answer, with some
details omitted from the question, can LLMs effectively retrieve the missing
information?
  In this paper, we formally define the backward reasoning task on math word
problems and modify three datasets to evaluate this task: GSM8k, SVAMP and
MultiArith. Our findings show a significant drop in the accuracy of models on
backward reasoning compared to forward reasoning across four SOTA LLMs (GPT4,
GPT3.5, PaLM-2, and LLaMa-2). Utilizing the specific format of this task, we
propose three novel techniques that improve performance: Rephrase reformulates
the given problem into a forward reasoning problem, PAL-Tools combines the idea
of Program-Aided LLMs to produce a set of equations that can be solved by an
external solver, and Check your Work exploits the availability of natural
verifier of high accuracy in the forward direction, interleaving solving and
verification steps. Finally, realizing that each of our base methods correctly
solves a different set of problems, we propose a novel Bayesian formulation for
creating an ensemble over these base methods aided by a verifier to further
boost the accuracy by a significant margin. Extensive experimentation
demonstrates that our techniques successively improve the performance of LLMs
on the backward reasoning task, with the final ensemble-based method resulting
in a substantial performance gain compared to the raw LLMs with standard
prompting techniques such as chain-of-thought.


# Language Models as Knowledge Bases for Visual Word Sense Disambiguation

[Link to the paper](http://arxiv.org/abs/2310.01960v1)

## Authors
- Anastasia Kritharoula
- Maria Lymperaiou
- Giorgos Stamou

## Summary
  Visual Word Sense Disambiguation (VWSD) is a novel challenging task that lies
between linguistic sense disambiguation and fine-grained multimodal retrieval.
The recent advancements in the development of visiolinguistic (VL) transformers
suggest some off-the-self implementations with encouraging results, which
however we argue that can be further improved. To this end, we propose some
knowledge-enhancement techniques towards improving the retrieval performance of
VL transformers via the usage of Large Language Models (LLMs) as Knowledge
Bases. More specifically, knowledge stored in LLMs is retrieved with the help
of appropriate prompts in a zero-shot manner, achieving performance
advancements. Moreover, we convert VWSD to a purely textual question-answering
(QA) problem by considering generated image captions as multiple-choice
candidate answers. Zero-shot and few-shot prompting strategies are leveraged to
explore the potential of such a transformation, while Chain-of-Thought (CoT)
prompting in the zero-shot setting is able to reveal the internal reasoning
steps an LLM follows to select the appropriate candidate. In total, our
presented approach is the first one to analyze the merits of exploiting
knowledge stored in LLMs in different ways to solve WVSD.


# Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving

[Link to the paper](http://arxiv.org/abs/2310.01957v1)

## Authors
- Long Chen
- Oleg Sinavski
- Jan Hünermann
- Alice Karnsund
- Andrew James Willmott
- Danny Birch
- Daniel Maund
- Jamie Shotton

## Summary
  Large Language Models (LLMs) have shown promise in the autonomous driving
sector, particularly in generalization and interpretability. We introduce a
unique object-level multimodal LLM architecture that merges vectorized numeric
modalities with a pre-trained LLM to improve context understanding in driving
situations. We also present a new dataset of 160k QA pairs derived from 10k
driving scenarios, paired with high quality control commands collected with RL
agent and question answer pairs generated by teacher LLM (GPT-3.5). A distinct
pretraining strategy is devised to align numeric vector modalities with static
LLM representations using vector captioning language data. We also introduce an
evaluation metric for Driving QA and demonstrate our LLM-driver's proficiency
in interpreting driving scenarios, answering questions, and decision-making.
Our findings highlight the potential of LLM-based driving action generation in
comparison to traditional behavioral cloning. We make our benchmark, datasets,
and model available for further exploration.


# In-Context Learning Learns Label Relationships but Is Not Conventional Learning

[Link to the paper](http://arxiv.org/abs/2307.12375v3)

## Authors
- Jannik Kossen
- Yarin Gal
- Tom Rainforth

## Summary
  The predictions of Large Language Models (LLMs) on downstream tasks often
improve significantly when including examples of the input--label relationship
in the context. However, there is currently no consensus about how this
in-context learning (ICL) ability of LLMs works. For example, while Xie et al.
(2021) liken ICL to a general-purpose learning algorithm, Min et al. (2022)
argue ICL does not even learn label relationships from in-context examples. In
this paper, we provide novel insights into how ICL leverages label information,
revealing both capabilities and limitations. To ensure we obtain a
comprehensive picture of ICL behavior, we study probabilistic aspects of ICL
predictions and thoroughly examine the dynamics of ICL as more examples are
provided. Our experiments show that ICL predictions almost always depend on
in-context labels, and that ICL can learn truly novel tasks in-context.
However, we also find that ICL struggles to fully overcome prediction
preferences acquired from pre-training data, and, further, that ICL does not
consider all in-context information equally.


# DeepDecipher: Accessing and Investigating Neuron Activation in Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.01870v1)

## Authors
- Albert Garde
- Esben Kran
- Fazl Barez

## Summary
  As large language models (LLMs) become more capable, there is an urgent need
for interpretable and transparent tools. Current methods are difficult to
implement, and accessible tools to analyze model internals are lacking. To
bridge this gap, we present DeepDecipher - an API and interface for probing
neurons in transformer models' MLP layers. DeepDecipher makes the outputs of
advanced interpretability techniques for LLMs readily available. The
easy-to-use interface also makes inspecting these complex models more
intuitive. This paper outlines DeepDecipher's design and capabilities. We
demonstrate how to analyze neurons, compare models, and gain insights into
model behavior. For example, we contrast DeepDecipher's functionality with
similar tools like Neuroscope and OpenAI's Neuron Explainer. DeepDecipher
enables efficient, scalable analysis of LLMs. By granting access to
state-of-the-art interpretability methods, DeepDecipher makes LLMs more
transparent, trustworthy, and safe. Researchers, engineers, and developers can
quickly diagnose issues, audit systems, and advance the field.


# Formalizing Natural Language Intent into Program Specifications via Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.01831v1)

## Authors
- Madeline Endres
- Sarah Fakhoury
- Saikat Chakraborty
- Shuvendu K. Lahiri

## Summary
  Informal natural language that describes code functionality, such as code
comments or function documentation, may contain substantial information about a
programs intent. However, there is typically no guarantee that a programs
implementation and natural language documentation are aligned. In the case of a
conflict, leveraging information in code-adjacent natural language has the
potential to enhance fault localization, debugging, and code trustworthiness.
In practice, however, this information is often underutilized due to the
inherent ambiguity of natural language which makes natural language intent
challenging to check programmatically. The "emergent abilities" of Large
Language Models (LLMs) have the potential to facilitate the translation of
natural language intent to programmatically checkable assertions. However, it
is unclear if LLMs can correctly translate informal natural language
specifications into formal specifications that match programmer intent.
Additionally, it is unclear if such translation could be useful in practice. In
this paper, we describe LLM4nl2post, the problem leveraging LLMs for
transforming informal natural language to formal method postconditions,
expressed as program assertions. We introduce and validate metrics to measure
and compare different LLM4nl2post approaches, using the correctness and
discriminative power of generated postconditions. We then perform qualitative
and quantitative methods to assess the quality of LLM4nl2post postconditions,
finding that they are generally correct and able to discriminate incorrect
code. Finally, we find that LLM4nl2post via LLMs has the potential to be
helpful in practice; specifications generated from natural language were able
to catch 70 real-world historical bugs from Defects4J.


# Data Race Detection Using Large Language Models

[Link to the paper](http://arxiv.org/abs/2308.07505v2)

## Authors
- Le Chen
- Xianzhong Ding
- Murali Emani
- Tristan Vanderbruggen
- Pei-hung Lin
- Chuanhua Liao

## Summary
  Large language models (LLMs) are demonstrating significant promise as an
alternate strategy to facilitate analyses and optimizations of high-performance
computing programs, circumventing the need for resource-intensive manual tool
creation. In this paper, we explore a novel LLM-based data race detection
approach combining prompting engineering and fine-tuning techniques. We create
a dedicated dataset named DRB-ML, which is derived from DataRaceBench, with
fine-grain labels showing the presence of data race pairs and their associated
variables, line numbers, and read/write information. DRB-ML is then used to
evaluate representative LLMs and fine-tune open-source ones. Our experiment
shows that LLMs can be a viable approach to data race detection. However, they
still cannot compete with traditional data race detection tools when we need
detailed information about variable pairs causing data races.


# Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs

[Link to the paper](http://arxiv.org/abs/2310.01801v1)

## Authors
- Suyu Ge
- Yunan Zhang
- Liyuan Liu
- Minjia Zhang
- Jiawei Han
- Jianfeng Gao

## Summary
  In this study, we introduce adaptive KV cache compression, a plug-and-play
method that reduces the memory footprint of generative inference for Large
Language Models (LLMs). Different from the conventional KV cache that retains
key and value vectors for all context tokens, we conduct targeted profiling to
discern the intrinsic structure of attention modules. Based on the recognized
structure, we then construct the KV cache in an adaptive manner: evicting
long-range contexts on attention heads emphasizing local contexts, discarding
non-special tokens on attention heads centered on special tokens, and only
employing the standard KV cache for attention heads that broadly attend to all
tokens. Moreover, with the lightweight attention profiling used to guide the
construction of the adaptive KV cache, FastGen can be deployed without
resource-intensive fine-tuning or re-training. In our experiments across
various asks, FastGen demonstrates substantial reduction on GPU memory
consumption with negligible generation quality loss. We will release our code
and the compatible CUDA kernel for reproducibility.


# Intuitive or Dependent? Investigating LLMs' Robustness to Conflicting Prompts

[Link to the paper](http://arxiv.org/abs/2309.17415v2)

## Authors
- Jiahao Ying
- Yixin Cao
- Kai Xiong
- Yidong He
- Long Cui
- Yongbin Liu

## Summary
  This paper explores the robustness of LLMs' preference to their internal
memory or the given prompt, which may contain contrasting information in
real-world applications due to noise or task settings. To this end, we
establish a quantitative benchmarking framework and conduct the role playing
intervention to control LLMs' preference. In specific, we define two types of
robustness, factual robustness targeting the ability to identify the correct
fact from prompts or memory, and decision style to categorize LLMs' behavior in
making consistent choices -- assuming there is no definitive "right" answer --
intuitive, dependent, or rational based on cognitive theory. Our findings,
derived from extensive experiments on seven open-source and closed-source LLMs,
reveal that these models are highly susceptible to misleading prompts,
especially for instructing commonsense knowledge. While detailed instructions
can mitigate the selection of misleading answers, they also increase the
incidence of invalid responses. After Unraveling the preference, we intervene
different sized LLMs through specific style of role instruction, showing their
varying upper bound of robustness and adaptivity.


# Large Language Models Cannot Self-Correct Reasoning Yet

[Link to the paper](http://arxiv.org/abs/2310.01798v1)

## Authors
- Jie Huang
- Xinyun Chen
- Swaroop Mishra
- Huaixiu Steven Zheng
- Adams Wei Yu
- Xinying Song
- Denny Zhou

## Summary
  Large Language Models (LLMs) have emerged as a groundbreaking technology with
their unparalleled text generation capabilities across various applications.
Nevertheless, concerns persist regarding the accuracy and appropriateness of
their generated content. A contemporary methodology, self-correction, has been
proposed as a remedy to these issues. Building upon this premise, this paper
critically examines the role and efficacy of self-correction within LLMs,
shedding light on its true potential and limitations. Central to our
investigation is the notion of intrinsic self-correction, whereby an LLM
attempts to correct its initial responses based solely on its inherent
capabilities, without the crutch of external feedback. In the context of
reasoning, our research indicates that LLMs struggle to self-correct their
responses without external feedback, and at times, their performance might even
degrade post self-correction. Drawing from these insights, we offer suggestions
for future research and practical applications in this field.


# LLMParser: A LLM-based Log Parsing Framework

[Link to the paper](http://arxiv.org/abs/2310.01796v1)

## Authors
- Zhihan Jiang
- Jinyang Liu
- Zhuangbin Chen
- Yichen Li
- Junjie Huang
- Yintong Huo
- Pinjia He
- Jiazhen Gu
- Michael R. Lyu

## Summary
  The process of log parsing, which converts log messages into structured
formats, is a crucial step for various log analysis tasks. Although numerous
log parsers have been proposed, their effectiveness on complex log data is
often hindered due to reliance on human-made rules or learning-based models
with limited training data. The recent rise of powerful large language models
(LLMs) shows potential for log parsing due to their extensive pre-trained
knowledge related to code and logging. However, their accuracy is currently
limited due to the lack of specialized log parsing capabilities. Additionally,
the inconsistency of their answers and significant overhead obstruct the
practical implementation of LLM-based log parsing.
  To tackle these challenges, we introduce LLMParser, the first practical
LLM-based log parsing framework. LLMParser enables accurate and robust log
parsing by leveraging the in-context learning (ICL) capability of the LLM,
employing a hierarchical candidate sampling algorithm, and selecting
high-quality demonstrations. LLMParser also includes a novel adaptive parsing
cache component to store and refine the templates generated by the LLM. This
design aids in addressing the inefficiency of LLMs by rapid matching to
previously parsed log templates. LLMParser also adaptively updates the
templates in the parsing cache to ensure consistent parsed results. Extensive
evaluation on large-scale public datasets demonstrates that LLMParser surpasses
the state-of-the-art methods. Furthermore, LLMParser significantly reduces the
query times to LLMs, achieving efficiency comparable to the most efficient
baseline, Drain.


# Can large language models provide useful feedback on research papers? A large-scale empirical analysis

[Link to the paper](http://arxiv.org/abs/2310.01783v1)

## Authors
- Weixin Liang
- Yuhui Zhang
- Hancheng Cao
- Binglu Wang
- Daisy Ding
- Xinyu Yang
- Kailas Vodrahalli
- Siyu He
- Daniel Smith
- Yian Yin
- Daniel McFarland
- James Zou

## Summary
  Expert feedback lays the foundation of rigorous research. However, the rapid
growth of scholarly production and intricate knowledge specialization challenge
the conventional scientific feedback mechanisms. High-quality peer reviews are
increasingly difficult to obtain. Researchers who are more junior or from
under-resourced settings have especially hard times getting timely feedback.
With the breakthrough of large language models (LLM) such as GPT-4, there is
growing interest in using LLMs to generate scientific feedback on research
manuscripts. However, the utility of LLM-generated feedback has not been
systematically studied. To address this gap, we created an automated pipeline
using GPT-4 to provide comments on the full PDFs of scientific papers. We
evaluated the quality of GPT-4's feedback through two large-scale studies. We
first quantitatively compared GPT-4's generated feedback with human peer
reviewer feedback in 15 Nature family journals (3,096 papers in total) and the
ICLR machine learning conference (1,709 papers). The overlap in the points
raised by GPT-4 and by human reviewers (average overlap 30.85% for Nature
journals, 39.23% for ICLR) is comparable to the overlap between two human
reviewers (average overlap 28.58% for Nature journals, 35.25% for ICLR). The
overlap between GPT-4 and human reviewers is larger for the weaker papers. We
then conducted a prospective user study with 308 researchers from 110 US
institutions in the field of AI and computational biology to understand how
researchers perceive feedback generated by our GPT-4 system on their own
papers. Overall, more than half (57.4%) of the users found GPT-4 generated
feedback helpful/very helpful and 82.4% found it more beneficial than feedback
from at least some human reviewers. While our findings show that LLM-generated
feedback can help researchers, we also identify several limitations.


# A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis

[Link to the paper](http://arxiv.org/abs/2307.12856v3)

## Authors
- Izzeddin Gur
- Hiroki Furuta
- Austin Huang
- Mustafa Safdari
- Yutaka Matsuo
- Douglas Eck
- Aleksandra Faust

## Summary
  Pre-trained large language models (LLMs) have recently achieved better
generalization and sample efficiency in autonomous web automation. However, the
performance on real-world websites has still suffered from (1) open domainness,
(2) limited context length, and (3) lack of inductive bias on HTML. We
introduce WebAgent, an LLM-driven agent that learns from self-experience to
complete tasks on real websites following natural language instructions.
WebAgent plans ahead by decomposing instructions into canonical
sub-instructions, summarizes long HTML documents into task-relevant snippets,
and acts on websites via Python programs generated from those. We design
WebAgent with Flan-U-PaLM, for grounded code generation, and HTML-T5, new
pre-trained LLMs for long HTML documents using local and global attention
mechanisms and a mixture of long-span denoising objectives, for planning and
summarization. We empirically demonstrate that our modular recipe improves the
success on real websites by over 50%, and that HTML-T5 is the best model to
solve various HTML understanding tasks; achieving 18.7% higher success rate
than the prior method on MiniWoB web automation benchmark, and SoTA performance
on Mind2Web, an offline task planning evaluation.


# How well does LLM generate security tests?

[Link to the paper](http://arxiv.org/abs/2310.00710v2)

## Authors
- Ying Zhang
- Wenjia Song
- Zhengjie Ji
-  Danfeng
-  Yao
- Na Meng

## Summary
  Developers often build software on top of third-party libraries (Libs) to
improve programmer productivity and software quality. The libraries may contain
vulnerabilities exploitable by hackers to attack the applications (Apps) built
on top of them. People refer to such attacks as supply chain attacks, the
documented number of which has increased 742% in 2022. People created tools to
mitigate such attacks, by scanning the library dependencies of Apps,
identifying the usage of vulnerable library versions, and suggesting secure
alternatives to vulnerable dependencies. However, recent studies show that many
developers do not trust the reports by these tools; they ask for code or
evidence to demonstrate how library vulnerabilities lead to security exploits,
in order to assess vulnerability severity and modification necessity.
Unfortunately, manually crafting demos of application-specific attacks is
challenging and time-consuming, and there is insufficient tool support to
automate that procedure.
  In this study, we used ChatGPT-4.0 to generate security tests, and to
demonstrate how vulnerable library dependencies facilitate the supply chain
attacks to given Apps. We explored various prompt styles/templates, and found
that ChatGPT-4.0 generated tests for all 55 Apps, demonstrating 24 attacks
successfully. It outperformed two state-of-the-art security test generators --
TRANSFER and SIEGE -- by generating a lot more tests and achieving more
exploits. ChatGPT-4.0 worked better when prompts described more on the
vulnerabilities, possible exploits, and code context. Our research will shed
light on new research in security test generation. The generated tests will
help developers create secure by design and secure by default software.


# MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning

[Link to the paper](http://arxiv.org/abs/2309.05653v3)

## Authors
- Xiang Yue
- Xingwei Qu
- Ge Zhang
- Yao Fu
- Wenhao Huang
- Huan Sun
- Yu Su
- Wenhu Chen

## Summary
  We introduce MAmmoTH, a series of open-source large language models (LLMs)
specifically tailored for general math problem-solving. The MAmmoTH models are
trained on MathInstruct, our meticulously curated instruction tuning dataset.
MathInstruct is compiled from 13 math datasets with intermediate rationales,
six of which have rationales newly curated by us. It presents a unique hybrid
of chain-of-thought (CoT) and program-of-thought (PoT) rationales, and also
ensures extensive coverage of diverse fields in math. The hybrid of CoT and PoT
not only unleashes the potential of tool use but also allows different thought
processes for different math problems. As a result, the MAmmoTH series
substantially outperform existing open-source models on nine mathematical
reasoning datasets across all scales with an average accuracy gain between 16%
and 32%. Remarkably, our MAmmoTH-7B model reaches 33% on MATH (a
competition-level dataset), which exceeds the best open-source 7B model
(WizardMath) by 23%, and the MAmmoTH-34B model achieves 44% accuracy on MATH,
even surpassing GPT-4's CoT result. Our work underscores the importance of
diverse problem coverage and the use of hybrid rationales in developing
superior math generalist models.


# Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts

[Link to the paper](http://arxiv.org/abs/2305.13300v3)

## Authors
- Jian Xie
- Kai Zhang
- Jiangjie Chen
- Renze Lou
- Yu Su

## Summary
  By providing external information to large language models (LLMs), tool
augmentation (including retrieval augmentation) has emerged as a promising
solution for addressing the limitations of LLMs' static parametric memory.
However, how receptive are LLMs to such external evidence, especially when the
evidence conflicts with their parametric memory? We present the first
comprehensive and controlled investigation into the behavior of LLMs when
encountering knowledge conflicts. We propose a systematic framework to elicit
high-quality parametric memory from LLMs and construct the corresponding
counter-memory, which enables us to conduct a series of controlled experiments.
Our investigation reveals seemingly contradicting behaviors of LLMs. On the one
hand, different from prior wisdom, we find that LLMs can be highly receptive to
external evidence even when that conflicts with their parametric memory, given
that the external evidence is coherent and convincing. On the other hand, LLMs
also demonstrate a strong confirmation bias when the external evidence contains
some information that is consistent with their parametric memory, despite being
presented with conflicting evidence at the same time. These results pose
important implications that are worth careful consideration for the further
development and deployment of tool- and retrieval-augmented LLMs.


# LM-Infinite: Simple On-the-Fly Length Generalization for Large Language Models

[Link to the paper](http://arxiv.org/abs/2308.16137v4)

## Authors
- Chi Han
- Qifan Wang
- Wenhan Xiong
- Yu Chen
- Heng Ji
- Sinong Wang

## Summary
  In recent years, there have been remarkable advancements in the performance
of Transformer-based Large Language Models (LLMs) across various domains. As
these LLMs are deployed for increasingly complex domains, they often face the
need to follow longer user prompts or generate longer texts. In these
situations, the $\textit{length generalization failure}$ of LLMs on long
sequences becomes more prominent. Most pre-training schemes truncate training
sequences to a fixed length. LLMs often struggle to generate fluent and
coherent texts after longer contexts, even with relative positional encoding
specifically designed to cope with this problem. Common solutions such as
finetuning on longer corpora often involve daunting hardware and time costs and
require careful training process design. To more efficiently extrapolate
existing LLMs' generation quality to longer texts, we theoretically and
empirically investigate the main out-of-distribution (OOD) factors contributing
to this problem. Inspired by this diagnosis, we propose a simple yet effective
solution for on-the-fly length generalization, LM-Infinite. It involves only a
$\mathbf{\Lambda}$-shaped attention mask (to avoid excessive attended tokens)
and a distance limit (to avoid unseen distances) while requiring no parameter
updates or learning. We find it applicable to a variety of LLMs using
relative-position encoding methods. LM-Infinite is computationally efficient
with $O(n)$ time and space, and demonstrates consistent text generation fluency
and quality to as long as 128k tokens on ArXiv and OpenWebText2 datasets, with
2.72x decoding speedup. We will make the codes publicly available following
publication.


# Time-LLM: Time Series Forecasting by Reprogramming Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.01728v1)

## Authors
- Ming Jin
- Shiyu Wang
- Lintao Ma
- Zhixuan Chu
- James Y. Zhang
- Xiaoming Shi
- Pin-Yu Chen
- Yuxuan Liang
- Yuan-Fang Li
- Shirui Pan
- Qingsong Wen

## Summary
  Time series forecasting holds significant importance in many real-world
dynamic systems and has been extensively studied. Unlike natural language
process (NLP) and computer vision (CV), where a single large model can tackle
multiple tasks, models for time series forecasting are often specialized,
necessitating distinct designs for different tasks and applications. While
pre-trained foundation models have made impressive strides in NLP and CV, their
development in time series domains has been constrained by data sparsity.
Recent studies have revealed that large language models (LLMs) possess robust
pattern recognition and reasoning abilities over complex sequences of tokens.
However, the challenge remains in effectively aligning the modalities of time
series data and natural language to leverage these capabilities. In this work,
we present Time-LLM, a reprogramming framework to repurpose LLMs for general
time series forecasting with the backbone language models kept intact. We begin
by reprogramming the input time series with text prototypes before feeding it
into the frozen LLM to align the two modalities. To augment the LLM's ability
to reason with time series data, we propose Prompt-as-Prefix (PaP), which
enriches the input context and directs the transformation of reprogrammed input
patches. The transformed time series patches from the LLM are finally projected
to obtain the forecasts. Our comprehensive evaluations demonstrate that
Time-LLM is a powerful time series learner that outperforms state-of-the-art,
specialized forecasting models. Moreover, Time-LLM excels in both few-shot and
zero-shot learning scenarios.


# Can GPT-4 Replicate Empirical Software Engineering Research?

[Link to the paper](http://arxiv.org/abs/2310.01727v1)

## Authors
- Jenny T. Liang
- Carmen Badea
- Christian Bird
- Robert DeLine
- Denae Ford
- Nicole Forsgren
- Thomas Zimmermann

## Summary
  Empirical software engineering research on production systems has brought
forth a better understanding of the software engineering process for
practitioners and researchers alike. However, only a small subset of production
systems is studied, limiting the impact of this research. While software
engineering practitioners benefit from replicating research on their own data,
this poses its own set of challenges, since performing replications requires a
deep understanding of research methodologies and subtle nuances in software
engineering data. Given that large language models (LLMs), such as GPT-4, show
promise in tackling both software engineering- and science-related tasks, these
models could help democratize empirical software engineering research.
  In this paper, we examine LLMs' abilities to perform replications of
empirical software engineering research on new data. We specifically study
their ability to surface assumptions made in empirical software engineering
research methodologies, as well as their ability to plan and generate code for
analysis pipelines on seven empirical software engineering papers. We perform a
user study with 14 participants with software engineering research expertise,
who evaluate GPT-4-generated assumptions and analysis plans (i.e., a list of
module specifications) from the papers. We find that GPT-4 is able to surface
correct assumptions, but struggle to generate ones that reflect common
knowledge about software engineering data. In a manual analysis of the
generated code, we find that the GPT-4-generated code contains the correct
high-level logic, given a subset of the methodology. However, the code contains
many small implementation-level errors, reflecting a lack of software
engineering knowledge. Our findings have implications for leveraging LLMs for
software engineering research as well as practitioner data scientists in
software teams.


# Large Language Models for Test-Free Fault Localization

[Link to the paper](http://arxiv.org/abs/2310.01726v1)

## Authors
- Aidan Z. H. Yang
- Ruben Martins
- Claire Le Goues
- Vincent J. Hellendoorn

## Summary
  Fault Localization (FL) aims to automatically localize buggy lines of code, a
key first step in many manual and automatic debugging tasks. Previous FL
techniques assume the provision of input tests, and often require extensive
program analysis, program instrumentation, or data preprocessing. Prior work on
deep learning for APR struggles to learn from small datasets and produces
limited results on real-world programs. Inspired by the ability of large
language models (LLMs) of code to adapt to new tasks based on very few
examples, we investigate the applicability of LLMs to line level fault
localization. Specifically, we propose to overcome the left-to-right nature of
LLMs by fine-tuning a small set of bidirectional adapter layers on top of the
representations learned by LLMs to produce LLMAO, the first language model
based fault localization approach that locates buggy lines of code without any
test coverage information. We fine-tune LLMs with 350 million, 6 billion, and
16 billion parameters on small, manually curated corpora of buggy programs such
as the Defects4J corpus. We observe that our technique achieves substantially
more confidence in fault localization when built on the larger models, with bug
localization performance scaling consistently with the LLM size. Our empirical
evaluation shows that LLMAO improves the Top-1 results over the
state-of-the-art machine learning fault localization (MLFL) baselines by
2.3%-54.4%, and Top-5 results by 14.4%-35.6%. LLMAO is also the first FL
technique trained using a language model architecture that can detect security
vulnerabilities down to the code line level.


# RecallM: An Adaptable Memory Mechanism with Temporal Understanding for Large Language Models

[Link to the paper](http://arxiv.org/abs/2307.02738v3)

## Authors
- Brandon Kynoch
- Hugo Latapie
- Dwane van der Sluis

## Summary
  Large Language Models (LLMs) have made extraordinary progress in the field of
Artificial Intelligence and have demonstrated remarkable capabilities across a
large variety of tasks and domains. However, as we venture closer to creating
Artificial General Intelligence (AGI) systems, we recognize the need to
supplement LLMs with long-term memory to overcome the context window limitation
and more importantly, to create a foundation for sustained reasoning,
cumulative learning and long-term user interaction. In this paper we propose
RecallM, a novel architecture for providing LLMs with an adaptable and
updatable long-term memory mechanism. Unlike previous methods, the RecallM
architecture is particularly effective at belief updating and maintaining a
temporal understanding of the knowledge provided to it. We demonstrate through
various experiments the effectiveness of this architecture. Furthermore,
through our own temporal understanding and belief updating experiments, we show
that RecallM is four times more effective than using a vector database for
updating knowledge previously stored in long-term memory. We also demonstrate
that RecallM shows competitive performance on general question-answering and
in-context learning tasks.


# Deciphering Diagnoses: How Large Language Models Explanations Influence Clinical Decision Making

[Link to the paper](http://arxiv.org/abs/2310.01708v1)

## Authors
- D. Umerenkov
- G. Zubkova
- A. Nesterov

## Summary
  Clinical Decision Support Systems (CDSS) utilize evidence-based knowledge and
patient data to offer real-time recommendations, with Large Language Models
(LLMs) emerging as a promising tool to generate plain-text explanations for
medical decisions. This study explores the effectiveness and reliability of
LLMs in generating explanations for diagnoses based on patient complaints.
Three experienced doctors evaluated LLM-generated explanations of the
connection between patient complaints and doctor and model-assigned diagnoses
across several stages. Experimental results demonstrated that LLM explanations
significantly increased doctors' agreement rates with given diagnoses and
highlighted potential errors in LLM outputs, ranging from 5% to 30%. The study
underscores the potential and challenges of LLMs in healthcare and emphasizes
the need for careful integration and evaluation to ensure patient safety and
optimal clinical utility.


# In-context Autoencoder for Context Compression in a Large Language Model

[Link to the paper](http://arxiv.org/abs/2307.06945v2)

## Authors
- Tao Ge
- Jing Hu
- Lei Wang
- Xun Wang
- Si-Qing Chen
- Furu Wei

## Summary
  We propose the In-context Autoencoder (ICAE), leveraging the power of a large
language models (LLM) to compress a long context into short compact memory
slots that can be directly conditioned on by the LLM for various purposes. ICAE
is first pretrained using both autoencoding and language modeling objectives on
massive text data, enabling it to generate memory slots that accurately and
comprehensively represent the original context; Then, it is fine-tuned on
instruction data for producing desirable responses to various prompts.
Experiments demonstrate that our lightweight ICAE, introducing fewer than 1%
additional parameters, effectively achieves 4X context compression based on
Llama, offering advantages in both improved latency and GPU memory cost during
inference, and showing an interesting insight in memorization as well as
potential for scalability. These promising results imply a novel perspective on
the connection between working memory in cognitive science and representation
learning in LLMs, revealing ICAE's significant implications in addressing the
long context problem and suggesting further research in LLM context management.
Our data, code and model are released at https://github.com/getao/icae.


# On the Possibilities of AI-Generated Text Detection

[Link to the paper](http://arxiv.org/abs/2304.04736v3)

## Authors
- Souradip Chakraborty
- Amrit Singh Bedi
- Sicheng Zhu
- Bang An
- Dinesh Manocha
- Furong Huang

## Summary
  Our work addresses the critical issue of distinguishing text generated by
Large Language Models (LLMs) from human-produced text, a task essential for
numerous applications. Despite ongoing debate about the feasibility of such
differentiation, we present evidence supporting its consistent achievability,
except when human and machine text distributions are indistinguishable across
their entire support. Drawing from information theory, we argue that as
machine-generated text approximates human-like quality, the sample size needed
for detection increases. We establish precise sample complexity bounds for
detecting AI-generated text, laying groundwork for future research aimed at
developing advanced, multi-sample detectors. Our empirical evaluations across
multiple datasets (Xsum, Squad, IMDb, and Kaggle FakeNews) confirm the
viability of enhanced detection methods. We test various state-of-the-art text
generators, including GPT-2, GPT-3.5-Turbo, Llama, Llama-2-13B-Chat-HF, and
Llama-2-70B-Chat-HF, against detectors, including oBERTa-Large/Base-Detector,
GPTZero. Our findings align with OpenAI's empirical data related to sequence
length, marking the first theoretical substantiation for these observations.


# VAL: Interactive Task Learning with GPT Dialog Parsing

[Link to the paper](http://arxiv.org/abs/2310.01627v1)

## Authors
- Lane Lawley
- Christopher J. MacLellan

## Summary
  Reinforcement learning often requires millions of examples to produce static,
black-box models. In contrast, interactive task learning (ITL) emphasizes
incremental knowledge acquisition from limited instruction provided by humans
in modalities such as natural language. However, in practice, ITL systems often
suffers from brittle, error-prone language parsing. Large language models
(LLMs) are resistant to brittleness but are not interpretable and cannot learn
incrementally. We present VAL, an ITL system with a new philosophy for
LLM/symbolic integration. By using LLMs only for specific tasks -- such as
predicate and argument selection -- within an algorithmic framework, VAL reaps
the benefits of LLMs to support interactive learning of hierarchical task
knowledge from natural language. Acquired knowledge is human interpretable and
generalizes to support execution of novel tasks without additional training. We
studied users' interactions with VAL in a video game setting, finding that most
users could successfully teach VAL using language they felt was natural.


# Towards Efficient and Effective Adaptation of Large Language Models for Sequential Recommendation

[Link to the paper](http://arxiv.org/abs/2310.01612v1)

## Authors
- Bo Peng
- Ben Burns
- Ziqi Chen
- Srinivasan Parthasarathy
- Xia Ning

## Summary
  In recent years, with large language models (LLMs) achieving state-of-the-art
performance in context understanding, increasing efforts have been dedicated to
developing LLM-enhanced sequential recommendation (SR) methods. Considering
that most existing LLMs are not specifically optimized for recommendation
tasks, adapting them for SR becomes a critical step in LLM-enhanced SR methods.
Though numerous adaptation methods have been developed, it still remains a
significant challenge to adapt LLMs for SR both efficiently and effectively. To
address this challenge, in this paper, we introduce a novel side sequential
network adaptation method, denoted as SSNA, for LLM enhanced SR. SSNA features
three key designs to allow both efficient and effective LLM adaptation. First,
SSNA learns adapters separate from LLMs, while fixing all the pre-trained
parameters within LLMs to allow efficient adaptation. In addition, SSNA adapts
the top-a layers of LLMs jointly, and integrates adapters sequentially for
enhanced effectiveness (i.e., recommendation performance). We compare SSNA
against five state-of-the-art baseline methods on five benchmark datasets using
three LLMs. The experimental results demonstrate that SSNA significantly
outperforms all the baseline methods in terms of recommendation performance,
and achieves substantial improvement over the best-performing baseline methods
at both run-time and memory efficiency during training. Our analysis shows the
effectiveness of integrating adapters in a sequential manner. Our parameter
study demonstrates the effectiveness of jointly adapting the top-a layers of
LLMs.


# Impact of Large Language Models on Generating Software Specifications

[Link to the paper](http://arxiv.org/abs/2306.03324v2)

## Authors
- Danning Xie
- Byungwoo Yoo
- Nan Jiang
- Mijung Kim
- Lin Tan
- Xiangyu Zhang
- Judy S. Lee

## Summary
  Software specifications are essential for ensuring the reliability of
software systems. Existing specification extraction approaches, however, suffer
from limited generalizability and require manual efforts. The recent emergence
of Large Language Models (LLMs), which have been successfully applied to
numerous software engineering tasks, offers a promising avenue for automating
this process. In this paper, we conduct the first empirical study to evaluate
the capabilities of LLMs for generating software specifications from software
comments or documentation. We evaluate LLMs' performance with Few Shot Learning
(FSL), enabling LLMs to generalize from a small number of examples, as well as
different prompt construction strategies, and compare the performance of LLMs
with traditional approaches. Additionally, we conduct a comparative diagnosis
of the failure cases from both LLMs and traditional methods, identifying their
unique strengths and weaknesses. Lastly, we conduct extensive experiments on 15
state of the art LLMs, evaluating their performance and cost effectiveness for
generating software specifications.
  Our results show that with FSL, LLMs outperform traditional methods (by
5.6%), and more sophisticated prompt construction strategies can further
enlarge this performance gap (up to 5.1 to 10.0%). Yet, LLMs suffer from their
unique challenges, such as ineffective prompts and the lack of domain
knowledge, which together account for 53 to 60% of LLM unique failures. The
strong performance of open source models (e.g., StarCoder) makes closed source
models (e.g., GPT 3 Davinci) less desirable due to size and cost. Our study
offers valuable insights for future research to improve specification
generation.


# On the Safety of Open-Sourced Large Language Models: Does Alignment Really Prevent Them From Being Misused?

[Link to the paper](http://arxiv.org/abs/2310.01581v1)

## Authors
- Hangfan Zhang
- Zhimeng Guo
- Huaisheng Zhu
- Bochuan Cao
- Lu Lin
- Jinyuan Jia
- Jinghui Chen
- Dinghao Wu

## Summary
  Large Language Models (LLMs) have achieved unprecedented performance in
Natural Language Generation (NLG) tasks. However, many existing studies have
shown that they could be misused to generate undesired content. In response,
before releasing LLMs for public access, model developers usually align those
language models through Supervised Fine-Tuning (SFT) or Reinforcement Learning
with Human Feedback (RLHF). Consequently, those aligned large language models
refuse to generate undesired content when facing potentially harmful/unethical
requests. A natural question is "could alignment really prevent those
open-sourced large language models from being misused to generate undesired
content?''. In this work, we provide a negative answer to this question. In
particular, we show those open-sourced, aligned large language models could be
easily misguided to generate undesired content without heavy computations or
careful prompt designs. Our key idea is to directly manipulate the generation
process of open-sourced LLMs to misguide it to generate undesired content
including harmful or biased information and even private data. We evaluate our
method on 4 open-sourced LLMs accessible publicly and our finding highlights
the need for more advanced mitigation strategies for open-sourced LLMs.


# GPT-Driver: Learning to Drive with GPT

[Link to the paper](http://arxiv.org/abs/2310.01415v1)

## Authors
- Jiageng Mao
- Yuxi Qian
- Hang Zhao
- Yue Wang

## Summary
  We present a simple yet effective approach that can transform the OpenAI
GPT-3.5 model into a reliable motion planner for autonomous vehicles. Motion
planning is a core challenge in autonomous driving, aiming to plan a driving
trajectory that is safe and comfortable. Existing motion planners predominantly
leverage heuristic methods to forecast driving trajectories, yet these
approaches demonstrate insufficient generalization capabilities in the face of
novel and unseen driving scenarios. In this paper, we propose a novel approach
to motion planning that capitalizes on the strong reasoning capabilities and
generalization potential inherent to Large Language Models (LLMs). The
fundamental insight of our approach is the reformulation of motion planning as
a language modeling problem, a perspective not previously explored.
Specifically, we represent the planner inputs and outputs as language tokens,
and leverage the LLM to generate driving trajectories through a language
description of coordinate positions. Furthermore, we propose a novel
prompting-reasoning-finetuning strategy to stimulate the numerical reasoning
potential of the LLM. With this strategy, the LLM can describe highly precise
trajectory coordinates and also its internal decision-making process in natural
language. We evaluate our approach on the large-scale nuScenes dataset, and
extensive experiments substantiate the effectiveness, generalization ability,
and interpretability of our GPT-based motion planner. Code will be released
upon acceptance.


# DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model

[Link to the paper](http://arxiv.org/abs/2310.01412v1)

## Authors
- Zhenhua Xu
- Yujia Zhang
- Enze Xie
- Zhen Zhao
- Yong Guo
- Kenneth K. Y. Wong
- Zhenguo Li
- Hengshuang Zhao

## Summary
  In the past decade, autonomous driving has experienced rapid development in
both academia and industry. However, its limited interpretability remains a
significant unsolved problem, severely hindering autonomous vehicle
commercialization and further development. Previous approaches utilizing small
language models have failed to address this issue due to their lack of
flexibility, generalization ability, and robustness. Recently, multimodal large
language models (LLMs) have gained considerable attention from the research
community for their capability to process and reason non-text data (e.g.,
images and videos) by text. In this paper, we present DriveGPT4, an
interpretable end-to-end autonomous driving system utilizing LLMs. DriveGPT4 is
capable of interpreting vehicle actions and providing corresponding reasoning,
as well as answering diverse questions posed by human users for enhanced
interaction. Additionally, DriveGPT4 predicts vehicle low-level control signals
in an end-to-end fashion. These capabilities stem from a customized visual
instruction tuning dataset specifically designed for autonomous driving. To the
best of our knowledge, DriveGPT4 is the first work focusing on interpretable
end-to-end autonomous driving. When evaluated on multiple tasks alongside
conventional methods and video understanding LLMs, DriveGPT4 demonstrates
superior qualitative and quantitative performance. Additionally, DriveGPT4 can
be generalized in a zero-shot fashion to accommodate more unseen scenarios. The
project page is available at https://tonyxuqaq.github.io/projects/DriveGPT4/ .


# Evaluating the Decency and Consistency of Data Validation Tests Generated by LLMs

[Link to the paper](http://arxiv.org/abs/2310.01402v1)

## Authors
- Rohan Alexander
- Lindsay Katz
- Callandra Moore
- Zane Schwartz

## Summary
  We investigated the potential of large language models (LLMs) in developing
dataset validation tests. We carried out 96 experiments each for both GPT-3.5
and GPT-4, examining different prompt scenarios, learning modes, temperature
settings, and roles. The prompt scenarios were: 1) Asking for expectations, 2)
Asking for expectations with a given context, 3) Asking for expectations after
requesting a simulation, and 4) Asking for expectations with a provided data
sample. For learning modes, we tested: 1) zero-shot, 2) one-shot, and 3)
few-shot learning. We also tested four temperature settings: 0, 0.4, 0.6, and
1. Furthermore, two distinct roles were considered: 1) "helpful assistant", 2)
"expert data scientist". To gauge consistency, every setup was tested five
times. The LLM-generated responses were benchmarked against a gold standard
suite, created by an experienced data scientist knowledgeable about the data in
question. We find there are considerable returns to the use of few-shot
learning, and that the more explicit the data setting can be the better. The
best LLM configurations complement, rather than substitute, the gold standard
results. This study underscores the value LLMs can bring to the data cleaning
and preparation stages of the data science workflow.


# Who is ChatGPT? Benchmarking LLMs' Psychological Portrayal Using PsychoBench

[Link to the paper](http://arxiv.org/abs/2310.01386v1)

## Authors
- Jen-tse Huang
- Wenxuan Wang
- Eric John Li
- Man Ho Lam
- Shujie Ren
- Youliang Yuan
- Wenxiang Jiao
- Zhaopeng Tu
- Michael R. Lyu

## Summary
  Large Language Models (LLMs) have recently showcased their remarkable
capacities, not only in natural language processing tasks but also across
diverse domains such as clinical medicine, legal consultation, and education.
LLMs become more than mere applications, evolving into assistants capable of
addressing diverse user requests. This narrows the distinction between human
beings and artificial intelligence agents, raising intriguing questions
regarding the potential manifestation of personalities, temperaments, and
emotions within LLMs. In this paper, we propose a framework, PsychoBench, for
evaluating diverse psychological aspects of LLMs. Comprising thirteen scales
commonly used in clinical psychology, PsychoBench further classifies these
scales into four distinct categories: personality traits, interpersonal
relationships, motivational tests, and emotional abilities. Our study examines
five popular models, namely \texttt{text-davinci-003}, ChatGPT, GPT-4,
LLaMA-2-7b, and LLaMA-2-13b. Additionally, we employ a jailbreak approach to
bypass the safety alignment protocols and test the intrinsic natures of LLMs.
We have made PsychoBench openly accessible via
\url{https://github.com/CUHK-ARISE/PsychoBench}.


# Compressing LLMs: The Truth is Rarely Pure and Never Simple

[Link to the paper](http://arxiv.org/abs/2310.01382v1)

## Authors
- Ajay Jaiswal
- Zhe Gan
- Xianzhi Du
- Bowen Zhang
- Zhangyang Wang
- Yinfei Yang

## Summary
  Despite their remarkable achievements, modern Large Language Models (LLMs)
encounter exorbitant computational and memory footprints. Recently, several
works have shown significant success in training-free and data-free compression
(pruning and quantization) of LLMs achieving 50-60% sparsity and reducing the
bit-width down to 3 or 4 bits per weight, with negligible perplexity
degradation over the uncompressed baseline. As recent research efforts are
focused on developing increasingly sophisticated compression methods, our work
takes a step back, and re-evaluates the effectiveness of existing SoTA
compression methods, which rely on a fairly simple and widely questioned
metric, perplexity (even for dense LLMs). We introduce Knowledge-Intensive
Compressed LLM BenchmarK (LLM-KICK), a collection of carefully-curated tasks to
re-define the evaluation protocol for compressed LLMs, which have significant
alignment with their dense counterparts, and perplexity fail to capture subtle
change in their true capabilities. LLM-KICK unveils many favorable merits and
unfortunate plights of current SoTA compression methods: all pruning methods
suffer significant performance degradation, sometimes at trivial sparsity
ratios (e.g., 25-30%), and fail for N:M sparsity on knowledge-intensive tasks;
current quantization methods are more successful than pruning; yet, pruned LLMs
even at $\geq 50$% sparsity are robust in-context retrieval and summarization
systems; among others. LLM-KICK is designed to holistically access compressed
LLMs' ability for language understanding, reasoning, generation, in-context
retrieval, in-context summarization, etc. We hope our study can foster the
development of better LLM compression methods. All our related codes are planed
to be open-sourced.


# UltraFeedback: Boosting Language Models with High-quality Feedback

[Link to the paper](http://arxiv.org/abs/2310.01377v1)

## Authors
- Ganqu Cui
- Lifan Yuan
- Ning Ding
- Guanming Yao
- Wei Zhu
- Yuan Ni
- Guotong Xie
- Zhiyuan Liu
- Maosong Sun

## Summary
  Reinforcement learning from human feedback (RLHF) has become a pivot
technique in aligning large language models (LLMs) with human preferences. In
RLHF practice, preference data plays a crucial role in bridging human
proclivity and LLMs. However, the scarcity of diverse, naturalistic datasets of
human preferences on LLM outputs at scale poses a great challenge to RLHF as
well as feedback learning research within the open-source community. Current
preference datasets, either proprietary or limited in size and prompt variety,
result in limited RLHF adoption in open-source models and hinder further
exploration. In this study, we propose ULTRAFEEDBACK, a large-scale,
high-quality, and diversified preference dataset designed to overcome these
limitations and foster RLHF development. To create ULTRAFEEDBACK, we compile a
diverse array of instructions and models from multiple sources to produce
comparative data. We meticulously devise annotation instructions and employ
GPT-4 to offer detailed feedback in both numerical and textual forms.
ULTRAFEEDBACK establishes a reproducible and expandable preference data
construction pipeline, serving as a solid foundation for future RLHF and
feedback learning research. Utilizing ULTRAFEEDBACK, we train various models to
demonstrate its effectiveness, including the reward model UltraRM, chat
language model UltraLM-13B-PPO, and critique model UltraCM. Experimental
results indicate that our models outperform existing open-source models,
achieving top performance across multiple benchmarks. Our data and models are
available at https://github.com/thunlp/UltraFeedback.


# GenSim: Generating Robotic Simulation Tasks via Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.01361v1)

## Authors
- Lirui Wang
- Yiyang Ling
- Zhecheng Yuan
- Mohit Shridhar
- Chen Bao
- Yuzhe Qin
- Bailin Wang
- Huazhe Xu
- Xiaolong Wang

## Summary
  Collecting large amounts of real-world interaction data to train general
robotic policies is often prohibitively expensive, thus motivating the use of
simulation data. However, existing methods for data generation have generally
focused on scene-level diversity (e.g., object instances and poses) rather than
task-level diversity, due to the human effort required to come up with and
verify novel tasks. This has made it challenging for policies trained on
simulation data to demonstrate significant task-level generalization. In this
paper, we propose to automatically generate rich simulation environments and
expert demonstrations by exploiting a large language models' (LLM) grounding
and coding ability. Our approach, dubbed GenSim, has two modes: goal-directed
generation, wherein a target task is given to the LLM and the LLM proposes a
task curriculum to solve the target task, and exploratory generation, wherein
the LLM bootstraps from previous tasks and iteratively proposes novel tasks
that would be helpful in solving more complex tasks. We use GPT4 to expand the
existing benchmark by ten times to over 100 tasks, on which we conduct
supervised finetuning and evaluate several LLMs including finetuned GPTs and
Code Llama on code generation for robotic simulation tasks. Furthermore, we
observe that LLMs-generated simulation programs can enhance task-level
generalization significantly when used for multitask policy training. We
further find that with minimal sim-to-real adaptation, the multitask policies
pretrained on GPT4-generated simulation tasks exhibit stronger transfer to
unseen long-horizon tasks in the real world and outperform baselines by 25%.
See the project website (https://liruiw.github.io/gensim) for code, demos, and
videos.


# Self-Refined Large Language Model as Automated Reward Function Designer for Deep Reinforcement Learning in Robotics

[Link to the paper](http://arxiv.org/abs/2309.06687v2)

## Authors
- Jiayang Song
- Zhehua Zhou
- Jiawei Liu
- Chunrong Fang
- Zhan Shu
- Lei Ma

## Summary
  Although Deep Reinforcement Learning (DRL) has achieved notable success in
numerous robotic applications, designing a high-performing reward function
remains a challenging task that often requires substantial manual input.
Recently, Large Language Models (LLMs) have been extensively adopted to address
tasks demanding in-depth common-sense knowledge, such as reasoning and
planning. Recognizing that reward function design is also inherently linked to
such knowledge, LLM offers a promising potential in this context. Motivated by
this, we propose in this work a novel LLM framework with a self-refinement
mechanism for automated reward function design. The framework commences with
the LLM formulating an initial reward function based on natural language
inputs. Then, the performance of the reward function is assessed, and the
results are presented back to the LLM for guiding its self-refinement process.
We examine the performance of our proposed framework through a variety of
continuous robotic control tasks across three diverse robotic systems. The
results indicate that our LLM-designed reward functions are able to rival or
even surpass manually designed reward functions, highlighting the efficacy and
applicability of our approach.


# RA-DIT: Retrieval-Augmented Dual Instruction Tuning

[Link to the paper](http://arxiv.org/abs/2310.01352v1)

## Authors
- Xi Victoria Lin
- Xilun Chen
- Mingda Chen
- Weijia Shi
- Maria Lomeli
- Rich James
- Pedro Rodriguez
- Jacob Kahn
- Gergely Szilvasy
- Mike Lewis
- Luke Zettlemoyer
- Scott Yih

## Summary
  Retrieval-augmented language models (RALMs) improve performance by accessing
long-tail and up-to-date knowledge from external data stores, but are
challenging to build. Existing approaches require either expensive
retrieval-specific modifications to LM pre-training or use post-hoc integration
of the data store that leads to suboptimal performance. We introduce
Retrieval-Augmented Dual Instruction Tuning (RA-DIT), a lightweight fine-tuning
methodology that provides a third option by retrofitting any LLM with retrieval
capabilities. Our approach operates in two distinct fine-tuning steps: (1) one
updates a pre-trained LM to better use retrieved information, while (2) the
other updates the retriever to return more relevant results, as preferred by
the LM. By fine-tuning over tasks that require both knowledge utilization and
contextual awareness, we demonstrate that each stage yields significant
performance improvements, and using both leads to additional gains. Our best
model, RA-DIT 65B, achieves state-of-the-art performance across a range of
knowledge-intensive zero- and few-shot learning benchmarks, significantly
outperforming existing in-context RALM approaches by up to +8.9% in 0-shot
setting and +1.4% in 5-shot setting on average.


# ChemCrow: Augmenting large-language models with chemistry tools

[Link to the paper](http://arxiv.org/abs/2304.05376v5)

## Authors
- Andres M Bran
- Sam Cox
- Oliver Schilter
- Carlo Baldassari
- Andrew D White
- Philippe Schwaller

## Summary
  Over the last decades, excellent computational chemistry tools have been
developed. Integrating them into a single platform with enhanced accessibility
could help reaching their full potential by overcoming steep learning curves.
Recently, large-language models (LLMs) have shown strong performance in tasks
across domains, but struggle with chemistry-related problems. Moreover, these
models lack access to external knowledge sources, limiting their usefulness in
scientific applications. In this study, we introduce ChemCrow, an LLM chemistry
agent designed to accomplish tasks across organic synthesis, drug discovery,
and materials design. By integrating 18 expert-designed tools, ChemCrow
augments the LLM performance in chemistry, and new capabilities emerge. Our
agent autonomously planned and executed the syntheses of an insect repellent,
three organocatalysts, and guided the discovery of a novel chromophore. Our
evaluation, including both LLM and expert assessments, demonstrates ChemCrow's
effectiveness in automating a diverse set of chemical tasks. Surprisingly, we
find that GPT-4 as an evaluator cannot distinguish between clearly wrong GPT-4
completions and Chemcrow's performance. Our work not only aids expert chemists
and lowers barriers for non-experts, but also fosters scientific advancement by
bridging the gap between experimental and computational chemistry.


# L2MAC: Large Language Model Automatic Computer for Unbounded Code Generation

[Link to the paper](http://arxiv.org/abs/2310.02003v1)

## Authors
- Samuel Holt
- Max Ruiz Luyten
- Mihaela van der Schaar

## Summary
  Transformer-based large language models (LLMs) are constrained by the fixed
context window of the underlying transformer architecture, hindering their
ability to produce long and logically consistent code. Memory-augmented LLMs
are a promising solution, but current approaches cannot handle long code
generation tasks since they (1) only focus on reading memory and reduce its
evolution to the concatenation of new memories or (2) use very specialized
memories that cannot adapt to other domains. This paper presents L2MAC, the
first practical LLM-based stored-program automatic computer for long and
consistent code generation. Its memory has two components: the instruction
registry, which is populated with a prompt program to solve the user-given
task, and a file store, which will contain the final and intermediate outputs.
Each instruction is executed by a separate LLM instance, whose context is
managed by a control unit capable of precise memory reading and writing to
ensure effective interaction with the file store. These components enable L2MAC
to generate virtually unbounded code structures, bypassing the constraints of
the finite context window while producing code that fulfills complex
user-specified requirements. We empirically show that L2MAC succeeds in
generating large code bases for system design tasks where other coding methods
fall short in implementing user requirements and provide insight into the
reasons for this performance gap.


# An Unsupervised Method for Estimating Class Separability of Datasets with Application to LLMs Fine-Tuning

[Link to the paper](http://arxiv.org/abs/2305.15016v2)

## Authors
- Najah Ghalyan
- Kostis Gourgoulias
- Yash Satsangi
- Sean Moran
- Maxime Labonne
- Joseph Sabelja

## Summary
  This paper proposes an unsupervised method that leverages topological
characteristics of data manifolds to estimate class separability of the data
without requiring labels. Experiments conducted in this paper on several
datasets demonstrate a clear correlation and consistency between the class
separability estimated by the proposed method with supervised metrics like
Fisher Discriminant Ratio~(FDR) and cross-validation of a classifier, which
both require labels. This can enable implementing learning paradigms aimed at
learning from both labeled and unlabeled data, like semi-supervised and
transductive learning. This would be particularly useful when we have limited
labeled data and a relatively large unlabeled dataset that can be used to
enhance the learning process. The proposed method is implemented for language
model fine-tuning with automated stopping criterion by monitoring class
separability of the embedding-space manifold in an unsupervised setting. The
proposed methodology has been first validated on synthetic data, where the
results show a clear consistency between class separability estimated by the
proposed method and class separability computed by FDR. The method has been
also implemented on both public and internal data. The results show that the
proposed method can effectively aid -- without the need for labels -- a
decision on when to stop or continue the fine-tuning of a language model and
which fine-tuning iteration is expected to achieve a maximum classification
performance through quantification of the class separability of the embedding
manifold.


# ChoiceMates: Supporting Unfamiliar Online Decision-Making with Multi-Agent Conversational Interactions

[Link to the paper](http://arxiv.org/abs/2310.01331v1)

## Authors
- Jeongeon Park
- Bryan Min
- Xiaojuan Ma
- Juho Kim

## Summary
  Unfamiliar decisions -- decisions where people lack adequate domain knowledge
or expertise -- specifically increase the complexity and uncertainty of the
process of searching for, understanding, and making decisions with online
information. Through our formative study (n=14), we observed users' challenges
in accessing diverse perspectives, identifying relevant information, and
deciding the right moment to make the final decision. We present ChoiceMates, a
system that enables conversations with a dynamic set of LLM-powered agents for
a holistic domain understanding and efficient discovery and management of
information to make decisions. Agents, as opinionated personas, flexibly join
the conversation, not only providing responses but also conversing among
themselves to elicit each agent's preferences. Our between-subjects study
(n=36) comparing ChoiceMates to conventional web search and single-agent showed
that ChoiceMates was more helpful in discovering, diving deeper, and managing
information compared to Web with higher confidence. We also describe how
participants utilized multi-agent conversations in their decision-making
process.


# Towards reporting bias in visual-language datasets: bimodal augmentation by decoupling object-attribute association

[Link to the paper](http://arxiv.org/abs/2310.01330v1)

## Authors
- Qiyu Wu
- Mengjie Zhao
- Yutong He
- Lang Huang
- Junya Ono
- Hiromi Wakaki
- Yuki Mitsufuji

## Summary
  Reporting bias arises when people assume that some knowledge is universally
understood and hence, do not necessitate explicit elaboration. In this paper,
we focus on the wide existence of reporting bias in visual-language datasets,
embodied as the object-attribute association, which can subsequentially degrade
models trained on them. To mitigate this bias, we propose a bimodal
augmentation (BiAug) approach through object-attribute decoupling to flexibly
synthesize visual-language examples with a rich array of object-attribute
pairing and construct cross-modal hard negatives. We employ large language
models (LLMs) in conjunction with a grounding object detector to extract target
objects. Subsequently, the LLM generates a detailed attribute description for
each object and produces a corresponding hard negative counterpart. An
inpainting model is then used to create images based on these detailed object
descriptions. By doing so, the synthesized examples explicitly complement
omitted objects and attributes to learn, and the hard negative pairs steer the
model to distinguish object attributes. Our experiments demonstrated that BiAug
is superior in object-attribute understanding. In addition, BiAug also improves
the performance on zero-shot retrieval tasks on general benchmarks like MSCOCO
and Flickr30K. BiAug refines the way of collecting text-image datasets.
Mitigating the reporting bias helps models achieve a deeper understanding of
visual-language phenomena, expanding beyond mere frequent patterns to encompass
the richness and diversity of real-world scenarios.


# MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models

[Link to the paper](http://arxiv.org/abs/2304.10592v2)

## Authors
- Deyao Zhu
- Jun Chen
- Xiaoqian Shen
- Xiang Li
- Mohamed Elhoseiny

## Summary
  The recent GPT-4 has demonstrated extraordinary multi-modal abilities, such
as directly generating websites from handwritten text and identifying humorous
elements within images. These features are rarely observed in previous
vision-language models. However, the technical details behind GPT-4 continue to
remain undisclosed. We believe that the enhanced multi-modal generation
capabilities of GPT-4 stem from the utilization of sophisticated large language
models (LLM). To examine this phenomenon, we present MiniGPT-4, which aligns a
frozen visual encoder with a frozen advanced LLM, Vicuna, using one projection
layer. Our work, for the first time, uncovers that properly aligning the visual
features with an advanced large language model can possess numerous advanced
multi-modal abilities demonstrated by GPT-4, such as detailed image description
generation and website creation from hand-drawn drafts. Furthermore, we also
observe other emerging capabilities in MiniGPT-4, including writing stories and
poems inspired by given images, teaching users how to cook based on food
photos, and so on. In our experiment, we found that the model trained on short
image caption pairs could produce unnatural language outputs (e.g., repetition
and fragmentation). To address this problem, we curate a detailed image
description dataset in the second stage to finetune the model, which
consequently improves the model's generation reliability and overall usability.
Our code, pre-trained model, and collected dataset are available at
https://minigpt-4.github.io/.


# Avalon's Game of Thoughts: Battle Against Deception through Recursive Contemplation

[Link to the paper](http://arxiv.org/abs/2310.01320v1)

## Authors
- Shenzhi Wang
- Chang Liu
- Zilong Zheng
- Siyuan Qi
- Shuo Chen
- Qisen Yang
- Andrew Zhao
- Chaofei Wang
- Shiji Song
- Gao Huang

## Summary
  Recent breakthroughs in large language models (LLMs) have brought remarkable
success in the field of LLM-as-Agent. Nevertheless, a prevalent assumption is
that the information processed by LLMs is consistently honest, neglecting the
pervasive deceptive or misleading information in human society and AI-generated
content. This oversight makes LLMs susceptible to malicious manipulations,
potentially resulting in detrimental outcomes. This study utilizes the
intricate Avalon game as a testbed to explore LLMs' potential in deceptive
environments. Avalon, full of misinformation and requiring sophisticated logic,
manifests as a "Game-of-Thoughts". Inspired by the efficacy of humans'
recursive thinking and perspective-taking in the Avalon game, we introduce a
novel framework, Recursive Contemplation (ReCon), to enhance LLMs' ability to
identify and counteract deceptive information. ReCon combines formulation and
refinement contemplation processes; formulation contemplation produces initial
thoughts and speech, while refinement contemplation further polishes them.
Additionally, we incorporate first-order and second-order perspective
transitions into these processes respectively. Specifically, the first-order
allows an LLM agent to infer others' mental states, and the second-order
involves understanding how others perceive the agent's mental state. After
integrating ReCon with different LLMs, extensive experiment results from the
Avalon game indicate its efficacy in aiding LLMs to discern and maneuver around
deceptive information without extra fine-tuning and data. Finally, we offer a
possible explanation for the efficacy of ReCon and explore the current
limitations of LLMs in terms of safety, reasoning, speaking style, and format,
potentially furnishing insights for subsequent research.


# Large Language Models in Fault Localisation

[Link to the paper](http://arxiv.org/abs/2308.15276v3)

## Authors
- Yonghao Wu
- Zheng Li
- Jie M. Zhang
- Mike Papadakis
- Mark Harman
- Yong Liu

## Summary
  Large Language Models (LLMs) have shown promise in multiple software
engineering tasks including code generation, program repair, code
summarisation, and test generation. Fault localisation is instrumental in
enabling automated debugging and repair of programs and was prominently
featured as a highlight during the launch event of ChatGPT-4. Nevertheless, the
performance of LLMs compared to state-of-the-art methods, as well as the impact
of prompt design and context length on their efficacy, remains unclear. To fill
this gap, this paper presents an in-depth investigation into the capability of
ChatGPT-3.5 and ChatGPT-4, the two state-of-the-art LLMs, on fault
localisation. Using the widely-adopted large-scale Defects4J dataset, we
compare the two LLMs with the existing fault localisation techniques. We also
investigate the consistency of LLMs in fault localisation, as well as how
prompt engineering and the length of code context affect the fault localisation
effectiveness.
  Our findings demonstrate that within function-level context, ChatGPT-4
outperforms all the existing fault localisation methods. Additional error logs
can further improve ChatGPT models' localisation accuracy and consistency, with
an average 46.9% higher accuracy over the state-of-the-art baseline SmartFL on
the Defects4J dataset in terms of TOP-1 metric. However, when the code context
of the Defects4J dataset expands to the class-level, ChatGPT-4's performance
suffers a significant drop, with 49.9% lower accuracy than SmartFL under TOP-1
metric. These observations indicate that although ChatGPT can effectively
localise faults under specific conditions, limitations are evident. Further
research is needed to fully harness the potential of LLMs like ChatGPT for
practical fault localisation applications.


# Co-audit: tools to help humans double-check AI-generated content

[Link to the paper](http://arxiv.org/abs/2310.01297v1)

## Authors
- Andrew D. Gordon
- Carina Negreanu
- José Cambronero
- Rasika Chakravarthy
- Ian Drosos
- Hao Fang
- Bhaskar Mitra
- Hannah Richardson
- Advait Sarkar
- Stephanie Simmons
- Jack Williams
- Ben Zorn

## Summary
  Users are increasingly being warned to check AI-generated content for
correctness. Still, as LLMs (and other generative models) generate more complex
output, such as summaries, tables, or code, it becomes harder for the user to
audit or evaluate the output for quality or correctness. Hence, we are seeing
the emergence of tool-assisted experiences to help the user double-check a
piece of AI-generated content. We refer to these as co-audit tools. Co-audit
tools complement prompt engineering techniques: one helps the user construct
the input prompt, while the other helps them check the output response. As a
specific example, this paper describes recent research on co-audit tools for
spreadsheet computations powered by generative models. We explain why co-audit
experiences are essential for any application of generative AI where quality is
important and errors are consequential (as is common in spreadsheet
computations). We propose a preliminary list of principles for co-audit, and
outline research challenges.


# Knowledge Crosswords: Geometric Reasoning over Structured Knowledge with Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.01290v1)

## Authors
- Wenxuan Ding
- Shangbin Feng
- Yuhan Liu
- Zhaoxuan Tan
- Vidhisha Balachandran
- Tianxing He
- Yulia Tsvetkov

## Summary
  Large language models (LLMs) are widely adopted in knowledge-intensive tasks
and have achieved impressive performance thanks to their knowledge abilities.
While LLMs have demonstrated outstanding performance on atomic or linear
(multi-hop) QA tasks, whether they can reason in knowledge-rich scenarios with
interweaving constraints remains an underexplored problem. In this work, we
propose geometric reasoning over structured knowledge, where pieces of
knowledge are connected in a graph structure and models need to fill in the
missing information. Such geometric knowledge reasoning would require the
ability to handle structured knowledge, reason with uncertainty, verify facts,
and backtrack when an error occurs. We propose Knowledge Crosswords, a
multi-blank QA dataset where each problem consists of a natural language
question representing the geometric constraints of an incomplete entity
network, where LLMs are tasked with working out the missing entities while
meeting all factual constraints. Knowledge Crosswords contains 2,101 individual
problems, covering various knowledge domains and further divided into three
difficulty levels. We conduct extensive experiments to evaluate existing LLM
prompting approaches on the Knowledge Crosswords benchmark. We additionally
propose two new approaches, Staged Prompting and Verify-All, to augment LLMs'
ability to backtrack and verify structured constraints. Our results demonstrate
that while baseline approaches perform well on easier problems but struggle
with hard ones, our proposed Verify-All outperforms other methods by a large
margin and is more robust with hard problems. Further analysis reveals that
LLMs' ability of geometric reasoning over structured knowledge is still far
from robust or perfect, susceptible to confounders such as the order of
options, certain structural patterns, assumption of existence of correct
answer, and more.


# Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models

[Link to the paper](http://arxiv.org/abs/2306.08018v3)

## Authors
- Yin Fang
- Xiaozhuan Liang
- Ningyu Zhang
- Kangwei Liu
- Rui Huang
- Zhuo Chen
- Xiaohui Fan
- Huajun Chen

## Summary
  Large Language Models (LLMs), with their remarkable task-handling
capabilities and innovative outputs, have catalyzed significant advancements
across a spectrum of fields. However, their proficiency within specialized
domains such as biomolecular studies remains limited. To address this
challenge, we introduce Mol-Instructions, a comprehensive instruction dataset
designed for the biomolecular domain. Mol-Instructions encompasses three key
components: molecule-oriented instructions, protein-oriented instructions, and
biomolecular text instructions. Each component aims to improve the
understanding and prediction capabilities of LLMs concerning biomolecular
features and behaviors. Through extensive instruction tuning experiments on
LLMs, we demonstrate the effectiveness of Mol-Instructions in enhancing large
models' performance in the intricate realm of biomolecular studies, thus
fostering progress in the biomolecular research community. Mol-Instructions is
publicly available for ongoing research and will undergo regular updates to
enhance its applicability.


# Two Failures of Self-Consistency in the Multi-Step Reasoning of LLMs

[Link to the paper](http://arxiv.org/abs/2305.14279v3)

## Authors
- Angelica Chen
- Jason Phang
- Alicia Parrish
- Vishakh Padmakumar
- Chen Zhao
- Samuel R. Bowman
- Kyunghyun Cho

## Summary
  Large language models (LLMs) have achieved widespread success on a variety of
in-context few-shot tasks, but this success is typically evaluated via
correctness rather than consistency. We argue that self-consistency is an
important criteria for valid multi-step reasoning in tasks where the solution
is composed of the answers to multiple sub-steps. We propose two types of
self-consistency that are particularly important for multi-step reasoning --
hypothetical consistency (a model's ability to predict what its output would be
in a hypothetical other context) and compositional consistency (consistency of
a model's final outputs when intermediate sub-steps are replaced with the
model's outputs for those steps). We demonstrate that multiple variants of the
GPT-3/-4 models exhibit poor consistency rates across both types of consistency
on a variety of tasks.


# SPELL: Semantic Prompt Evolution based on a LLM

[Link to the paper](http://arxiv.org/abs/2310.01260v1)

## Authors
- Yujian Betterest Li
- Kai Wu

## Summary
  Prompt engineering is a new paradigm for enhancing the performance of trained
neural network models. For optimizing text-style prompts, existing methods
usually individually operate small portions of a text step by step, which
either breaks the fluency or could not globally adjust a prompt. Since large
language models (LLMs) have powerful ability of generating coherent texts token
by token, can we utilize LLMs for improving prompts? Based on this motivation,
in this paper, considering a trained LLM as a text generator, we attempt to
design a black-box evolution algorithm for automatically optimizing texts,
namely SPELL (Semantic Prompt Evolution based on a LLM). The proposed method is
evaluated with different LLMs and evolution parameters in different text tasks.
Experimental results show that SPELL could rapidly improve the prompts indeed.
We further explore the evolution process and discuss on the limitations,
potential possibilities and future work.


# Listen, Think, and Understand

[Link to the paper](http://arxiv.org/abs/2305.10790v2)

## Authors
- Yuan Gong
- Hongyin Luo
- Alexander H. Liu
- Leonid Karlinsky
- James Glass

## Summary
  The ability of artificial intelligence (AI) systems to perceive and
comprehend audio signals is crucial for many applications. Although significant
progress has been made in this area since the development of AudioSet, most
existing models are designed to map audio inputs to pre-defined, discrete sound
label sets. In contrast, humans possess the ability to not only classify sounds
into general categories, but also to listen to the finer details of the sounds,
explain the reason for the predictions, think about what the sound infers, and
understand the scene and what action needs to be taken, if any. Such
capabilities beyond perception are not yet present in existing audio models. On
the other hand, modern large language models (LLMs) exhibit emerging reasoning
ability but they lack audio perception capabilities. Therefore, we ask the
question: can we build a model that has both audio perception and a reasoning
ability?
  In this paper, we propose a new audio foundation model, called LTU (Listen,
Think, and Understand). To train LTU, we created a new OpenAQA-5M dataset
consisting of 1.9 million closed-ended and 3.7 million open-ended, diverse
(audio, question, answer) tuples, and have used an autoregressive training
framework with a perception-to-understanding curriculum. LTU demonstrates
strong performance and generalization ability on conventional audio tasks such
as classification and captioning. More importantly, it exhibits emerging audio
reasoning and comprehension abilities that are absent in existing audio models.
To the best of our knowledge, LTU is one of the first multimodal large language
models that focus on general audio (rather than just speech) understanding.


# MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning

[Link to the paper](http://arxiv.org/abs/2309.07915v2)

## Authors
- Haozhe Zhao
- Zefan Cai
- Shuzheng Si
- Xiaojian Ma
- Kaikai An
- Liang Chen
- Zixuan Liu
- Sheng Wang
- Wenjuan Han
- Baobao Chang

## Summary
  Since the resurgence of deep learning, vision-language models (VLMs) enhanced
by large language models (LLMs) have grown exponentially in popularity.
However, while LLMs can utilize extensive background knowledge and task
information with in-context learning, most VLMs still struggle with
understanding complex multi-modal prompts with multiple images, making VLMs
less effective in downstream vision-language tasks. In this paper, we address
the limitation above by 1) introducing MMICL, a new approach to allow the VLM
to deal with multi-modal inputs efficiently; 2) proposing a novel context
scheme to augment the in-context learning ability of the VLM; 3) constructing
the Multi-modal In-Context Learning (MIC) dataset, designed to enhance the
VLM's ability to understand complex multi-modal prompts. Our experiments
confirm that MMICL achieves new state-of-the-art zero-shot performance on a
wide range of general vision-language tasks, especially for complex benchmarks,
including MME and MMBench. Our analysis demonstrates that MMICL effectively
tackles the challenge of complex multi-modal prompt understanding and emerges
the impressive ICL ability. Furthermore, we observe that MMICL successfully
alleviates language bias in VLMs, a common issue for VLMs that often leads to
hallucination when faced with extensive textual context.


# Making LLaMA SEE and Draw with SEED Tokenizer

[Link to the paper](http://arxiv.org/abs/2310.01218v1)

## Authors
- Yuying Ge
- Sijie Zhao
- Ziyun Zeng
- Yixiao Ge
- Chen Li
- Xintao Wang
- Ying Shan

## Summary
  The great success of Large Language Models (LLMs) has expanded the potential
of multimodality, contributing to the gradual evolution of General Artificial
Intelligence (AGI). A true AGI agent should not only possess the capability to
perform predefined multi-tasks but also exhibit emergent abilities in an
open-world context. However, despite the considerable advancements made by
recent multimodal LLMs, they still fall short in effectively unifying
comprehension and generation tasks, let alone open-world emergent abilities. We
contend that the key to overcoming the present impasse lies in enabling text
and images to be represented and processed interchangeably within a unified
autoregressive Transformer. To this end, we introduce SEED, an elaborate image
tokenizer that empowers LLMs with the ability to SEE and Draw at the same time.
We identify two crucial design principles: (1) Image tokens should be
independent of 2D physical patch positions and instead be produced with a 1D
causal dependency, exhibiting intrinsic interdependence that aligns with the
left-to-right autoregressive prediction mechanism in LLMs. (2) Image tokens
should capture high-level semantics consistent with the degree of semantic
abstraction in words, and be optimized for both discriminativeness and
reconstruction during the tokenizer training phase. With SEED tokens, LLM is
able to perform scalable multimodal autoregression under its original training
recipe, i.e., next-word prediction. SEED-LLaMA is therefore produced by
large-scale pretraining and instruction tuning on the interleaved textual and
visual data, demonstrating impressive performance on a broad range of
multimodal comprehension and generation tasks. More importantly, SEED-LLaMA has
exhibited compositional emergent abilities such as multi-turn in-context
multimodal generation, acting like your AI assistant.


# Label Supervised LLaMA Finetuning

[Link to the paper](http://arxiv.org/abs/2310.01208v1)

## Authors
- Zongxi Li
- Xianming Li
- Yuzhang Liu
- Haoran Xie
- Jing Li
- Fu-lee Wang
- Qing Li
- Xiaoqin Zhong

## Summary
  The recent success of Large Language Models (LLMs) has gained significant
attention in both academia and industry. Substantial efforts have been made to
enhance the zero- and few-shot generalization capabilities of open-source LLMs
through finetuning. Currently, the prevailing approach is instruction-tuning,
which trains LLMs to complete real-world tasks by generating responses guided
by natural language instructions. It is worth noticing that such an approach
may underperform in sequence and token classification tasks. Unlike text
generation tasks, classification tasks have a limited label space, where
precise label prediction is more appreciated than generating diverse and
human-like responses. Prior research has unveiled that instruction-tuned LLMs
cannot outperform BERT, prompting us to explore the potential of leveraging
latent representations from LLMs for supervised label prediction. In this
paper, we introduce a label-supervised adaptation for LLMs, which aims to
finetuning the model with discriminant labels. We evaluate this approach with
Label Supervised LLaMA (LS-LLaMA), based on LLaMA-2-7B, a relatively
small-scale LLM, and can be finetuned on a single GeForce RTX4090 GPU. We
extract latent representations from the final LLaMA layer and project them into
the label space to compute the cross-entropy loss. The model is finetuned by
Low-Rank Adaptation (LoRA) to minimize this loss. Remarkably, without intricate
prompt engineering or external knowledge, LS-LLaMA substantially outperforms
LLMs ten times its size in scale and demonstrates consistent improvements
compared to robust baselines like BERT-Large and RoBERTa-Large in text
classification. Moreover, by removing the causal mask from decoders, LS-unLLaMA
achieves the state-of-the-art performance in named entity recognition (NER).
Our work will shed light on a novel approach to adapting LLMs for various
downstream tasks.


# NarrativePlay: Interactive Narrative Understanding

[Link to the paper](http://arxiv.org/abs/2310.01459v1)

## Authors
- Runcong Zhao
- Wenjia Zhang
- Jiazheng Li
- Lixing Zhu
- Yanran Li
- Yulan He
- Lin Gui

## Summary
  In this paper, we introduce NarrativePlay, a novel system that allows users
to role-play a fictional character and interact with other characters in
narratives such as novels in an immersive environment. We leverage Large
Language Models (LLMs) to generate human-like responses, guided by personality
traits extracted from narratives. The system incorporates auto-generated visual
display of narrative settings, character portraits, and character speech,
greatly enhancing user experience. Our approach eschews predefined sandboxes,
focusing instead on main storyline events extracted from narratives from the
perspective of a user-selected character. NarrativePlay has been evaluated on
two types of narratives, detective and adventure stories, where users can
either explore the world or improve their favorability with the narrative
characters through conversations.


# Large Language Model-Powered Smart Contract Vulnerability Detection: New Perspectives

[Link to the paper](http://arxiv.org/abs/2310.01152v1)

## Authors
- Sihao Hu
- Tiansheng Huang
- Fatih İlhan
- Selim Fukan Tekin
- Ling Liu

## Summary
  This paper provides a systematic analysis of the opportunities, challenges,
and potential solutions of harnessing LLMs to dig out vulnerabilities within
smart contracts based on our ongoing research. For the smart contract
vulnerability detection task, the key to achieving practical usability lies in
detecting as many true vulnerabilities as possible while minimizing the number
of false positives. However, our empirical study using LLM as a detection tool
reveals interesting yet contradictory findings: generating more answers with
higher randomness largely increases the likelihood of a correct answer being
generated while inevitably leading to a higher number of false positives,
resulting in exhaustive manual verification efforts. To mitigate this tension,
we propose an adversarial framework dubbed GPTLens that breaks the traditional
one-stage detection into two synergistic stages $-$ generation and
discrimination, for progressive detection and fine-tuning, wherein the LLM
plays dual roles, i.e., auditor and critic, respectively. The goal of auditor
is to identify multiple diverse vulnerabilities with intermediate reasoning,
while the goal of critic is to evaluate the accuracy of identified
vulnerabilities and to examine the integrity of the detection reasoning.
Experimental results and illustrative examples demonstrate that auditor and
critic work together harmoniously to yield significant improvements over the
traditional one-stage detection. GPTLens is intuitive, strategic, and entirely
LLM-driven without relying on specialist expertise in smart contracts,
showcasing its methodical generality and potential to detect a broad spectrum
of vulnerabilities. Our code is available at:
https://github.com/git-disl/GPTLens.


# Automated Evaluation of Classroom Instructional Support with LLMs and BoWs: Connecting Global Predictions to Specific Feedback

[Link to the paper](http://arxiv.org/abs/2310.01132v1)

## Authors
- Jacob Whitehill
- Jennifer LoCasale-Crouch

## Summary
  With the aim to provide teachers with more specific, frequent, and actionable
feedback about their teaching, we explore how Large Language Models (LLMs) can
be used to estimate ``Instructional Support'' domain scores of the CLassroom
Assessment Scoring System (CLASS), a widely used observation protocol. We
design a machine learning architecture that uses either zero-shot prompting of
Meta's Llama2, and/or a classic Bag of Words (BoW) model, to classify
individual utterances of teachers' speech (transcribed automatically using
OpenAI's Whisper) for the presence of 11 behavioral indicators of Instructional
Support. Then, these utterance-level judgments are aggregated over an entire
15-min observation session to estimate a global CLASS score. Experiments on two
CLASS-coded datasets of toddler and pre-kindergarten classrooms indicate that
(1) automatic CLASS Instructional Support estimation accuracy using the
proposed method (Pearson $R$ up to $0.46$) approaches human inter-rater
reliability (up to $R=0.55$); (2) LLMs yield slightly greater accuracy than BoW
for this task; and (3) the best models often combined features extracted from
both LLM and BoW. Finally, (4) we illustrate how the model's outputs can be
visualized at the utterance level to provide teachers with explainable feedback
on which utterances were most positively or negatively correlated with specific
CLASS dimensions.


# mBLIP: Efficient Bootstrapping of Multilingual Vision-LLMs

[Link to the paper](http://arxiv.org/abs/2307.06930v2)

## Authors
- Gregor Geigle
- Abhay Jain
- Radu Timofte
- Goran Glavaš

## Summary
  Modular vision-language models (Vision-LLMs) align pretrained image encoders
with frozen large language models (LLMs), representing a computationally much
more efficient alternative to end-to-end training of large vision-language
models from scratch, which is prohibitively expensive for most researchers and
practitioners. Vision-LLMs instead post-hoc condition LLMs to `understand' the
output of an image encoder. With the abundance of readily available
high-quality English image-text data as well as monolingual English LLMs, the
research focus has been on English-only Vision-LLMs. Multilingual
vision-language models are still predominantly obtained via expensive
end-to-end pretraining, resulting in comparatively smaller models, trained on
limited multilingual image data supplemented with text-only multilingual
corpora. In this work, we present mBLIP, the first multilingual Vision-LLM,
which we obtain in a computationally efficient manner -- on consumer hardware
and using only a few million training examples -- by leveraging a pretrained
multilingual LLM. To this end, we \textit{re-align} an image encoder previously
tuned to an English LLM to a new, multilingual LLM -- for this, we leverage
multilingual data from a mix of vision-and-language tasks, which we obtain by
machine-translating high-quality English data to 95 languages. On the IGLUE
benchmark, mBLIP yields results competitive with state-of-the-art models.
Moreover, in image captioning on XM3600, mBLIP (zero-shot) even outperforms
PaLI-X (a model with 55B parameters). Compared to these very large multilingual
vision-language models trained from scratch, we obtain mBLIP by training orders
of magnitude fewer parameters on magnitudes less data. We release our model and
code at \url{https://github.com/gregor-ge/mBLIP}.


# Text Data Augmentation in Low-Resource Settings via Fine-Tuning of Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.01119v1)

## Authors
- Jean Kaddour
- Qi Liu

## Summary
  The in-context learning ability of large language models (LLMs) enables them
to generalize to novel downstream tasks with relatively few labeled examples.
However, they require enormous computational resources to be deployed.
Alternatively, smaller models can solve specific tasks if fine-tuned with
enough labeled examples. These examples, however, are expensive to obtain. In
pursuit of the best of both worlds, we study the annotation and generation of
fine-tuning training data via fine-tuned teacher LLMs to improve the downstream
performance of much smaller models. In four text classification and two text
generation tasks, we find that both data generation and annotation dramatically
improve the respective downstream model's performance, occasionally
necessitating only a minor fraction of the original training dataset.


# GraphText: Graph Reasoning in Text Space

[Link to the paper](http://arxiv.org/abs/2310.01089v1)

## Authors
- Jianan Zhao
- Le Zhuo
- Yikang Shen
- Meng Qu
- Kai Liu
- Michael Bronstein
- Zhaocheng Zhu
- Jian Tang

## Summary
  Large Language Models (LLMs) have gained the ability to assimilate human
knowledge and facilitate natural language interactions with both humans and
other LLMs. However, despite their impressive achievements, LLMs have not made
significant advancements in the realm of graph machine learning. This
limitation arises because graphs encapsulate distinct relational data, making
it challenging to transform them into natural language that LLMs understand. In
this paper, we bridge this gap with a novel framework, GraphText, that
translates graphs into natural language. GraphText derives a graph-syntax tree
for each graph that encapsulates both the node attributes and inter-node
relationships. Traversal of the tree yields a graph text sequence, which is
then processed by an LLM to treat graph tasks as text generation tasks.
Notably, GraphText offers multiple advantages. It introduces training-free
graph reasoning: even without training on graph data, GraphText with ChatGPT
can achieve on par with, or even surpassing, the performance of
supervised-trained graph neural networks through in-context learning (ICL).
Furthermore, GraphText paves the way for interactive graph reasoning, allowing
both humans and LLMs to communicate with the model seamlessly using natural
language. These capabilities underscore the vast, yet-to-be-explored potential
of LLMs in the domain of graph machine learning.


# Towards human-like spoken dialogue generation between AI agents from written dialogue

[Link to the paper](http://arxiv.org/abs/2310.01088v1)

## Authors
- Kentaro Mitsui
- Yukiya Hono
- Kei Sawada

## Summary
  The advent of large language models (LLMs) has made it possible to generate
natural written dialogues between two agents. However, generating human-like
spoken dialogues from these written dialogues remains challenging. Spoken
dialogues have several unique characteristics: they frequently include
backchannels and laughter, and the smoothness of turn-taking significantly
influences the fluidity of conversation. This study proposes CHATS - CHatty
Agents Text-to-Speech - a discrete token-based system designed to generate
spoken dialogues based on written dialogues. Our system can generate speech for
both the speaker side and the listener side simultaneously, using only the
transcription from the speaker side, which eliminates the need for
transcriptions of backchannels or laughter. Moreover, CHATS facilitates natural
turn-taking; it determines the appropriate duration of silence after each
utterance in the absence of overlap, and it initiates the generation of
overlapping speech based on the phoneme sequence of the next utterance in case
of overlap. Experimental evaluations indicate that CHATS outperforms the
text-to-speech baseline, producing spoken dialogues that are more interactive
and fluid while retaining clarity and intelligibility.


# Back to the Future: Towards Explainable Temporal Reasoning with Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.01074v1)

## Authors
- Chenhan Yuan
- Qianqian Xie
- Jimin Huang
- Sophia Ananiadou

## Summary
  Temporal reasoning is a crucial NLP task, providing a nuanced understanding
of time-sensitive contexts within textual data. Although recent advancements in
LLMs have demonstrated their potential in temporal reasoning, the predominant
focus has been on tasks such as temporal expression and temporal relation
extraction. These tasks are primarily designed for the extraction of direct and
past temporal cues and to engage in simple reasoning processes. A significant
gap remains when considering complex reasoning tasks such as event forecasting,
which requires multi-step temporal reasoning on events and prediction on the
future timestamp. Another notable limitation of existing methods is their
incapability to provide an illustration of their reasoning process, hindering
explainability. In this paper, we introduce the first task of explainable
temporal reasoning, to predict an event's occurrence at a future timestamp
based on context which requires multiple reasoning over multiple events, and
subsequently provide a clear explanation for their prediction. Our task offers
a comprehensive evaluation of both the LLMs' complex temporal reasoning
ability, the future event prediction ability, and explainability-a critical
attribute for AI applications. To support this task, we present the first
multi-source instruction-tuning dataset of explainable temporal reasoning
(ExpTime) with 26k derived from the temporal knowledge graph datasets and their
temporal reasoning paths, using a novel knowledge-graph-instructed-generation
strategy. Based on the dataset, we propose the first open-source LLM series
TimeLlaMA based on the foundation LlaMA2, with the ability of instruction
following for explainable temporal reasoning. We compare the performance of our
method and a variety of LLMs, where our method achieves the state-of-the-art
performance of temporal prediction and explanation.


# Lyra: Orchestrating Dual Correction in Automated Theorem Proving

[Link to the paper](http://arxiv.org/abs/2309.15806v2)

## Authors
- Chuanyang Zheng
- Haiming Wang
- Enze Xie
- Zhengying Liu
- Jiankai Sun
- Huajian Xin
- Jianhao Shen
- Zhenguo Li
- Yu Li

## Summary
  Large Language Models (LLMs) present an intriguing avenue for exploration in
the field of formal theorem proving. Nevertheless, their full potential,
particularly concerning the mitigation of hallucinations and refinement through
prover error messages, remains an area that has yet to be thoroughly
investigated. To enhance the effectiveness of LLMs in the field, we introduce
the Lyra, a new framework that employs two distinct correction mechanisms: Tool
Correction (TC) and Conjecture Correction (CC). To implement Tool Correction in
the post-processing of formal proofs, we leverage prior knowledge to utilize
predefined prover tools (e.g., Sledgehammer) for guiding the replacement of
incorrect tools. Tool Correction significantly contributes to mitigating
hallucinations, thereby improving the overall accuracy of the proof. In
addition, we introduce Conjecture Correction, an error feedback mechanism
designed to interact with prover to refine formal proof conjectures with prover
error messages. Compared to the previous refinement framework, the proposed
Conjecture Correction refines generation with instruction but does not collect
paired (generation, error & refinement) prompts. Our method has achieved
state-of-the-art (SOTA) performance on both miniF2F validation (48.0% -> 55.3%)
and test (45.5% -> 51.2%). We also present 3 IMO problems solved by Lyra. We
believe Tool Correction (post-process for hallucination mitigation) and
Conjecture Correction (subgoal adjustment from interaction with environment)
could provide a promising avenue for future research in this field.


# Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning

[Link to the paper](http://arxiv.org/abs/2310.01061v1)

## Authors
- Linhao Luo
- Yuan-Fang Li
- Gholamreza Haffari
- Shirui Pan

## Summary
  Large language models (LLMs) have demonstrated impressive reasoning abilities
in complex tasks. However, they lack up-to-date knowledge and experience
hallucinations during reasoning, which can lead to incorrect reasoning
processes and diminish their performance and trustworthiness. Knowledge graphs
(KGs), which capture vast amounts of facts in a structured format, offer a
reliable source of knowledge for reasoning. Nevertheless, existing KG-based LLM
reasoning methods only treat KGs as factual knowledge bases and overlook the
importance of their structural information for reasoning. In this paper, we
propose a novel method called reasoning on graphs (RoG) that synergizes LLMs
with KGs to enable faithful and interpretable reasoning. Specifically, we
present a planning-retrieval-reasoning framework, where RoG first generates
relation paths grounded by KGs as faithful plans. These plans are then used to
retrieve valid reasoning paths from the KGs for LLMs to conduct faithful
reasoning. Furthermore, RoG not only distills knowledge from KGs to improve the
reasoning ability of LLMs through training but also allows seamless integration
with any arbitrary LLMs during inference. Extensive experiments on two
benchmark KGQA datasets demonstrate that RoG achieves state-of-the-art
performance on KG reasoning tasks and generates faithful and interpretable
reasoning results.


# L2CEval: Evaluating Language-to-Code Generation Capabilities of Large Language Models

[Link to the paper](http://arxiv.org/abs/2309.17446v2)

## Authors
- Ansong Ni
- Pengcheng Yin
- Yilun Zhao
- Martin Riddell
- Troy Feng
- Rui Shen
- Stephen Yin
- Ye Liu
- Semih Yavuz
- Caiming Xiong
- Shafiq Joty
- Yingbo Zhou
- Dragomir Radev
- Arman Cohan

## Summary
  Recently, large language models (LLMs), especially those that are pretrained
on code, have demonstrated strong capabilities in generating programs from
natural language inputs in a few-shot or even zero-shot manner. Despite
promising results, there is a notable lack of a comprehensive evaluation of
these models language-to-code generation capabilities. Existing studies often
focus on specific tasks, model architectures, or learning paradigms, leading to
a fragmented understanding of the overall landscape. In this work, we present
L2CEval, a systematic evaluation of the language-to-code generation
capabilities of LLMs on 7 tasks across the domain spectrum of semantic parsing,
math reasoning and Python programming, analyzing the factors that potentially
affect their performance, such as model size, pretraining data, instruction
tuning, and different prompting methods. In addition to assessing model
performance, we measure confidence calibration for the models and conduct human
evaluations of the output programs. This enables us to identify and analyze the
typical failure modes across various tasks and models. L2CEval offers a
comprehensive understanding of the capabilities and limitations of LLMs in
language-to-code generation. We also release the evaluation framework and all
model outputs, hoping to lay the groundwork for further future research in this
domain.


# ARN: A Comprehensive Framework and Dataset for Analogical Reasoning on Narratives

[Link to the paper](http://arxiv.org/abs/2310.00996v1)

## Authors
- Zhivar Sourati
- Filip Ilievski
- Pia Sommerauer

## Summary
  Analogical reasoning is one of the prime abilities of humans and is linked to
creativity and scientific discoveries. This ability has been studied
extensively in natural language processing (NLP) as well as in cognitive
psychology by proposing various benchmarks and evaluation setups. Yet, a
substantial gap exists between evaluations of analogical reasoning in cognitive
psychology and NLP. Our aim is to bridge this by computationally adapting
theories related to analogical reasoning from cognitive psychology in the
context of narratives and developing an evaluation framework large in scale.
More concretely, we propose the task of matching narratives based on system
mappings and release the Analogical Reasoning on Narratives (ARN) dataset. To
create the dataset, we devise a framework inspired by cognitive psychology
theories about analogical reasoning to utilize narratives and their components
to form mappings of different abstractness levels. These mappings are then
leveraged to create pairs of analogies and disanalogies/distractors with more
than 1k triples of query narratives, analogies, and distractors. We cover four
categories of far/near analogies and far/near distractors that allow us to
study analogical reasoning in models from distinct perspectives. In this study,
we evaluate different large language models (LLMs) on this task. Our results
demonstrate that LLMs struggle to recognize higher-order mappings when they are
not accompanied by lower-order mappings (far analogies) and show better
performance when all mappings are present simultaneously (near analogies). We
observe that in all the settings, the analogical reasoning abilities of LLMs
can be easily impaired by near distractors that form lower-order mappings with
the query narratives.


# Resolving Knowledge Conflicts in Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.00935v1)

## Authors
- Yike Wang
- Shangbin Feng
- Heng Wang
- Weijia Shi
- Vidhisha Balachandran
- Tianxing He
- Yulia Tsvetkov

## Summary
  Large language models (LLMs) often encounter knowledge conflicts, scenarios
where discrepancy arises between the internal parametric knowledge of LLMs and
non-parametric information provided in the prompt context. In this work we ask
what are the desiderata for LLMs when a knowledge conflict arises and whether
existing LLMs fulfill them. We posit that LLMs should 1) identify knowledge
conflicts, 2) pinpoint conflicting information segments, and 3) provide
distinct answers or viewpoints in conflicting scenarios. To this end, we
introduce KNOWLEDGE CONFLICT, an evaluation framework for simulating contextual
knowledge conflicts and quantitatively evaluating to what extent LLMs achieve
these goals. KNOWLEDGE CONFLICT includes diverse and complex situations of
knowledge conflict, knowledge from diverse entities and domains, two synthetic
conflict creation methods, and settings with progressively increasing
difficulty to reflect realistic knowledge conflicts. Extensive experiments with
the KNOWLEDGE CONFLICT framework reveal that while LLMs perform well in
identifying the existence of knowledge conflicts, they struggle to determine
the specific conflicting knowledge and produce a response with distinct answers
amidst conflicting information. To address these challenges, we propose new
instruction-based approaches that augment LLMs to better achieve the three
goals. Further analysis shows that abilities to tackle knowledge conflicts are
greatly impacted by factors such as knowledge domain and prompt text, while
generating robust responses to knowledge conflict scenarios remains an open
research question.


# On decoder-only architecture for speech-to-text and large language model integration

[Link to the paper](http://arxiv.org/abs/2307.03917v3)

## Authors
- Jian Wu
- Yashesh Gaur
- Zhuo Chen
- Long Zhou
- Yimeng Zhu
- Tianrui Wang
- Jinyu Li
- Shujie Liu
- Bo Ren
- Linquan Liu
- Yu Wu

## Summary
  Large language models (LLMs) have achieved remarkable success in the field of
natural language processing, enabling better human-computer interaction using
natural language. However, the seamless integration of speech signals into LLMs
has not been explored well. The "decoder-only" architecture has also not been
well studied for speech processing tasks. In this research, we introduce
Speech-LLaMA, a novel approach that effectively incorporates acoustic
information into text-based large language models. Our method leverages
Connectionist Temporal Classification and a simple audio encoder to map the
compressed acoustic features to the continuous semantic space of the LLM. In
addition, we further probe the decoder-only architecture for speech-to-text
tasks by training a smaller scale randomly initialized speech-LLaMA model from
speech-text paired data alone. We conduct experiments on multilingual
speech-to-text translation tasks and demonstrate a significant improvement over
strong baselines, highlighting the potential advantages of decoder-only models
for speech-to-text conversion.


# LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud

[Link to the paper](http://arxiv.org/abs/2309.17157v2)

## Authors
- Mengke Zhang
- Tianxing He
- Tianle Wang
- Lu Mi
- Fatemehsadat Mireshghallah
- Binyi Chen
- Hao Wang
- Yulia Tsvetkov

## Summary
  In the current user-server interaction paradigm of prompted generation with
large language models (LLM) on cloud, the server fully controls the generation
process, which leaves zero options for users who want to keep the generated
text to themselves. We propose LatticeGen, a cooperative framework in which the
server still handles most of the computation while the user controls the
sampling operation. The key idea is that the true generated sequence is mixed
with noise tokens by the user and hidden in a noised lattice. Considering
potential attacks from a hypothetically malicious server and how the user can
defend against it, we propose the repeated beam-search attack and the mixing
noise scheme. In our experiments we apply LatticeGen to protect both prompt and
generation. It is shown that while the noised lattice degrades generation
quality, LatticeGen successfully protects the true generation to a remarkable
degree under strong attacks (more than 50% of the semantic remains hidden as
measured by BERTScore).


# All Languages Matter: On the Multilingual Safety of Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.00905v1)

## Authors
- Wenxuan Wang
- Zhaopeng Tu
- Chang Chen
- Youliang Yuan
- Jen-tse Huang
- Wenxiang Jiao
- Michael R. Lyu

## Summary
  Safety lies at the core of developing and deploying large language models
(LLMs). However, previous safety benchmarks only concern the safety in one
language, e.g. the majority language in the pretraining data such as English.
In this work, we build the first multilingual safety benchmark for LLMs,
XSafety, in response to the global deployment of LLMs in practice. XSafety
covers 14 kinds of commonly used safety issues across 10 languages that span
several language families. We utilize XSafety to empirically study the
multilingual safety for 4 widely-used LLMs, including both close-API and
open-source models. Experimental results show that all LLMs produce
significantly more unsafe responses for non-English queries than English ones,
indicating the necessity of developing safety alignment for non-English
languages. In addition, we propose several simple and effective prompting
methods to improve the multilingual safety of ChatGPT by evoking safety
knowledge and improving cross-lingual generalization of safety alignment. Our
prompting method can significantly reduce the ratio of unsafe responses from
19.1% to 9.7% for non-English queries. We release our data at
https://github.com/Jarviswang94/Multilingual_safety_benchmark.


# DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models

[Link to the paper](http://arxiv.org/abs/2310.00902v1)

## Authors
- Yongchan Kwon
- Eric Wu
- Kevin Wu
- James Zou

## Summary
  Quantifying the impact of training data points is crucial for understanding
the outputs of machine learning models and for improving the transparency of
the AI pipeline. The influence function is a principled and popular data
attribution method, but its computational cost often makes it challenging to
use. This issue becomes more pronounced in the setting of large language models
and text-to-image models. In this work, we propose DataInf, an efficient
influence approximation method that is practical for large-scale generative AI
models. Leveraging an easy-to-compute closed-form expression, DataInf
outperforms existing influence computation algorithms in terms of computational
and memory efficiency. Our theoretical analysis shows that DataInf is
particularly well-suited for parameter-efficient fine-tuning techniques such as
LoRA. Through systematic empirical evaluations, we show that DataInf accurately
approximates influence scores and is orders of magnitude faster than existing
methods. In applications to RoBERTa-large, Llama-2-13B-chat, and
stable-diffusion-v1.5 models, DataInf effectively identifies the most
influential fine-tuning examples better than other approximate influence
scores. Moreover, it can help to identify which data points are mislabeled.


# TADIS: Steering Models for Deep-Thinking about Demonstration Examples

[Link to the paper](http://arxiv.org/abs/2310.00901v1)

## Authors
- Tianci Xue
- Ziqi Wang
- Yixia Li
- Yun Chen
- Guanhua Chen

## Summary
  Instruction tuning has been demonstrated that could significantly improve the
zero-shot generalization capability to unseen tasks by an apparent margin. By
incorporating additional context (e.g., task definition, examples) during the
fine-tuning process, Large Language Models (LLMs) achieved much higher
performance than before. However, recent work reported that delusive task
examples can achieve almost the same performance as correct task examples,
indicating the input-label correspondence is less important than previously
thought. Intrigued by this counter-intuitive observation, we suspect models
have the same illusion of competence as humans. Therefore, we propose a novel
method called TADIS that steers LLMs for "Deep-Thinking'' about demonstration
examples instead of merely seeing. To alleviate the illusion of competence of
models, we first ask the model to verify the correctness of shown examples.
Then, using the verification results as conditions to elicit models for a
better answer. Our experimental results show that TADIS consistently
outperforms competitive baselines on in-domain and out-domain tasks (improving
2.79 and 4.03 average ROUGLE-L on out-domain and in-domain datasets,
respectively). Despite the presence of generated examples (not all of the
thinking labels are accurate), TADIS can notably enhance performance in
zero-shot and few-shot settings. This also suggests that our approach can be
adopted on a large scale to improve the instruction following capabilities of
models without any manual labor. Moreover, we construct three types of thinking
labels with different model sizes and find that small models learn from the
format of TADIS but larger models can be steered for "Deep-Thinking''.


# Enable Language Models to Implicitly Learn Self-Improvement From Data

[Link to the paper](http://arxiv.org/abs/2310.00898v1)

## Authors
- Ziqi Wang
- Le Hou
- Tianjian Lu
- Yuexin Wu
- Yunxuan Li
- Hongkun Yu
- Heng Ji

## Summary
  Large Language Models (LLMs) have demonstrated remarkable capabilities in
open-ended text generation tasks. However, the inherent open-ended nature of
these tasks implies that there is always room for improvement in the quality of
model responses. To address this challenge, various approaches have been
proposed to enhance the performance of LLMs. There has been a growing focus on
enabling LLMs to self-improve their response quality, thereby reducing the
reliance on extensive human annotation efforts for collecting diverse and
high-quality training data. Recently, prompting-based methods have been widely
explored among self-improvement methods owing to their effectiveness,
efficiency, and convenience. However, those methods usually require explicitly
and thoroughly written rubrics as inputs to LLMs. It is expensive and
challenging to manually derive and provide all necessary rubrics with a
real-world complex goal for improvement (e.g., being more helpful and less
harmful). To this end, we propose an ImPlicit Self-ImprovemenT (PIT) framework
that implicitly learns the improvement goal from human preference data. PIT
only requires preference data that are used to train reward models without
extra human efforts. Specifically, we reformulate the training objective of
reinforcement learning from human feedback (RLHF) -- instead of maximizing
response quality for a given input, we maximize the quality gap of the response
conditioned on a reference response. In this way, PIT is implicitly trained
with the improvement goal of better aligning with human preferences.
Experiments on two real-world datasets and one synthetic dataset show that our
method significantly outperforms prompting-based methods.


# RCOT: Detecting and Rectifying Factual Inconsistency in Reasoning by Reversing Chain-of-Thought

[Link to the paper](http://arxiv.org/abs/2305.11499v2)

## Authors
- Tianci Xue
- Ziqi Wang
- Zhenhailong Wang
- Chi Han
- Pengfei Yu
- Heng Ji

## Summary
  Large language Models (LLMs) have achieved promising performance on
arithmetic reasoning tasks by incorporating step-by-step chain-of-thought (CoT)
prompting. However, LLMs face challenges in maintaining factual consistency
during reasoning, exhibiting tendencies to condition overlooking, question
misinterpretation, and condition hallucination over given problems. Existing
methods use coarse-grained feedback (e.g., whether the answer is correct) to
improve factual consistency. In this work, we propose RCoT (Reversing
Chain-of-Thought), a novel method to improve LLMs' reasoning abilities by
automatically detecting and rectifying factual inconsistency in LLMs, generated
solutions. To detect factual inconsistency, RCoT first asks LLMs to reconstruct
the problem based on generated solutions. Then fine-grained comparisons between
the original problem and the reconstructed problem expose the factual
inconsistency in the original solutions. To rectify the solution, RCoT
formulates detected factual inconsistency into fine-grained feedback to guide
LLMs in revising solutions. Experimental results demonstrate improvements of
RCoT over standard CoT, Self-Consistency and Self-Refine across seven
arithmetic datasets. Moreover, we find that manually written fine-grained
feedback can dramatically improve LLMs' reasoning abilities (e.g., ChatGPT
reaches 94.6% accuracy on GSM8K), encouraging the community to further explore
the fine-grained feedback generation methods.


# (Dynamic) Prompting might be all you need to repair Compressed LLMs

[Link to the paper](http://arxiv.org/abs/2310.00867v1)

## Authors
- Duc N. M Hoang
- Minsik Cho
- Thomas Merth
- Mohammad Rastegari
- Zhangyang Wang

## Summary
  Large language models (LLMs), while transformative for NLP, come with
significant computational demands, underlining the need for efficient,
training-free compression. Notably, the reliability of perplexity as a
benchmark for compressed model efficacy is in question, as our tests using
LLaMA-7B and OPT-6.7b reveal a significant performance drop in several
realistic downstream tasks, underscoring the disparity between perplexity as a
performance indicator and real-world performance. Investigation into the
trade-off between resource-intensive post-compression re-training highlights
the prospect of prompt-driven recovery as a lightweight adaption tool. However,
existing studies, confined mainly to perplexity evaluations and simple tasks,
fail to offer unequivocal confidence in the scalability and generalizability of
prompting. We tackle this uncertainty in two key ways. First, we uncover the
vulnerability of naive prompts in LLM compression as an over-reliance on a
singular prompt per input. In response, we propose inference-time dynamic
prompting (IDP), a mechanism that autonomously chooses from a set of curated
prompts based on the context of each individual input. Second, we delve into a
scientific understanding of why ``prompting might be all you need post-LLM
compression". Our findings suggest that compression doesn't irretrievably erase
LLM model knowledge but displace it, necessitating a new inference path. IDP
effectively redirects this path, enabling the model to tap into its inherent
yet displaced knowledge and thereby recover performance. Empirical tests affirm
the value of IDP, demonstrating an average performance improvement of 1.24%
across nine varied tasks spanning multiple knowledge domains.


# Use Your INSTINCT: INSTruction optimization usIng Neural bandits Coupled with Transformers

[Link to the paper](http://arxiv.org/abs/2310.02905v1)

## Authors
- Xiaoqiang Lin
- Zhaoxuan Wu
- Zhongxiang Dai
- Wenyang Hu
- Yao Shu
- See-Kiong Ng
- Patrick Jaillet
- Bryan Kian Hsiang Low

## Summary
  Large language models (LLMs) have shown remarkable instruction-following
capabilities and achieved impressive performances in various applications.
However, the performances of LLMs depend heavily on the instructions given to
them, which are typically manually tuned with substantial human efforts. Recent
work has used the query-efficient Bayesian optimization (BO) algorithm to
automatically optimize the instructions given to black-box LLMs. However, BO
usually falls short when optimizing highly sophisticated (e.g.,
high-dimensional) objective functions, such as the functions mapping an
instruction to the performance of an LLM. This is mainly due to the limited
expressive power of the Gaussian process (GP) model which is used by BO as a
surrogate to model the objective function. Meanwhile, it has been repeatedly
shown that neural networks (NNs), especially pre-trained transformers, possess
strong expressive power and can model highly complex functions. So, we adopt a
neural bandit algorithm which replaces the GP in BO by an NN surrogate to
optimize instructions for black-box LLMs. More importantly, the neural bandit
algorithm allows us to naturally couple the NN surrogate with the hidden
representation learned by a pre-trained transformer (i.e., an open-source LLM),
which significantly boosts its performance. These motivate us to propose our
INSTruction optimization usIng Neural bandits Coupled with Transformers}
(INSTINCT) algorithm. We perform instruction optimization for ChatGPT and use
extensive experiments to show that our INSTINCT consistently outperforms the
existing methods in different tasks, such as in various instruction induction
tasks and the task of improving the zero-shot chain-of-thought instruction.


# LLM-grounded Video Diffusion Models

[Link to the paper](http://arxiv.org/abs/2309.17444v2)

## Authors
- Long Lian
- Baifeng Shi
- Adam Yala
- Trevor Darrell
- Boyi Li

## Summary
  Text-conditioned diffusion models have emerged as a promising tool for neural
video generation. However, current models still struggle with intricate
spatiotemporal prompts and often generate restricted or incorrect motion (e.g.,
even lacking the ability to be prompted for objects moving from left to right).
To address these limitations, we introduce LLM-grounded Video Diffusion (LVD).
Instead of directly generating videos from the text inputs, LVD first leverages
a large language model (LLM) to generate dynamic scene layouts based on the
text inputs and subsequently uses the generated layouts to guide a diffusion
model for video generation. We show that LLMs are able to understand complex
spatiotemporal dynamics from text alone and generate layouts that align closely
with both the prompts and the object motion patterns typically observed in the
real world. We then propose to guide video diffusion models with these layouts
by adjusting the attention maps. Our approach is training-free and can be
integrated into any video diffusion model that admits classifier guidance. Our
results demonstrate that LVD significantly outperforms its base video diffusion
model and several strong baseline methods in faithfully generating videos with
the desired attributes and motion patterns.


# Application of frozen large-scale models to multimodal task-oriented dialogue

[Link to the paper](http://arxiv.org/abs/2310.00845v1)

## Authors
- Tatsuki Kawamoto
- Takuma Suzuki
- Ko Miyama
- Takumi Meguro
- Tomohiro Takagi

## Summary
  In this study, we use the existing Large Language Models ENnhanced to See
Framework (LENS Framework) to test the feasibility of multimodal task-oriented
dialogues. The LENS Framework has been proposed as a method to solve computer
vision tasks without additional training and with fixed parameters of
pre-trained models. We used the Multimodal Dialogs (MMD) dataset, a multimodal
task-oriented dialogue benchmark dataset from the fashion field, and for the
evaluation, we used the ChatGPT-based G-EVAL, which only accepts textual
modalities, with arrangements to handle multimodal data. Compared to
Transformer-based models in previous studies, our method demonstrated an
absolute lift of 10.8% in fluency, 8.8% in usefulness, and 5.2% in relevance
and coherence. The results show that using large-scale models with fixed
parameters rather than using models trained on a dataset from scratch improves
performance in multimodal task-oriented dialogues. At the same time, we show
that Large Language Models (LLMs) are effective for multimodal task-oriented
dialogues. This is expected to lead to efficient applications to existing
systems.


# Towards LogiGLUE: A Brief Survey and A Benchmark for Analyzing Logical Reasoning Capabilities of Language Models

[Link to the paper](http://arxiv.org/abs/2310.00836v1)

## Authors
- Man Luo
- Shrinidhi Kumbhar
- Ming shen
- Mihir Parmar
- Neeraj Varshney
- Pratyay Banerjee
- Somak Aditya
- Chitta Baral

## Summary
  Logical reasoning is fundamental for humans yet presents a substantial
challenge in the domain of Artificial Intelligence. Initially, researchers used
Knowledge Representation and Reasoning (KR) systems that did not scale and
required non trivial manual effort. Recently, the emergence of large language
models (LLMs) has demonstrated the ability to overcome various limitations of
formal Knowledge Representation (KR) systems. Consequently, there is a growing
interest in using LLMs for logical reasoning via natural language. This work
strives to understand the proficiency of LLMs in logical reasoning by offering
a brief review of the latest progress in this area; with a focus on the logical
reasoning datasets, tasks, and the methods adopted to utilize LLMs for
reasoning. To offer a thorough analysis, we have compiled a benchmark titled
LogiGLUE. This includes 24 varied datasets encompassing deductive, abductive,
and inductive reasoning. We have standardized these datasets into Seq2Seq tasks
to facilitate straightforward training and evaluation for future research.
Utilizing LogiGLUE as a foundation, we have trained an instruction fine tuned
language model, resulting in LogiT5. We study single task training, multi task
training, and a chain of thought knowledge distillation fine tuning technique
to assess the performance of model across the different logical reasoning
categories. By this comprehensive process, we aim to shed light on the
capabilities and potential pathways for enhancing logical reasoning proficiency
in LLMs, paving the way for more advanced and nuanced developments in this
critical field.


# Necessary and Sufficient Watermark for Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.00833v1)

## Authors
- Yuki Takezawa
- Ryoma Sato
- Han Bao
- Kenta Niwa
- Makoto Yamada

## Summary
  In recent years, large language models (LLMs) have achieved remarkable
performances in various NLP tasks. They can generate texts that are
indistinguishable from those written by humans. Such remarkable performance of
LLMs increases their risk of being used for malicious purposes, such as
generating fake news articles. Therefore, it is necessary to develop methods
for distinguishing texts written by LLMs from those written by humans.
Watermarking is one of the most powerful methods for achieving this. Although
existing watermarking methods have successfully detected texts generated by
LLMs, they significantly degrade the quality of the generated texts. In this
study, we propose the Necessary and Sufficient Watermark (NS-Watermark) for
inserting watermarks into generated texts without degrading the text quality.
More specifically, we derive minimum constraints required to be imposed on the
generated texts to distinguish whether LLMs or humans write the texts. Then, we
formulate the NS-Watermark as a constrained optimization problem and propose an
efficient algorithm to solve it. Through the experiments, we demonstrate that
the NS-Watermark can generate more natural texts than existing watermarking
methods and distinguish more accurately between texts written by LLMs and those
written by humans. Especially in machine translation tasks, the NS-Watermark
can outperform the existing watermarking method by up to 30 BLEU scores.


# Parameter-Efficient Tuning Helps Language Model Alignment

[Link to the paper](http://arxiv.org/abs/2310.00819v1)

## Authors
- Tianci Xue
- Ziqi Wang
- Heng Ji

## Summary
  Aligning large language models (LLMs) with human preferences is essential for
safe and useful LLMs. Previous works mainly adopt reinforcement learning (RLHF)
and direct preference optimization (DPO) with human feedback for alignment.
Nevertheless, they have certain drawbacks. One such limitation is that they can
only align models with one preference at the training time (e.g., they cannot
learn to generate concise responses when the preference data prefers detailed
responses), or have certain constraints for the data format (e.g., DPO only
supports pairwise preference data). To this end, prior works incorporate
controllable generations for alignment to make language models learn multiple
preferences and provide outputs with different preferences during inference if
asked. Controllable generation also offers more flexibility with regard to data
format (e.g., it supports pointwise preference data). Specifically, it uses
different control tokens for different preferences during training and
inference, making LLMs behave differently when required. Current controllable
generation methods either use a special token or hand-crafted prompts as
control tokens, and optimize them together with LLMs. As control tokens are
typically much lighter than LLMs, this optimization strategy may not
effectively optimize control tokens. To this end, we first use
parameter-efficient tuning (e.g., prompting tuning and low-rank adaptation) to
optimize control tokens and then fine-tune models for controllable generations,
similar to prior works. Our approach, alignMEnt with parameter-Efficient Tuning
(MEET), improves the quality of control tokens, thus improving controllable
generation quality consistently by an apparent margin on two well-recognized
datasets compared with prior works.


# ReAcTable: Enhancing ReAct for Table Question Answering

[Link to the paper](http://arxiv.org/abs/2310.00815v1)

## Authors
- Yunjia Zhang
- Jordan Henkel
- Avrilia Floratou
- Joyce Cahoon
- Shaleen Deep
- Jignesh M. Patel

## Summary
  Table Question Answering (TQA) presents a substantial challenge at the
intersection of natural language processing and data analytics. This task
involves answering natural language (NL) questions on top of tabular data,
demanding proficiency in logical reasoning, understanding of data semantics,
and fundamental analytical capabilities. Due to its significance, a substantial
volume of research has been dedicated to exploring a wide range of strategies
aimed at tackling this challenge including approaches that leverage Large
Language Models (LLMs) through in-context learning or Chain-of-Thought (CoT)
prompting as well as approaches that train and fine-tune custom models.
  Nonetheless, a conspicuous gap exists in the research landscape, where there
is limited exploration of how innovative foundational research, which
integrates incremental reasoning with external tools in the context of LLMs, as
exemplified by the ReAct paradigm, could potentially bring advantages to the
TQA task. In this paper, we aim to fill this gap, by introducing ReAcTable
(ReAct for Table Question Answering tasks), a framework inspired by the ReAct
paradigm that is carefully enhanced to address the challenges uniquely
appearing in TQA tasks such as interpreting complex data semantics, dealing
with errors generated by inconsistent data and generating intricate data
transformations. ReAcTable relies on external tools such as SQL and Python code
executors, to progressively enhance the data by generating intermediate data
representations, ultimately transforming it into a more accessible format for
answering the questions with greater ease. We demonstrate that ReAcTable
achieves remarkable performance even when compared to fine-tuned approaches. In
particular, it outperforms the best prior result on the WikiTQ benchmark,
achieving an accuracy of 68.0% without requiring training a new model or
fine-tuning.


# Testing the Limits of Unified Sequence to Sequence LLM Pretraining on Diverse Table Data Tasks

[Link to the paper](http://arxiv.org/abs/2310.00789v1)

## Authors
- Soumajyoti Sarkar
- Leonard Lausen

## Summary
  Tables stored in databases and tables which are present in web pages and
articles account for a large part of semi-structured data that is available on
the internet. It then becomes pertinent to develop a modeling approach with
large language models (LLMs) that can be used to solve diverse table tasks such
as semantic parsing, question answering as well as classification problems.
Traditionally, there existed separate models specialized for each task
individually. It raises the question of how far can we go to build a unified
model that works well on some table tasks without significant degradation on
others. To that end, we attempt at creating a shared modeling approach in the
pretraining stage with encoder-decoder style LLMs that can cater to diverse
tasks. We evaluate our approach that continually pretrains and finetunes
different model families of T5 with data from tables and surrounding context,
on these downstream tasks at different model scales. Through multiple ablation
studies, we observe that our pretraining with self-supervised objectives can
significantly boost the performance of the models on these tasks. As an example
of one improvement, we observe that the instruction finetuned public models
which come specialized on text question answering (QA) and have been trained on
table data still have room for improvement when it comes to table specific QA.
Our work is the first attempt at studying the advantages of a unified approach
to table specific pretraining when scaled from 770M to 11B sequence to sequence
models while also comparing the instruction finetuned variants of the models.


# BooookScore: A systematic exploration of book-length summarization in the era of LLMs

[Link to the paper](http://arxiv.org/abs/2310.00785v1)

## Authors
- Yapei Chang
- Kyle Lo
- Tanya Goyal
- Mohit Iyyer

## Summary
  Summarizing book-length documents (>100K tokens) that exceed the context
window size of large language models (LLMs) requires first breaking the input
document into smaller chunks and then prompting an LLM to merge, update, and
compress chunk-level summaries. Despite the complexity and importance of this
task, it has yet to be meaningfully studied due to the challenges of
evaluation: existing book-length summarization datasets (e.g., BookSum) are in
the pretraining data of most public LLMs, and existing evaluation methods
struggle to capture errors made by modern LLM summarizers. In this paper, we
present the first study of the coherence of LLM-based book-length summarizers
implemented via two prompting workflows: (1) hierarchically merging chunk-level
summaries, and (2) incrementally updating a running summary. We obtain 1193
fine-grained human annotations on GPT-4 generated summaries of 100
recently-published books and identify eight common types of coherence errors
made by LLMs. Because human evaluation is expensive and time-consuming, we
develop an automatic metric, BooookScore, that measures the proportion of
sentences in a summary that do not contain any of the identified error types.
BooookScore has high agreement with human annotations and allows us to
systematically evaluate the impact of many other critical parameters (e.g.,
chunk size, base LLM) while saving $15K and 500 hours in human evaluation
costs. We find that closed-source LLMs such as GPT-4 and Claude 2 produce
summaries with higher BooookScore than the oft-repetitive ones generated by
LLaMA 2. Incremental updating yields lower BooookScore but higher level of
detail than hierarchical merging, a trade-off sometimes preferred by human
annotators. We release code and annotations after blind review to spur more
principled research on book-length summarization.


# OpenBA: An Open-sourced 15B Bilingual Asymmetric seq2seq Model Pre-trained from Scratch

[Link to the paper](http://arxiv.org/abs/2309.10706v2)

## Authors
- Juntao Li
- Zecheng Tang
- Yuyang Ding
- Pinzheng Wang
- Pei Guo
- Wangjie You
- Dan Qiao
- Wenliang Chen
- Guohong Fu
- Qiaoming Zhu
- Guodong Zhou
- Min Zhang

## Summary
  Large language models (LLMs) with billions of parameters have demonstrated
outstanding performance on various natural language processing tasks. This
report presents OpenBA, an open-sourced 15B bilingual asymmetric seq2seq model,
to contribute an LLM variant to the Chinese-oriented open-source model
community. We enhance OpenBA with effective and efficient techniques as well as
adopt a three-stage training strategy to train the model from scratch. Our
solution can also achieve very competitive performance with only 380B tokens,
which is better than LLaMA-70B on the BELEBELE benchmark, BLOOM-176B on the
MMLU benchmark, GLM-130B on the C-Eval (hard) benchmark. This report provides
the main details to pre-train an analogous model, including pre-training data
processing, Bilingual Flan data collection, the empirical observations that
inspire our model architecture design, training objectives of different stages,
and other enhancement techniques. Additionally, we also provide the fine-tuning
details of OpenBA on four downstream tasks. We have refactored our code to
follow the design principles of the Huggingface Transformers Library, making it
more convenient for developers to use, and released checkpoints of different
training stages at https://huggingface.co/openBA. More details of our project
are available at https://github.com/OpenNLG/openBA.git.


# SEED: Simple, Efficient, and Effective Data Management via Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.00749v1)

## Authors
- Zui CHen
- Lei Cao
- Sam Madden
- Ju Fan
- Nan Tang
- Zihui Gu
- Zeyuan Shang
- Chunwei Liu
- Michael Cafarella
- Tim Kraska

## Summary
  We introduce SEED, an LLM-centric system that allows users to easily create
efficient, and effective data management applications. SEED comprises three
main components: code generation, model generation, and augmented LLM query to
address the challenges that LLM services are computationally and economically
expensive and do not always work well on all cases for a given data management
task. SEED addresses the expense challenge by localizing LLM computation as
much as possible. This includes replacing most of LLM calls with local code,
local models, and augmenting LLM queries with batching and data access tools,
etc. To ensure effectiveness, SEED features a bunch of optimization techniques
to enhance the localized solution and the LLM queries, including automatic code
validation, code ensemble, model representatives selection, selective tool
usages, etc. Moreover, with SEED users are able to easily construct a data
management solution customized to their applications. It allows the users to
configure each component and compose an execution pipeline in natural language.
SEED then automatically compiles it into an executable program. We showcase the
efficiency and effectiveness of SEED using diverse data management tasks such
as data imputation, NL2SQL translation, etc., achieving state-of-the-art
few-shot performance while significantly reducing the number of required LLM
calls.


# RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.00746v1)

## Authors
- Zekun Moore Wang
- Zhongyuan Peng
- Haoran Que
- Jiaheng Liu
- Wangchunshu Zhou
- Yuhan Wu
- Hongcheng Guo
- Ruitong Gan
- Zehao Ni
- Man Zhang
- Zhaoxiang Zhang
- Wanli Ouyang
- Ke Xu
- Wenhu Chen
- Jie Fu
- Junran Peng

## Summary
  The advent of Large Language Models (LLMs) has paved the way for complex
tasks such as role-playing, which enhances user interactions by enabling models
to imitate various characters. However, the closed-source nature of
state-of-the-art LLMs and their general-purpose training limit role-playing
optimization. In this paper, we introduce RoleLLM, a framework to benchmark,
elicit, and enhance role-playing abilities in LLMs. RoleLLM comprises four
stages: (1) Role Profile Construction for 100 roles; (2) Context-Based
Instruction Generation (Context-Instruct) for role-specific knowledge
extraction; (3) Role Prompting using GPT (RoleGPT) for speaking style
imitation; and (4) Role-Conditioned Instruction Tuning (RoCIT) for fine-tuning
open-source models along with role customization. By Context-Instruct and
RoleGPT, we create RoleBench, the first systematic and fine-grained
character-level benchmark dataset for role-playing with 168,093 samples.
Moreover, RoCIT on RoleBench yields RoleLLaMA (English) and RoleGLM (Chinese),
significantly enhancing role-playing abilities and even achieving comparable
results with RoleGPT (using GPT-4).


# FELM: Benchmarking Factuality Evaluation of Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.00741v1)

## Authors
- Shiqi Chen
- Yiran Zhao
- Jinghan Zhang
- I-Chun Chern
- Siyang Gao
- Pengfei Liu
- Junxian He

## Summary
  Assessing factuality of text generated by large language models (LLMs) is an
emerging yet crucial research area, aimed at alerting users to potential errors
and guiding the development of more reliable LLMs. Nonetheless, the evaluators
assessing factuality necessitate suitable evaluation themselves to gauge
progress and foster advancements. This direction remains under-explored,
resulting in substantial impediments to the progress of factuality evaluators.
To mitigate this issue, we introduce a benchmark for Factuality Evaluation of
large Language Models, referred to as felm. In this benchmark, we collect
responses generated from LLMs and annotate factuality labels in a fine-grained
manner. Contrary to previous studies that primarily concentrate on the
factuality of world knowledge (e.g.~information from Wikipedia), felm focuses
on factuality across diverse domains, spanning from world knowledge to math and
reasoning. Our annotation is based on text segments, which can help pinpoint
specific factual errors. The factuality annotations are further supplemented by
predefined error types and reference links that either support or contradict
the statement. In our experiments, we investigate the performance of several
LLM-based factuality evaluators on felm, including both vanilla LLMs and those
augmented with retrieval mechanisms and chain-of-thought processes. Our
findings reveal that while retrieval aids factuality evaluation, current LLMs
are far from satisfactory to faithfully detect factual errors.


# GenAI Against Humanity: Nefarious Applications of Generative Artificial Intelligence and Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.00737v1)

## Authors
- Emilio Ferrara

## Summary
  Generative Artificial Intelligence (GenAI) and Large Language Models (LLMs)
are marvels of technology; celebrated for their prowess in natural language
processing and multimodal content generation, they promise a transformative
future. But as with all powerful tools, they come with their shadows. Picture
living in a world where deepfakes are indistinguishable from reality, where
synthetic identities orchestrate malicious campaigns, and where targeted
misinformation or scams are crafted with unparalleled precision. Welcome to the
darker side of GenAI applications.
  This article is not just a journey through the meanders of potential misuse
of GenAI and LLMs, but also a call to recognize the urgency of the challenges
ahead. As we navigate the seas of misinformation campaigns, malicious content
generation, and the eerie creation of sophisticated malware, we'll uncover the
societal implications that ripple through the GenAI revolution we are
witnessing. From AI-powered botnets on social media platforms to the unnerving
potential of AI to generate fabricated identities, or alibis made of synthetic
realities, the stakes have never been higher.
  The lines between the virtual and the real worlds are blurring, and the
consequences of potential GenAI's nefarious applications impact us all. This
article serves both as a synthesis of rigorous research presented on the risks
of GenAI and misuse of LLMs and as a thought-provoking vision of the different
types of harmful GenAI applications we might encounter in the near future, and
some ways we can prepare for them.


# Meta Semantic Template for Evaluation of Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.01448v1)

## Authors
- Yachuan Liu
- Liang Chen
- Jindong Wang
- Qiaozhu Mei
- Xing Xie

## Summary
  Do large language models (LLMs) genuinely understand the semantics of the
language, or just memorize the training data? The recent concern on potential
data contamination of LLMs has raised awareness of the community to conduct
research on LLMs evaluation. In this paper, we propose MSTemp, an approach that
creates meta semantic templates to evaluate the semantic understanding ability
of LLMs. The core of MSTemp is not to perform evaluation directly on existing
benchmark datasets, but to generate new out-of-distribution (OOD) evaluation
sets using existing datasets as seeds. Specifically, for a given sentence,
MSTemp leverages another language model to generate new samples while
preserving its semantics. The new samples are called semantic templates to the
original sentence. Then, MSTemp generates evaluation samples via sentence
parsing and random word replacement on the semantic templates. MSTemp is highly
flexible, dynamic, and cost-effective. Our initial experiments show that
MSTemp-generated samples can significantly reduce the performance of LLMs using
existing datasets as seeds. We hope this initial work can shed light on future
research of LLMs evaluation.


# The Robots are Here: Navigating the Generative AI Revolution in Computing Education

[Link to the paper](http://arxiv.org/abs/2310.00658v1)

## Authors
- James Prather
- Paul Denny
- Juho Leinonen
- Brett A. Becker
- Ibrahim Albluwi
- Michelle Craig
- Hieke Keuning
- Natalie Kiesler
- Tobias Kohn
- Andrew Luxton-Reilly
- Stephen MacNeil
- Andrew Peterson
- Raymond Pettit
- Brent N. Reeves
- Jaromir Savelka

## Summary
  Recent advancements in artificial intelligence (AI) are fundamentally
reshaping computing, with large language models (LLMs) now effectively being
able to generate and interpret source code and natural language instructions.
These emergent capabilities have sparked urgent questions in the computing
education community around how educators should adapt their pedagogy to address
the challenges and to leverage the opportunities presented by this new
technology. In this working group report, we undertake a comprehensive
exploration of LLMs in the context of computing education and make five
significant contributions. First, we provide a detailed review of the
literature on LLMs in computing education and synthesise findings from 71
primary articles. Second, we report the findings of a survey of computing
students and instructors from across 20 countries, capturing prevailing
attitudes towards LLMs and their use in computing education contexts. Third, to
understand how pedagogy is already changing, we offer insights collected from
in-depth interviews with 22 computing educators from five continents who have
already adapted their curricula and assessments. Fourth, we use the ACM Code of
Ethics to frame a discussion of ethical issues raised by the use of large
language models in computing education, and we provide concrete advice for
policy makers, educators, and students. Finally, we benchmark the performance
of LLMs on various computing education datasets, and highlight the extent to
which the capabilities of current models are rapidly improving. Our aim is that
this report will serve as a focal point for both researchers and practitioners
who are exploring, adapting, using, and evaluating LLMs and LLM-based tools in
computing classrooms.


# LEGO-Prover: Neural Theorem Proving with Growing Libraries

[Link to the paper](http://arxiv.org/abs/2310.00656v1)

## Authors
- Huajian Xin
- Haiming Wang
- Chuanyang Zheng
- Lin Li
- Zhengying Liu
- Qingxing Cao
- Yinya Huang
- Jing Xiong
- Han Shi
- Enze Xie
- Jian Yin
- Zhenguo Li
- Xiaodan Liang

## Summary
  Despite the success of large language models (LLMs), the task of theorem
proving still remains one of the hardest reasoning tasks that is far from being
fully solved. Prior methods using language models have demonstrated promising
results, but they still struggle to prove even middle school level theorems.
One common limitation of these methods is that they assume a fixed theorem
library during the whole theorem proving process. However, as we all know,
creating new useful theorems or even new theories is not only helpful but
crucial and necessary for advancing mathematics and proving harder and deeper
results. In this work, we present LEGO-Prover, which employs a growing skill
library containing verified lemmas as skills to augment the capability of LLMs
used in theorem proving. By constructing the proof modularly, LEGO-Prover
enables LLMs to utilize existing skills retrieved from the library and to
create new skills during the proving process. These skills are further evolved
(by prompting an LLM) to enrich the library on another scale. Modular and
reusable skills are constantly added to the library to enable tackling
increasingly intricate mathematical problems. Moreover, the learned library
further bridges the gap between human proofs and formal proofs by making it
easier to impute missing steps. LEGO-Prover advances the state-of-the-art pass
rate on miniF2F-valid (48.0% to 57.0%) and miniF2F-test (45.5% to 47.1%).
During the proving process, LEGO-Prover also manages to generate over 20,000
skills (theorems/lemmas) and adds them to the growing library. Our ablation
study indicates that these newly added skills are indeed helpful for proving
theorems, resulting in an improvement from a success rate of 47.1% to 50.4%. We
also release our code and all the generated skills.


# Adaptive-Solver Framework for Dynamic Strategy Selection in Large Language Model Reasoning

[Link to the paper](http://arxiv.org/abs/2310.01446v1)

## Authors
- Jianpeng Zhou
- Wanjun Zhong
- Yanlin Wang
- Jiahai Wang

## Summary
  Large Language Models (LLMs) are showcasing impressive ability in handling
complex reasoning tasks. In real-world situations, problems often span a
spectrum of complexities. Humans inherently adjust their problem-solving
approaches based on task complexity. However, most methodologies that leverage
LLMs tend to adopt a uniform approach: utilizing consistent models, prompting
methods, and degrees of problem decomposition, regardless of the problem
complexity. Inflexibility of them can bring unnecessary computational overhead
or sub-optimal performance. To address this problem, we introduce an
Adaptive-Solver framework. It strategically modulates solving strategies based
on the difficulties of the problems. Given an initial solution, the framework
functions with two primary modules. The initial evaluation module assesses the
adequacy of the current solution. If improvements are needed, the subsequent
adaptation module comes into play. Within this module, three key adaptation
strategies are employed: (1) Model Adaptation: Switching to a stronger LLM when
a weaker variant is inadequate. (2) Prompting Method Adaptation: Alternating
between different prompting techniques to suit the problem's nuances. (3)
Decomposition Granularity Adaptation: Breaking down a complex problem into more
fine-grained sub-questions to enhance solvability. Through such dynamic
adaptations, our framework not only enhances computational efficiency but also
elevates the overall performance. This dual-benefit ensures both the efficiency
of the system for simpler tasks and the precision required for more complex
questions. Experimental results from complex reasoning tasks reveal that the
prompting method adaptation and decomposition granularity adaptation enhance
performance across all tasks. Furthermore, the model adaptation approach
significantly reduces API costs (up to 50%) while maintaining superior
performance.


# Beyond Task Performance: Evaluating and Reducing the Flaws of Large Multimodal Models with In-Context Learning

[Link to the paper](http://arxiv.org/abs/2310.00647v1)

## Authors
- Mustafa Shukor
- Alexandre Rame
- Corentin Dancette
- Matthieu Cord

## Summary
  Following the success of Large Language Models (LLMs), Large Multimodal
Models (LMMs), such as the Flamingo model and its subsequent competitors, have
started to emerge as natural steps towards generalist agents. However,
interacting with recent LMMs reveals major limitations that are hardly captured
by the current evaluation benchmarks. Indeed, task performances (e.g., VQA
accuracy) alone do not provide enough clues to understand their real
capabilities, limitations, and to which extent such models are aligned to human
expectations. To refine our understanding of those flaws, we deviate from the
current evaluation paradigm and propose the EvALign-ICL framework, in which we
(1) evaluate 8 recent open-source LMMs (based on the Flamingo architecture such
as OpenFlamingo and IDEFICS) on 5 different axes; hallucinations, abstention,
compositionality, explainability and instruction following. Our evaluation on
these axes reveals major flaws in LMMs. To efficiently address these problems,
and inspired by the success of in-context learning (ICL) in LLMs, (2) we
explore ICL as a solution and study how it affects these limitations. Based on
our ICL study, (3) we push ICL further and propose new multimodal ICL
approaches such as; Multitask-ICL, Chain-of-Hindsight-ICL, and
Self-Correcting-ICL. Our findings are as follows; (1) Despite their success,
LMMs have flaws that remain unsolved with scaling alone. (2) The effect of ICL
on LMMs flaws is nuanced; despite its effectiveness for improved
explainability, abstention, and instruction following, ICL does not improve
compositional abilities, and actually even amplifies hallucinations. (3) The
proposed ICL variants are promising as post-hoc approaches to efficiently
tackle some of those flaws. The code is available here:
https://evalign-icl.github.io/


# WASA: WAtermark-based Source Attribution for Large Language Model-Generated Data

[Link to the paper](http://arxiv.org/abs/2310.00646v1)

## Authors
- Jingtan Wang
- Xinyang Lu
- Zitong Zhao
- Zhongxiang Dai
- Chuan-Sheng Foo
- See-Kiong Ng
- Bryan Kian Hsiang Low

## Summary
  The impressive performances of large language models (LLMs) and their immense
potential for commercialization have given rise to serious concerns over the
intellectual property (IP) of their training data. In particular, the synthetic
texts generated by LLMs may infringe the IP of the data being used to train the
LLMs. To this end, it is imperative to be able to (a) identify the data
provider who contributed to the generation of a synthetic text by an LLM
(source attribution) and (b) verify whether the text data from a data provider
has been used to train an LLM (data provenance). In this paper, we show that
both problems can be solved by watermarking, i.e., by enabling an LLM to
generate synthetic texts with embedded watermarks that contain information
about their source(s). We identify the key properties of such watermarking
frameworks (e.g., source attribution accuracy, robustness against adversaries),
and propose a WAtermarking for Source Attribution (WASA) framework that
satisfies these key properties due to our algorithmic designs. Our WASA
framework enables an LLM to learn an accurate mapping from the texts of
different data providers to their corresponding unique watermarks, which sets
the foundation for effective source attribution (and hence data provenance).
Extensive empirical evaluations show that our WASA framework achieves effective
source attribution and data provenance.


# Knowledge Engineering using Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.00637v1)

## Authors
- Bradley P. Allen
- Lise Stork
- Paul Groth

## Summary
  Knowledge engineering is a discipline that focuses on the creation and
maintenance of processes that generate and apply knowledge. Traditionally,
knowledge engineering approaches have focused on knowledge expressed in formal
languages. The emergence of large language models and their capabilities to
effectively work with natural language, in its broadest sense, raises questions
about the foundations and practice of knowledge engineering. Here, we outline
the potential role of LLMs in knowledge engineering, identifying two central
directions: 1) creating hybrid neuro-symbolic knowledge systems; and 2)
enabling knowledge engineering in natural language. Additionally, we formulate
key open research questions to tackle these directions.


# Time Travel in LLMs: Tracing Data Contamination in Large Language Models

[Link to the paper](http://arxiv.org/abs/2308.08493v2)

## Authors
- Shahriar Golchin
- Mihai Surdeanu

## Summary
  Data contamination, i.e., the presence of test data from downstream tasks in
the training data of large language models (LLMs), is a potential major issue
in measuring LLMs' real effectiveness on other tasks. We propose a
straightforward yet effective method for identifying data contamination within
LLMs. At its core, our approach starts by identifying potential contamination
at the instance level; using this information, our approach then assesses wider
contamination at the partition level. To estimate contamination of individual
instances, we employ "guided instruction:" a prompt consisting of the dataset
name, partition type, and the random-length initial segment of a reference
instance, asking the LLM to complete it. An instance is flagged as contaminated
if the LLM's output either exactly or nearly matches the latter segment of the
reference. To understand if an entire partition is contaminated, we propose two
ideas. The first idea marks a dataset partition as contaminated if the average
overlap score with the reference instances (as measured by ROUGE-L or BLEURT)
is statistically significantly better with the completions from guided
instruction compared to a "general instruction" that does not include the
dataset and partition name. The second idea marks a dataset partition as
contaminated if a classifier based on GPT-4 with few-shot in-context learning
prompt marks multiple generated completions as exact/near-exact matches of the
corresponding reference instances. Our best method achieves an accuracy between
92% and 100% in detecting if an LLM is contaminated with seven datasets,
containing train and test/validation partitions, when contrasted with manual
evaluation by human experts. Further, our findings indicate that GPT-4 is
contaminated with AG News, WNLI, and XSum datasets.


# Adapting LLM Agents Through Communication

[Link to the paper](http://arxiv.org/abs/2310.01444v1)

## Authors
- Kuan Wang
- Yadong Lu
- Michael Santacroce
- Yeyun Gong
- Chao Zhang
- Yelong Shen

## Summary
  Recent advancements in large language models (LLMs) have shown potential for
human-like agents. To help these agents adapt to new tasks without extensive
human supervision, we propose the Learning through Communication (LTC)
paradigm, a novel training approach enabling LLM agents to improve continuously
through interactions with their environments and other agents. Recent
advancements in large language models (LLMs) have shown potential for
human-like agents. To help these agents adapt to new tasks without extensive
human supervision, we propose the Learning through Communication (LTC)
paradigm, a novel training approach enabling LLM agents to improve continuously
through interactions with their environments and other agents. Through
iterative exploration and PPO training, LTC empowers the agent to assimilate
short-term experiences into long-term memory. To optimize agent interactions
for task-specific learning, we introduce three structured communication
patterns: Monologue, Dialogue, and Analogue-tailored for common tasks such as
decision-making, knowledge-intensive reasoning, and numerical reasoning. We
evaluated LTC on three datasets: ALFWorld (decision-making), HotpotQA
(knowledge-intensive reasoning), and GSM8k (numerical reasoning). On ALFWorld,
it exceeds the instruction tuning baseline by 12% in success rate. On HotpotQA,
LTC surpasses the instruction-tuned LLaMA-7B agent by 5.1% in EM score, and it
outperforms the instruction-tuned 9x larger PaLM-62B agent by 0.6%. On GSM8k,
LTC outperforms the CoT-Tuning baseline by 3.6% in accuracy. The results
showcase the versatility and efficiency of the LTC approach across diverse
domains. We will open-source our code to promote further development of the
community.


# Faithful Explanations of Black-box NLP Models Using LLM-generated Counterfactuals

[Link to the paper](http://arxiv.org/abs/2310.00603v1)

## Authors
- Yair Gat
- Nitay Calderon
- Amir Feder
- Alexander Chapanin
- Amit Sharma
- Roi Reichart

## Summary
  Causal explanations of the predictions of NLP systems are essential to ensure
safety and establish trust. Yet, existing methods often fall short of
explaining model predictions effectively or efficiently and are often
model-specific. In this paper, we address model-agnostic explanations,
proposing two approaches for counterfactual (CF) approximation. The first
approach is CF generation, where a large language model (LLM) is prompted to
change a specific text concept while keeping confounding concepts unchanged.
While this approach is demonstrated to be very effective, applying LLM at
inference-time is costly. We hence present a second approach based on matching,
and propose a method that is guided by an LLM at training-time and learns a
dedicated embedding space. This space is faithful to a given causal graph and
effectively serves to identify matches that approximate CFs. After showing
theoretically that approximating CFs is required in order to construct faithful
explanations, we benchmark our approaches and explain several models, including
LLMs with billions of parameters. Our empirical results demonstrate the
excellent performance of CF generation models as model-agnostic explainers.
Moreover, our matching approach, which requires far less test-time resources,
also provides effective explanations, surpassing many baselines. We also find
that Top-K techniques universally improve every tested method. Finally, we
showcase the potential of LLMs in constructing new benchmarks for model
explanation and subsequently validate our conclusions. Our work illuminates new
pathways for efficient and accurate approaches to interpreting NLP systems.


# Pink: Unveiling the Power of Referential Comprehension for Multi-modal LLMs

[Link to the paper](http://arxiv.org/abs/2310.00582v1)

## Authors
- Shiyu Xuan
- Qingpei Guo
- Ming Yang
- Shiliang Zhang

## Summary
  Multi-modal Large Language Models (MLLMs) have shown remarkable capabilities
in many vision-language tasks. Nevertheless, most MLLMs still lack the
Referential Comprehension (RC) ability to identify a specific object or area in
images, limiting their application in fine-grained perception tasks. This paper
proposes a novel method to enhance the RC capability for MLLMs. Our model
represents the referring object in the image using the coordinates of its
bounding box and converts the coordinates into texts in a specific format. This
allows the model to treat the coordinates as natural language. Moreover, we
construct the instruction tuning dataset with various designed RC tasks at a
low cost by unleashing the potential of annotations in existing datasets. To
further boost the RC ability of the model, we propose a self-consistent
bootstrapping method that extends dense object annotations of a dataset into
high-quality referring-expression-bounding-box pairs. The model is trained
end-to-end with a parameter-efficient tuning framework that allows both
modalities to benefit from multi-modal instruction tuning. This framework
requires fewer trainable parameters and less training data. Experimental
results on conventional vision-language and RC tasks demonstrate the superior
performance of our method. For instance, our model exhibits a 12.0% absolute
accuracy improvement over Instruct-BLIP on VSR and surpasses Kosmos-2 by 24.7%
on RefCOCO_val under zero-shot settings. We also attain the top position on the
leaderboard of MMBench. The models, datasets, and codes are publicly available
at https://github.com/SY-Xuan/Pink


# GrowLength: Accelerating LLMs Pretraining by Progressively Growing Training Length

[Link to the paper](http://arxiv.org/abs/2310.00576v1)

## Authors
- Hongye Jin
- Xiaotian Han
- Jingfeng Yang
- Zhimeng Jiang
- Chia-Yuan Chang
- Xia Hu

## Summary
  The evolving sophistication and intricacies of Large Language Models (LLMs)
yield unprecedented advancements, yet they simultaneously demand considerable
computational resources and incur significant costs. To alleviate these
challenges, this paper introduces a novel, simple, and effective method named
``\growlength'' to accelerate the pretraining process of LLMs. Our method
progressively increases the training length throughout the pretraining phase,
thereby mitigating computational costs and enhancing efficiency. For instance,
it begins with a sequence length of 128 and progressively extends to 4096. This
approach enables models to process a larger number of tokens within limited
time frames, potentially boosting their performance. In other words, the
efficiency gain is derived from training with shorter sequences optimizing the
utilization of resources. Our extensive experiments with various
state-of-the-art LLMs have revealed that models trained using our method not
only converge more swiftly but also exhibit superior performance metrics
compared to those trained with existing methods. Furthermore, our method for
LLMs pretraining acceleration does not require any additional engineering
efforts, making it a practical solution in the realm of LLMs.


# Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models

[Link to the paper](http://arxiv.org/abs/2310.00566v1)

## Authors
- Duanyu Feng
- Yongfu Dai
- Jimin Huang
- Yifang Zhang
- Qianqian Xie
- Weiguang Han
- Alejandro Lopez-Lira
- Hao Wang

## Summary
  Credit and risk assessments are cornerstones of the financial landscape,
impacting both individual futures and broader societal constructs. Existing
credit scoring models often exhibit limitations stemming from knowledge myopia
and task isolation. In response, we formulate three hypotheses and undertake an
extensive case study to investigate LLMs' viability in credit assessment. Our
empirical investigations unveil LLMs' ability to overcome the limitations
inherent in conventional models. We introduce a novel benchmark curated for
credit assessment purposes, fine-tune a specialized Credit and Risk Assessment
Large Language Model (CALM), and rigorously examine the biases that LLMs may
harbor. Our findings underscore LLMs' potential in revolutionizing credit
assessment, showcasing their adaptability across diverse financial evaluations,
and emphasizing the critical importance of impartial decision-making in the
financial sector. Our datasets, models, and benchmarks are open-sourced for
other researchers.


# Ground Manipulator Primitive Tasks to Executable Actions using Large Language Models

[Link to the paper](http://arxiv.org/abs/2308.06810v2)

## Authors
- Yue Cao
- C. S. George Lee

## Summary
  Layered architectures have been widely used in robot systems. The majority of
them implement planning and execution functions in separate layers. However,
there still lacks a straightforward way to transit high-level tasks in the
planning layer to the low-level motor commands in the execution layer. In order
to tackle this challenge, we propose a novel approach to ground the manipulator
primitive tasks to robot low-level actions using large language models (LLMs).
We designed a program-function-like prompt based on the task frame formalism.
In this way, we enable LLMs to generate position/force set-points for hybrid
control. Evaluations over several state-of-the-art LLMs are provided.


# Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging

[Link to the paper](http://arxiv.org/abs/2309.01026v2)

## Authors
- Rachel M. Harrison
- Anton Dereventsov
- Anton Bibin

## Summary
  We present a method for zero-shot recommendation of multimodal non-stationary
content that leverages recent advancements in the field of generative AI. We
propose rendering inputs of different modalities as textual descriptions and to
utilize pre-trained LLMs to obtain their numerical representations by computing
semantic embeddings. Once unified representations of all content items are
obtained, the recommendation can be performed by computing an appropriate
similarity metric between them without any additional learning. We demonstrate
our approach on a synthetic multimodal nudging environment, where the inputs
consist of tabular, textual, and visual data.


# Are Human-generated Demonstrations Necessary for In-context Learning?

[Link to the paper](http://arxiv.org/abs/2309.14681v2)

## Authors
- Rui Li
- Guoyin Wang
- Jiwei Li

## Summary
  Despite the promising few-shot ability of large language models (LLMs), the
standard paradigm of In-context Learning (ICL) suffers the disadvantages of
susceptibility to selected demonstrations and the intricacy to generate these
demonstrations. In this paper, we raise the fundamental question that whether
human-generated demonstrations are necessary for ICL. To answer this question,
we propose self-contemplation prompting strategy (SEC), a paradigm free from
human-crafted demonstrations. The key point of SEC is that, instead of using
hand-crafted examples as demonstrations in ICL, SEC asks LLMs to first create
demonstrations on their own, based on which the final output is generated. SEC
is a flexible framework and can be adapted to both the vanilla ICL and the
chain-of-thought (CoT), but with greater ease: as the manual-generation process
of both examples and rationale can be saved. Extensive experiments in
arithmetic reasoning, commonsense reasoning, multi-task language understanding,
and code generation benchmarks, show that SEC, which does not require
hand-crafted demonstrations, significantly outperforms the zero-shot learning
strategy, and achieves comparable results to ICL with hand-crafted
demonstrations. This demonstrates that, for many tasks, contemporary LLMs
possess a sufficient level of competence to exclusively depend on their own
capacity for decision making, removing the need for external training data.
Code is available at https://github.com/ruili33/SEC.


# SELF: Language-Driven Self-Evolution for Large Language Model

[Link to the paper](http://arxiv.org/abs/2310.00533v1)

## Authors
- Jianqiao Lu
- Wanjun Zhong
- Wenyong Huang
- Yufei Wang
- Fei Mi
- Baojun Wang
- Weichao Wang
- Lifeng Shang
- Qun Liu

## Summary
  Large Language Models (LLMs) have showcased remarkable versatility across
diverse domains. However, the pathway toward autonomous model development, a
cornerstone for achieving human-level learning and advancing autonomous AI,
remains largely uncharted. We introduce an innovative approach, termed "SELF"
(Self-Evolution with Language Feedback). This methodology empowers LLMs to
undergo continual self-evolution. Furthermore, SELF employs language-based
feedback as a versatile and comprehensive evaluative tool, pinpointing areas
for response refinement and bolstering the stability of self-evolutionary
training. Initiating with meta-skill learning, SELF acquires foundational
meta-skills with a focus on self-feedback and self-refinement. These
meta-skills are critical, guiding the model's subsequent self-evolution through
a cycle of perpetual training with self-curated data, thereby enhancing its
intrinsic abilities. Given unlabeled instructions, SELF equips the model with
the capability to autonomously generate and interactively refine responses.
This synthesized training data is subsequently filtered and utilized for
iterative fine-tuning, enhancing the model's capabilities. Experimental results
on representative benchmarks substantiate that SELF can progressively advance
its inherent abilities without the requirement of human intervention, thereby
indicating a viable pathway for autonomous model evolution. Additionally, SELF
can employ online self-refinement strategy to produce responses of superior
quality. In essence, the SELF framework signifies a progressive step towards
autonomous LLM development, transforming the LLM from a mere passive recipient
of information into an active participant in its own evolution.


# Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition

[Link to the paper](http://arxiv.org/abs/2307.14535v2)

## Authors
- Huy Ha
- Pete Florence
- Shuran Song

## Summary
  We present a framework for robot skill acquisition, which 1) efficiently
scale up data generation of language-labelled robot data and 2) effectively
distills this data down into a robust multi-task language-conditioned
visuo-motor policy. For (1), we use a large language model (LLM) to guide
high-level planning, and sampling-based robot planners (e.g. motion or grasp
samplers) for generating diverse and rich manipulation trajectories. To
robustify this data-collection process, the LLM also infers a code-snippet for
the success condition of each task, simultaneously enabling the data-collection
process to detect failure and retry as well as the automatic labeling of
trajectories with success/failure. For (2), we extend the diffusion policy
single-task behavior-cloning approach to multi-task settings with language
conditioning. Finally, we propose a new multi-task benchmark with 18 tasks
across five domains to test long-horizon behavior, common-sense reasoning,
tool-use, and intuitive physics. We find that our distilled policy successfully
learned the robust retrying behavior in its data collection procedure, while
improving absolute success rates by 33.2% on average across five domains. Code,
data, and additional qualitative results are available on
https://www.cs.columbia.edu/~huy/scalingup/.


# From Language Modeling to Instruction Following: Understanding the Behavior Shift in LLMs after Instruction Tuning

[Link to the paper](http://arxiv.org/abs/2310.00492v1)

## Authors
- Xuansheng Wu
- Wenlin Yao
- Jianshu Chen
- Xiaoman Pan
- Xiaoyang Wang
- Ninghao Liu
- Dong Yu

## Summary
  Large Language Models (LLMs) have achieved remarkable success, demonstrating
powerful instruction-following capabilities across diverse tasks. Instruction
fine-tuning is critical in enabling LLMs to align with user intentions and
effectively follow instructions. In this work, we investigate how instruction
fine-tuning modifies pre-trained models, focusing on two perspectives:
instruction recognition and knowledge evolution. To study the behavior shift of
LLMs, we employ a suite of local and global explanation methods, including a
gradient-based approach for input-output attribution and techniques for
interpreting patterns and concepts in self-attention and feed-forward layers.
Our findings reveal three significant impacts of instruction fine-tuning: 1) It
empowers LLMs to better recognize the instruction parts from user prompts,
thereby facilitating high-quality response generation and addressing the
``lost-in-the-middle'' issue observed in pre-trained models; 2) It aligns the
knowledge stored in feed-forward layers with user-oriented tasks, exhibiting
minimal shifts across linguistic levels. 3) It facilitates the learning of
word-word relations with instruction verbs through the self-attention
mechanism, particularly in the lower and middle layers, indicating enhanced
recognition of instruction words. These insights contribute to a deeper
understanding of the behavior shifts in LLMs after instruction fine-tuning and
lay the groundwork for future research aimed at interpreting and optimizing
LLMs for various applications. We will release our code and data soon.


# Prompting Language-Informed Distribution for Compositional Zero-Shot Learning

[Link to the paper](http://arxiv.org/abs/2305.14428v2)

## Authors
- Wentao Bao
- Lichang Chen
- Heng Huang
- Yu Kong

## Summary
  Compositional zero-shot learning (CZSL) task aims to recognize unseen
compositional visual concepts, e.g., sliced tomatoes, where the model is
learned only from the seen compositions, e.g., sliced potatoes and red
tomatoes. Thanks to the prompt tuning on large pre-trained visual language
models such as CLIP, recent literature shows impressively better CZSL
performance than traditional vision-based methods. However, the key aspects
that impact the generalization to unseen compositions, including the diversity
and informativeness of class context, and the entanglement between visual
primitives, i.e., state and object, are not properly addressed in existing
CLIP-based CZSL literature. In this paper, we propose a model by prompting the
language-informed distribution, aka., PLID, for the CZSL task. Specifically,
the PLID leverages pre-trained large language models (LLM) to 1) formulate the
language-informed class distributions which are diverse and informative, and 2)
enhance the compositionality of the class embedding. Moreover, a
visual-language primitive decomposition (VLPD) module and a stochastic logit
mixup (SLM) strategy are proposed to dynamically fuse the decisions from the
compositional and the primitive logit space. Orthogonal to the existing
literature of soft, hard, or distributional prompts, our method advocates
prompting the LLM-supported class distribution that leads to a better zero-shot
generalization. Experimental results on MIT-States, UT-Zappos, and C-GQA
datasets show the superior performance of the PLID to the prior arts.


# Prompting Code Interpreter to Write Better Unit Tests on Quixbugs Functions

[Link to the paper](http://arxiv.org/abs/2310.00483v1)

## Authors
- Vincent Li
- Nick Doiron

## Summary
  Unit testing is a commonly-used approach in software engineering to test the
correctness and robustness of written code. Unit tests are tests designed to
test small components of a codebase in isolation, such as an individual
function or method. Although unit tests have historically been written by human
programmers, recent advancements in AI, particularly LLMs, have shown
corresponding advances in automatic unit test generation. In this study, we
explore the effect of different prompts on the quality of unit tests generated
by Code Interpreter, a GPT-4-based LLM, on Python functions provided by the
Quixbugs dataset, and we focus on prompting due to the ease with which users
can make use of our findings and observations. We find that the quality of the
generated unit tests is not sensitive to changes in minor details in the
prompts provided. However, we observe that Code Interpreter is often able to
effectively identify and correct mistakes in code that it writes, suggesting
that providing it runnable code to check the correctness of its outputs would
be beneficial, even though we find that it is already often able to generate
correctly-formatted unit tests. Our findings suggest that, when prompting
models similar to Code Interpreter, it is important to include the basic
information necessary to generate unit tests, but minor details are not as
important.


# LANCAR: Leveraging Language for Context-Aware Robot Locomotion in Unstructured Environments

[Link to the paper](http://arxiv.org/abs/2310.00481v1)

## Authors
- Chak Lam Shek
- Xiyang Wu
- Dinesh Manocha
- Pratap Tokekar
- Amrit Singh Bedi

## Summary
  Robotic locomotion is a challenging task, especially in unstructured
terrains. In practice, the optimal locomotion policy can be context-dependent
by using the contextual information of encountered terrains in decision-making.
Humans can interpret the environmental context for robots, but the ambiguity of
human language makes it challenging to use in robot locomotion directly. In
this paper, we propose a novel approach, LANCAR, that introduces a context
translator that works with reinforcement learning (RL) agents for context-aware
locomotion. Our formulation allows a robot to interpret the contextual
information from environments generated by human observers or Vision-Language
Models (VLM) with Large Language Models (LLM) and use this information to
generate contextual embeddings. We incorporate the contextual embeddings with
the robot's internal environmental observations as the input to the RL agent's
decision neural network. We evaluate LANCAR with contextual information in
varying ambiguity levels and compare its performance using several alternative
approaches. Our experimental results demonstrate that our approach exhibits
good generalizability and adaptability across diverse terrains, by achieving at
least 10% of performance improvement in episodic reward over baselines. The
experiment video can be found at the following link:
https://raaslab.org/projects/LLM_Context_Estimation/.


# UPAR: A Kantian-Inspired Prompting Framework for Enhancing Large Language Model Capabilities

[Link to the paper](http://arxiv.org/abs/2310.01441v1)

## Authors
- Hejia Geng
- Boxun Xu
- Peng Li

## Summary
  Large Language Models (LLMs) have demonstrated impressive inferential
capabilities, with numerous research endeavors devoted to enhancing this
capacity through prompting. Despite these efforts, a unified epistemological
foundation is still conspicuously absent. Drawing inspiration from Kant's a
priori philosophy, we propose the UPAR prompting framework, designed to emulate
the structure of human cognition within LLMs. The UPAR framework is delineated
into four phases: "Understand", "Plan", "Act", and "Reflect", enabling the
extraction of structured information from complex contexts, prior planning of
solutions, execution according to plan, and self-reflection. This structure
significantly augments the explainability and accuracy of LLM inference,
producing a human-understandable and inspectable inferential trajectory.
Furthermore, our work offers an epistemological foundation for existing
prompting techniques, allowing for a possible systematic integration of these
methods. With GPT-4, our approach elevates the accuracy from COT baseline of
22.92% to 58.33% in a challenging subset of GSM8K, and from 67.91% to 75.40% in
the causal judgment task.


# Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection

[Link to the paper](http://arxiv.org/abs/2308.10819v2)

## Authors
- Zekun Li
- Baolin Peng
- Pengcheng He
- Xifeng Yan

## Summary
  Large Language Models (LLMs) have shown remarkable proficiency in following
instructions, making them valuable in customer-facing applications. However,
their impressive capabilities also raise concerns about the amplification of
risks posed by adversarial instructions, which can be injected into the model
input by third-party attackers to manipulate LLMs' original instructions and
prompt unintended actions and content. Therefore, it is crucial to understand
LLMs' ability to accurately discern which instructions to follow to ensure
their safe deployment in real-world scenarios. In this paper, we propose a
pioneering benchmark for automatically evaluating the robustness of
instruction-following LLMs against adversarial instructions injected in the
prompt. The objective of this benchmark is to quantify the extent to which LLMs
are influenced by injected adversarial instructions and assess their ability to
differentiate between these injected adversarial instructions and original user
instructions. Through experiments conducted with state-of-the-art
instruction-following LLMs, we uncover significant limitations in their
robustness against adversarial instruction injection attacks. Furthermore, our
findings indicate that prevalent instruction-tuned models are prone to being
``overfitted'' to follow any instruction phrase in the prompt without truly
understanding which instructions should be followed. This highlights the need
to address the challenge of training models to comprehend prompts instead of
merely following instruction phrases and completing the text. The data and code
can be found at \url{https://github.com/Leezekun/Adv-Instruct-Eval}.


# Dynamic Demonstrations Controller for In-Context Learning

[Link to the paper](http://arxiv.org/abs/2310.00385v1)

## Authors
- Fei Zhao
- Taotian Pang
- Zhen Wu
- Zheng Ma
- Shujian Huang
- Xinyu Dai

## Summary
  In-Context Learning (ICL) is a new paradigm for natural language processing
(NLP), where a large language model (LLM) observes a small number of
demonstrations and a test instance as its input, and directly makes predictions
without updating model parameters. Previous studies have revealed that ICL is
sensitive to the selection and the ordering of demonstrations. However, there
are few studies regarding the impact of the demonstration number on the ICL
performance within a limited input length of LLM, because it is commonly
believed that the number of demonstrations is positively correlated with model
performance. In this paper, we found this conclusion does not always hold true.
Through pilot experiments, we discover that increasing the number of
demonstrations does not necessarily lead to improved performance. Building upon
this insight, we propose a Dynamic Demonstrations Controller (D$^2$Controller),
which can improve the ICL performance by adjusting the number of demonstrations
dynamically. The experimental results show that D$^2$Controller yields a 5.4%
relative improvement on eight different sizes of LLMs across ten datasets.
Moreover, we also extend our method to previous ICL models and achieve
competitive results.


# Measuring Value Understanding in Language Models through Discriminator-Critique Gap

[Link to the paper](http://arxiv.org/abs/2310.00378v1)

## Authors
- Zhaowei Zhang
- Fengshuo Bai
- Jun Gao
- Yaodong Yang

## Summary
  Recent advancements in Large Language Models (LLMs) have heightened concerns
about their potential misalignment with human values. However, evaluating their
grasp of these values is complex due to their intricate and adaptable nature.
We argue that truly understanding values in LLMs requires considering both
"know what" and "know why". To this end, we present the Value Understanding
Measurement (VUM) framework that quantitatively assess both "know what" and
"know why" by measuring the discriminator-critique gap related to human values.
Using the Schwartz Value Survey, we specify our evaluation values and develop a
thousand-level dialogue dataset with GPT-4. Our assessment looks at both the
value alignment of LLM's outputs compared to baseline answers and how LLM
responses align with reasons for value recognition versus GPT-4's annotations.
We evaluate five representative LLMs and provide strong evidence that the
scaling law significantly impacts "know what" but not much on "know why", which
has consistently maintained a high level. This may further suggest that LLMs
might craft plausible explanations based on the provided context without truly
understanding their inherent value, indicating potential risks.


# DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment

[Link to the paper](http://arxiv.org/abs/2307.00329v3)

## Authors
- Yanjiang Guo
- Yen-Jen Wang
- Lihan Zha
- Zheyuan Jiang
- Jianyu Chen

## Summary
  Large language models (LLMs) encode a vast amount of semantic knowledge and
possess remarkable understanding and reasoning capabilities. Previous work has
explored how to ground LLMs in robotic tasks to generate feasible and
executable textual plans. However, low-level execution in the physical world
may deviate from the high-level textual plan due to environmental perturbations
or imperfect controller design. In this paper, we propose \textbf{DoReMi}, a
novel language model grounding framework that enables immediate Detection and
Recovery from Misalignments between plan and execution. Specifically, we
leverage LLMs to play a dual role, aiding not only in high-level planning but
also generating constraints that can indicate misalignment during execution.
Then vision language models (VLMs) are utilized to detect constraint violations
continuously. Our pipeline can monitor the low-level execution and enable
timely recovery if certain plan-execution misalignment occurs. Experiments on
various complex tasks including robot arms and humanoid robots demonstrate that
our method can lead to higher task success rates and shorter task completion
times. Videos of DoReMi are available at
\url{https://sites.google.com/view/doremi-paper}.


# Privacy-Preserving In-Context Learning for Large Language Models

[Link to the paper](http://arxiv.org/abs/2305.01639v2)

## Authors
- Tong Wu
- Ashwinee Panda
- Jiachen T. Wang
- Prateek Mittal

## Summary
  In-context learning (ICL) is an important capability of Large Language Models
(LLMs), enabling these models to dynamically adapt based on specific,
in-context exemplars, thereby improving accuracy and relevance. However, LLM's
responses may leak the sensitive private information contained in in-context
exemplars. To address this challenge, we propose Differentially Private
In-context Learning (DP-ICL), a general paradigm for privatizing ICL tasks. The
key idea for DP-ICL paradigm is generating differentially private responses
through a noisy consensus among an ensemble of LLM's responses based on
disjoint exemplar sets. Based on the general paradigm of DP-ICL, we instantiate
several techniques showing how to privatize ICL for text classification and
language generation. We evaluate DP-ICL on four text classification benchmarks
and two language generation tasks, and our empirical results show that DP-ICL
achieves a strong utility-privacy tradeoff.


# Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models

[Link to the paper](http://arxiv.org/abs/2310.00322v1)

## Authors
- Chengdong Ma
- Ziran Yang
- Minquan Gao
- Hai Ci
- Jun Gao
- Xuehai Pan
- Yaodong Yang

## Summary
  Deployable Large Language Models (LLMs) must conform to the criterion of
helpfulness and harmlessness, thereby achieving consistency between LLMs
outputs and human values. Red-teaming techniques constitute a critical way
towards this criterion. Existing work rely solely on manual red team designs
and heuristic adversarial prompts for vulnerability detection and optimization.
These approaches lack rigorous mathematical formulation, thus limiting the
exploration of diverse attack strategy within quantifiable measure and
optimization of LLMs under convergence guarantees. In this paper, we present
Red-teaming Game (RTG), a general game-theoretic framework without manual
annotation. RTG is designed for analyzing the multi-turn attack and defense
interactions between Red-team language Models (RLMs) and Blue-team Language
Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with
diversity measure of the semantic space. GRTS is an automated red teaming
technique to solve RTG towards Nash equilibrium through meta-game analysis,
which corresponds to the theoretically guaranteed optimization direction of
both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that
GRTS autonomously discovered diverse attack strategies and effectively improved
security of LLMs, outperforming existing heuristic red-team designs. Overall,
RTG has established a foundational framework for red teaming tasks and
constructed a new scalable oversight technique for alignment.


# In-Context Learning in Large Language Models: A Neuroscience-inspired Analysis of Representations

[Link to the paper](http://arxiv.org/abs/2310.00313v1)

## Authors
- Safoora Yousefi
- Leo Betthauser
- Hosein Hasanbeig
- Akanksha Saran
- Raphaël Millière
- Ida Momennejad

## Summary
  Large language models (LLMs) exhibit remarkable performance improvement
through in-context learning (ICL) by leveraging task-specific examples in the
input. However, the mechanisms behind this improvement remain elusive. In this
work, we investigate how LLM embeddings and attention representations change
following in-context-learning, and how these changes mediate improvement in
behavior. We employ neuroscience-inspired techniques such as representational
similarity analysis (RSA) and propose novel methods for parameterized probing
and measuring ratio of attention to relevant vs. irrelevant information in
Llama-2 70B and Vicuna 13B. We designed three tasks with a priori relationships
among their conditions: reading comprehension, linear regression, and
adversarial prompt injection. We formed hypotheses about expected similarities
in task representations to investigate latent changes in embeddings and
attention. Our analyses revealed a meaningful correlation between changes in
both embeddings and attention representations with improvements in behavioral
performance after ICL. This empirical framework empowers a nuanced
understanding of how latent representations affect LLM behavior with and
without ICL, offering valuable tools and insights for future research and
practical applications.


# STAR: Improving Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models

[Link to the paper](http://arxiv.org/abs/2305.15090v2)

## Authors
- Mingyu Derek Ma
- Xiaoxuan Wang
- Po-Nien Kung
- P. Jeffrey Brantingham
- Nanyun Peng
- Wei Wang

## Summary
  Information extraction tasks such as event extraction require an in-depth
understanding of the output structure and sub-task dependencies. They heavily
rely on task-specific training data in the form of (passage, target structure)
pairs to obtain reasonable performance. However, obtaining such data through
human annotation is costly, leading to a pressing need for low-resource
information extraction approaches that require minimal human labeling for
real-world applications. Fine-tuning supervised models with synthesized
training data would be a generalizable method, but the existing data generation
methods either still rely on large-scale ground-truth data or cannot be applied
to complicated IE tasks due to their poor performance. To address these
challenges, we propose STAR, a data generation method that leverages Large
Language Models (LLMs) to synthesize data instances given limited seed
demonstrations, thereby boosting low-resource information extraction
performance. Our approach involves generating target structures (Y) followed by
generating passages (X), all accomplished with the aid of LLMs. We design
fine-grained step-by-step instructions to obtain the initial data instances. We
further reduce errors and improve data quality through self-reflection error
identification and self-refinement with iterative revision. Our experiments
show that the data generated by STAR significantly improves the performance of
low-resource event extraction and relation extraction tasks, even surpassing
the effectiveness of human-curated data. Human assessment of the data quality
shows STAR-generated data exhibits higher passage quality and better align with
the task definitions compared with the human-curated data.


# CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing

[Link to the paper](http://arxiv.org/abs/2305.11738v2)

## Authors
- Zhibin Gou
- Zhihong Shao
- Yeyun Gong
- Yelong Shen
- Yujiu Yang
- Nan Duan
- Weizhu Chen

## Summary
  Recent developments in large language models (LLMs) have been impressive.
However, these models sometimes show inconsistencies and problematic behavior,
such as hallucinating facts, generating flawed code, or creating offensive and
toxic content. Unlike these models, humans typically utilize external tools to
cross-check and refine their initial content, like using a search engine for
fact-checking, or a code interpreter for debugging. Inspired by this
observation, we introduce a framework called CRITIC that allows LLMs, which are
essentially "black boxes" to validate and progressively amend their own outputs
in a manner similar to human interaction with tools. More specifically,
starting with an initial output, CRITIC interacts with appropriate tools to
evaluate certain aspects of the text, and then revises the output based on the
feedback obtained during this validation process. Comprehensive evaluations
involving free-form question answering, mathematical program synthesis, and
toxicity reduction demonstrate that CRITIC consistently enhances the
performance of LLMs. Meanwhile, our research highlights the crucial importance
of external feedback in promoting the ongoing self-improvement of LLMs.


# Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method

[Link to the paper](http://arxiv.org/abs/2310.00305v1)

## Authors
- Xuan Zhang
- Wei Gao

## Summary
  While large pre-trained language models (LLMs) have shown their impressive
capabilities in various NLP tasks, they are still under-explored in the
misinformation domain. In this paper, we examine LLMs with in-context learning
(ICL) for news claim verification, and find that only with 4-shot demonstration
examples, the performance of several prompting methods can be comparable with
previous supervised models. To further boost performance, we introduce a
Hierarchical Step-by-Step (HiSS) prompting method which directs LLMs to
separate a claim into several subclaims and then verify each of them via
multiple questions-answering steps progressively. Experiment results on two
public misinformation datasets show that HiSS prompting outperforms
state-of-the-art fully-supervised approach and strong few-shot ICL-enabled
baselines.


# RelBERT: Embedding Relations with Language Models

[Link to the paper](http://arxiv.org/abs/2310.00299v1)

## Authors
- Asahi Ushio
- Jose Camacho-Collados
- Steven Schockaert

## Summary
  Many applications need access to background knowledge about how different
concepts and entities are related. Although Knowledge Graphs (KG) and Large
Language Models (LLM) can address this need to some extent, KGs are inevitably
incomplete and their relational schema is often too coarse-grained, while LLMs
are inefficient and difficult to control. As an alternative, we propose to
extract relation embeddings from relatively small language models. In
particular, we show that masked language models such as RoBERTa can be
straightforwardly fine-tuned for this purpose, using only a small amount of
training data. The resulting model, which we call RelBERT, captures relational
similarity in a surprisingly fine-grained way, allowing us to set a new
state-of-the-art in analogy benchmarks. Crucially, RelBERT is capable of
modelling relations that go well beyond what the model has seen during
training. For instance, we obtained strong results on relations between named
entities with a model that was only trained on lexical relations between
concepts, and we observed that RelBERT can recognise morphological analogies
despite not being trained on such examples. Overall, we find that RelBERT
significantly outperforms strategies based on prompting language models that
are several orders of magnitude larger, including recent GPT-based models and
open source models.


# Understanding In-Context Learning from Repetitions

[Link to the paper](http://arxiv.org/abs/2310.00297v1)

## Authors
- Jianhao Yan
- Jin Xu
- Chiyu Song
- Chenming Wu
- Yafu Li
- Yue Zhang

## Summary
  This paper explores the elusive mechanism underpinning in-context learning in
Large Language Models (LLMs). Our work provides a novel perspective by
examining in-context learning via the lens of surface repetitions. We
quantitatively investigate the role of surface features in text generation, and
empirically establish the existence of \emph{token co-occurrence
reinforcement}, a principle that strengthens the relationship between two
tokens based on their contextual co-occurrences. By investigating the dual
impacts of these features, our research illuminates the internal workings of
in-context learning and expounds on the reasons for its failures. This paper
provides an essential contribution to the understanding of in-context learning
and its potential limitations, providing a fresh perspective on this exciting
capability.


# Prompt-Based Length Controlled Generation with Reinforcement Learning

[Link to the paper](http://arxiv.org/abs/2308.12030v2)

## Authors
- Renlong Jie
- Xiaojun Meng
- Lifeng Shang
- Xin Jiang
- Qun Liu

## Summary
  Large language models (LLMs) like ChatGPT and GPT-4 have attracted great
attention given their surprising performance on a wide range of NLP tasks.
Length controlled generation of LLMs emerges as an important topic, which
enables users to fully leverage the capability of LLMs in more real-world
scenarios like generating a proper answer or essay of a desired length. In
addition, the autoregressive generation in LLMs is extremely time-consuming,
while the ability of controlling this generated length can reduce the inference
cost by limiting the length. Therefore, we propose a prompt-based length
control method to achieve high-accuracy length controlled generation. In
particular, we adopt reinforcement learning with the reward signal given by
either trainable or rule-based reward models, which further enhances the
length-control ability of LLMs by rewarding outputs that follows pre-defined
control instruction. To enable rule-based inference, we also introduce standard
prompt extractor to collect the standard control information from users' input.
Experiments show that our method significantly improves the accuracy of
prompt-based length control for summarization task on popular datasets like
CNNDM and NYT. Both the standard prompt extractor and the RL-tuned model have
show strong generalization ability to unseen control prompt templates.


# "With Great Power Comes Great Responsibility!": Student and Instructor Perspectives on the influence of LLMs on Undergraduate Engineering Education

[Link to the paper](http://arxiv.org/abs/2309.10694v2)

## Authors
- Ishika Joshi
- Ritvik Budhiraja
- Pranav Deepak Tanna
- Lovenya Jain
- Mihika Deshpande
- Arjun Srivastava
- Srinivas Rallapalli
- Harshal D Akolekar
- Jagat Sesh Challa
- Dhruv Kumar

## Summary
  The rise in popularity of Large Language Models (LLMs) has prompted
discussions in academic circles, with students exploring LLM-based tools for
coursework inquiries and instructors exploring them for teaching and research.
Even though a lot of work is underway to create LLM-based tools tailored for
students and instructors, there is a lack of comprehensive user studies that
capture the perspectives of students and instructors regarding LLMs. This paper
addresses this gap by conducting surveys and interviews within undergraduate
engineering universities in India. Using 1306 survey responses among students,
112 student interviews, and 27 instructor interviews around the academic usage
of ChatGPT (a popular LLM), this paper offers insights into the current usage
patterns, perceived benefits, threats, and challenges, as well as
recommendations for enhancing the adoption of LLMs among students and
instructors. These insights are further utilized to discuss the practical
implications of LLMs in undergraduate engineering education and beyond.


