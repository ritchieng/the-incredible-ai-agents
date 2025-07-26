# The Incredible AI Agents

I was blessed to have created [The Incredible PyTorch](https://github.com/ritchieng/the-incredible-pytorch) 8 years ago. 

Created this to curate resources across the whole lifecycle of building, deploying, evaluating, monitoring AI Agents.

## Table of Contents
- [Planning & Design](#planning--design)
- [Building & Development](#building--development)
- [Deployment & Infrastructure](#deployment--infrastructure)
- [Evaluation & Testing](#evaluation--testing)
- [Monitoring & Observability](#monitoring--observability)
- [Multi-Agent Systems](#multi-agent-systems)
- [Frameworks & Libraries](#frameworks--libraries)
  
## Planning & Design

### Planning & Reasoning
- [ReAct](https://github.com/ysymyth/ReAct) - Reasoning and Acting with Language Models
- [Tree of Thoughts](https://github.com/princeton-nlp/tree-of-thought-llm) - Deliberate problem solving with LLMs
- [Plan-and-Solve](https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting) - Zero-shot reasoning prompting

## Building & Development

### Core Agent Frameworks
- [Google Agent Development Kit (ADK)](https://developers.google.com/ai/agents) - Google's toolkit for building AI agents
- [OpenAI Agents SDK](https://platform.openai.com/docs/agents) - OpenAI's official agent development framework
- [LangChain](https://github.com/langchain-ai/langchain) - Building applications with LLMs through composability
- [LangGraph](https://github.com/langchain-ai/langgraph) - Build resilient language agents as graphs
- [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLM applications
- [CrewAI](https://github.com/joaomdmoura/crewAI) - Framework for orchestrating role-playing, autonomous AI agents
- [AutoGen](https://github.com/microsoft/autogen) - Multi-agent conversation framework
- [Haystack](https://github.com/deepset-ai/haystack) - End-to-end NLP framework
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) - SDK for integrating AI services

### Tool Integration
- [LangChain Tools](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/tools) - Pre-built tools for agents
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) - Native tool integration
- [Toolformer](https://github.com/lucidrains/toolformer-pytorch) - Language models that can use tools

### Memory & Context Management
- [MemGPT](https://github.com/cpacker/MemGPT) - Teaching LLMs memory management
- [LangChain Memory](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/memory) - Memory components
- [Zep](https://github.com/getzep/zep) - Long-term memory for AI assistants

## Deployment & Infrastructure

### Serving & APIs
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [Ollama](https://github.com/ollama/ollama) - Run LLMs locally
- [LocalAI](https://github.com/mudler/LocalAI) - Self-hosted OpenAI alternative
- [FastAPI for AI](https://github.com/tiangolo/fastapi) - Modern API framework

### Orchestration
- [Airflow](https://github.com/apache/airflow) - Workflow orchestration
- [Prefect](https://github.com/PrefectHQ/prefect) - Modern workflow orchestration
- [Temporal](https://github.com/temporalio/temporal) - Durable execution platform

### Scaling & Infrastructure
- [Ray](https://github.com/ray-project/ray) - Distributed computing for AI
- [Kubernetes](https://github.com/kubernetes/kubernetes) - Container orchestration
- [Docker](https://github.com/docker/docker-ce) - Containerization

## Evaluation & Testing

### Agent Evaluation Frameworks
- [AgentBench](https://github.com/THUDM/AgentBench) - Evaluating LLMs as agents
- [SWE-bench](https://github.com/princeton-nlp/SWE-bench) - Software engineering evaluation
- [HumanEval](https://github.com/openai/human-eval) - Code generation evaluation

### Testing Tools
- [LangSmith](https://docs.smith.langchain.com/) - Testing and evaluation platform
- [Promptfoo](https://github.com/promptfoo/promptfoo) - Test and evaluate LLM outputs
- [DeepEval](https://github.com/confident-ai/deepeval) - LLM evaluation framework

### Benchmarks & Datasets
- [HELM](https://github.com/stanford-crfm/helm) - Holistic evaluation of language models
- [BigBench](https://github.com/google/BIG-bench) - Beyond the imitation game benchmark
- [MMLU](https://github.com/hendrycks/test) - Massive multitask language understanding

## Monitoring & Observability

### LLM Observability
- [LangFuse](https://github.com/langfuse/langfuse) - Open source LLM engineering platform
- [Phoenix](https://github.com/Arize-ai/phoenix) - ML observability in a notebook
- [Weights & Biases](https://github.com/wandb/wandb) - Experiment tracking and monitoring

### Performance Monitoring
- [Prometheus](https://github.com/prometheus/prometheus) - Monitoring and alerting toolkit
- [Grafana](https://github.com/grafana/grafana) - Observability and data visualization
- [New Relic AI Monitoring](https://newrelic.com/solutions/ai-monitoring) - AI application monitoring

### Logging & Tracing
- [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-python) - Observability framework
- [Jaeger](https://github.com/jaegertracing/jaeger) - Distributed tracing platform

## Multi-Agent Systems

### Coordination & Communication
- [Mesa](https://github.com/projectmesa/mesa) - Agent-based modeling framework
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) - Multi-agent reinforcement learning
- [ChatDev](https://github.com/OpenBMB/ChatDev) - Communicative agents for software development

### Swarm Intelligence
- [OpenSwarm](https://github.com/openai/swarm) - Educational framework for multi-agent orchestration
- [AutoSwarm](https://github.com/bestmoon/AutoSwarm) - Swarm intelligence for AI agents

## Frameworks & Libraries

### Popular Agent Frameworks
- [Transformers Agents](https://github.com/huggingface/transformers/tree/main/src/transformers/agents) - Hugging Face agents
- [Guidance](https://github.com/guidance-ai/guidance) - Programming framework for LLMs
- [DSPy](https://github.com/stanfordnlp/dspy) - Programming foundation models

### Utility Libraries
- [LiteLLM](https://github.com/BerriAI/litellm) - Use any LLM as a drop in replacement
- [Instructor](https://github.com/jxnl/instructor) - Structured outputs from LLMs
- [Pydantic AI](https://github.com/pydantic/pydantic-ai) - Agent framework built on Pydantic

## Contributing

Feel free to submit pull requests to add more resources! Please ensure:
- Resources are actively maintained
- Include brief descriptions
- Organize by appropriate lifecycle phase
- Add relevant links and documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

