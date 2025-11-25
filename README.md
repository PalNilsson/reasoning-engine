# Ask PanDA Reasoning Engine

The Ask PanDA Reasoning Engine is a modular, transparent, and fully testable intent-routing 
framework designed to power natural-language (and optional audio-based) interactions with 
the PanDA workload management ecosystem. It provides a deterministic 
perceive → reason → act pipeline that interprets user input, extracts relevant entities such 
as task or job IDs, infers intent, and selects the appropriate Ask PanDA client (e.g., 
TaskQuery, LogAnalysis, QueueQuery, MetadataAnalysis) to handle the request.

Ask PanDA integrates AI-powered assistance with PanDA’s distributed workload and data 
management system, enabling users to query operational status, diagnose failures, inspect 
metadata, explore documentation, and understand workflows using natural, conversational 
queries. The reasoning engine sits at the core of this interface, providing predictable, 
explainable behavior that complements rather than replaces large language models.

## Why a Reasoning Engine?

Traditional LLM-based routing can be opaque, inconsistent, and difficult to validate.
This project solves those challenges by introducing a deterministic, rule-based classifier 
that:

* Interprets user intention through transparent heuristic logic 
* Extracts PanDA-specific entities (task IDs, site names, queues)
* Selects the appropriate Ask PanDA client with fully explainable reasoning 
* Supports both text and audio input paths
* Produces structured perception and reasoning traces for debugging or UI inspection 
* Integrates cleanly with Anthropic’s Model Context Protocol (MCP) tools and clients

The system is designed to be robust, testable, predictable, and safe, especially in
environments where operational correctness matters (monitoring, diagnosing job failures,
automated triage, etc.).

**Note**: This is intended as an alternative to the current LLM-based routing in Ask PanDA, which
is using LLM based classification. The reasoning engine can be used alongside or instead of that system.
E.g., the reasoning engine calculates a confidence score, and if it is above a threshold, it can be used to route the query.
If it falls below the threshold, the LLM-based routing can be used as a fallback.

### Audio Input (Optional, in development)

When configured with a speech-to-text (STT) module, the engine
accepts audio queries and seamlessly feeds them into the reasoning pipeline.

### Extensive Test Harness

The repository includes a test harness that:

* Loads test cases from a JSON file 
* Maps each prompt to its expected handler 
* Runs full reasoning x validation 
* Supports file or stdout reporting 
* Exits with non-zero status on test failures (CI-ready)

This provides regression protection and ensures the routing logic remains stable as new features are added.

### MCP-Friendly

The engine is built to integrate with Ask PanDA MCP Clients, which handle:

* Documentation lookups 
* Task status queries 
* Automatic log analysis 
* Task/job metadata interpretation
* Pilot monitoring (to be implemented)

The reasoning engine decides which client to call; those clients decide how to answer.

## Quick Start

1. Prepare prompts.json

Create a prompts.json file in the repo root (provided in the distribution). For example:

```json
{
  "What is PanDA?": "DocumentQuery",
  "Explain how I can run a job on the grid?": "DocumentQuery",
  "Why did job 123456 fail?": "LogAnalysis",
  "What happened to job 123456?": "LogAnalysis",
  "Explain job 123456.": "MetadataAnalysis",
  "What is happening with task 123456?": "MetadataAnalysis",
  "Please summarize task 123456": "TaskQuery",
  "Show me the status of queue ANALY_BNL_LONG": "QueueQuery",
  "Are there any problems with pilots at BNL_SITE_XYZ?": "PilotMonitor",
  "How does PanDA assign jobs to queues?": "DocumentQuery"
}
```

2. Run the test harness

From the repo root:

```bash
python test_panda_reasoning_engine.py --input prompts.json
```

You’ll see detailed output for each test, for example:

```text
=== PanDAReasoningEngine Automated Test ===

--- Test 1/10 ---
Prompt: 'What is PanDA?'
Expected handler: DocumentQuery
Actual handler:   DocumentQuery
Result: ✓ PASS

[Perception]
  intent:   documentation
  entities: {}

[Reasoning]
  goal:       Answer a documentation or how-to question about PanDA.
  handler:    DocumentQuery
  confidence: 0.60

[Formatted Answer]
[Routing] intent=documentation, handler=DocumentQuery, confidence=0.60

[DocumentQuery] Handling request
  goal: Answer a documentation or how-to question about PanDA.
  confidence: 0.60
  entities: {}
  prompt: 'What is PanDA?'

----------------------------------------------------------------
...
=== Summary (stdout) ===
Total tests:   10
Passed tests:  10
Failed tests:  0
```

If any test fails, the script will exit with a non-zero status code, which is ideal for CI. Also, the 
'Result:' will show FAIL.

3. Save detailed output to a file

If you want a log file (e.g. for CI artifacts):

```bash
python test_panda_reasoning_engine.py --input prompts.json --output results.txt
```

* Detailed per-test logs → results.txt 
* Short summary → still printed to stdout 
* Exit code → 0 on success, 1 if any failures

4. Using the engine directly in Python (*not yet tested properly*)

If you want to play with the reasoning engine in a Python shell or another script:

```python
from panda_reasoning_engine import PanDAReasoningEngine
from test_panda_reasoning_engine import (
    DocumentQuery,
    QueueQuery,
    TaskQuery,
    LogAnalysis,
    PilotMonitor,
    MetadataAnalysis,
    Selection,
)

# Build handlers
document_query = DocumentQuery()
queue_query = QueueQuery()
task_query = TaskQuery()
log_analysis = LogAnalysis()
pilot_monitor = PilotMonitor()
metadata_analysis = MetadataAnalysis()
selection = Selection()

# Construct the engine
engine = PanDAReasoningEngine(
    document_query=document_query,
    queue_query=queue_query,
    task_query=task_query,
    log_analysis=log_analysis,
    pilot_monitor=pilot_monitor,
    metadata_analysis=metadata_analysis,
    selection=selection,
    stt_callable=None,  # or your STT function if you want audio
)

# Try a query
result = engine.handle_text("Why did job 123456 fail?")

print("Intent:", result.perception.intent)
print("Handler:", result.reasoning.handler_name)
print("Formatted answer:\n", result.formatted_answer)
```

## Audio Input (Optional, in development)

To use audio input, you need to provide a speech-to-text (STT) callable. A script
mainly intended for testing is provided in the  `demo_audio.py` and `mic_demo.py` files.
The code demonstrates how to turn audio into text, and is based on OpenAI's Whisper model, 
but can be replaced with any STT system that matches the expected callable signature. 

A correspondinf preliminary version of a Streamlit app is provided in `streamlit_mic_app.py`.


## OpenAI Fine-Tuning

The reasoning engine can produce structured JSON outputs that can be used to fine-tune
OpenAI models for intent classification. This allows you to train a model to predict
the appropriate handler based on user prompts, potentially improving routing accuracy
over time.

To generate fine-tuning data, you can modify the test harness to output JSONL files
suitable for OpenAI fine-tuning. Each entry should include the prompt and the
expected handler as the label.

Example JSONL entry (not complete):

```json
{"messages": 
  [{"role": "system", 
    "content": "You are the AskPanDA Triage Assistant. You help ATLAS users diagnose and understand PanDA tasks and jobs, including queue status and pilot issues.\nKnowledge cutoff: 2024-06\nReasoning: high\n# Valid channels: analysis, commentary, final.\nCalls to these tools must go to the commentary channel: 'functions'."}, 
    {"role": "developer", 
      "content": "# Instructions\nYou are the AskPanDA Triage Assistant. Use the provided tool functions to look up PanDA metadata, retrieve context, and inspect logs before answering.\n# Tools\n## functions\nnamespace functions {\n  // (Schemas are defined externally; the tool names are:\n  //  - metadata_search\n  //  - context_retrieve\n  //  - log_query\n}\n"}, 
    {"role": "user", 
      "content": "What is PanDA?"}, 
    {"role": "assistant", 
      "channel": "commentary", 
      "name": "functions.context_retrieve", 
      "arguments": {}}, 
    {"role": "tool", 
      "name": "functions.context_retrieve", 
      "content": ""}, 
    {"role": "assistant", 
      "channel": "final", 
      "content": "[DocumentQuery] Handling request\n  goal: Answer a documentation or how-to question about PanDA.\n  confidence: 0.60\n  entities: {'queues': ['what', 'panda']}\n  prompt: 'What is PanDA?'"
    }
  ]}
..
```
Initial JSONL files can be created like so:

```bash
python test_panda_reasoning_engine.py --input prompts.json --jsonl-output fine_tuning_data.json
```

A second script can be used to loop over the prompts in the generated JSONL file and
and query an LLM to produce a more final fine-tuning dataset:

```bash
python generate_finetuning_dataset.py --input fine_tuning_data.json --output openai_finetuning.jsonl --llm openai --model gpt-4-vision-preview
```

Function arguments will need to be added manually based on the expected schema for each tool.

(Currently in progress, not all options are implemented and it will not contact the LLM just yet).


