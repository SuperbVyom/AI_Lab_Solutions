# AI Laboratory Submissions

Each lab has its own subfolder containing the report and program deliverables.

| Lab folder | Contents |
|---|---|
| [agents_lab](agents_lab/) | [Report](agents_lab/agents_lab_report.md), [program](agents_lab/goal_based_agent.py) |
| [search_lab](search_lab/) | [Report](search_lab/search_lab_report.md), [program](search_lab/search_agent.py) |
| [bn_lab](bn_lab/) | [Report](bn_lab/bn_lab_report.md), [program](bn_lab/bn_language_model.py), [20 generated sentences](bn_lab/bn_generated_sentences.txt) |
| [logic_lab](logic_lab/) | [Report](logic_lab/logic_lab_report.md), [program](logic_lab/logical_planner.py) |
| [neural_models_lab](neural_models_lab/) | [Report](neural_models_lab/neural_models_lab_report.md), [program](neural_models_lab/neural_xor.py) |

The search, navigation, planner, and n-gram programs use only the Python standard library. The neural source uses PyTorch, as requested by the lab; its training metrics still need to be produced in an environment with PyTorch installed. The n-gram program writes its 20 sampled sentences to `bn_generated_sentences.txt` when run.
