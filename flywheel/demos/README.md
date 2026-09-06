
# Alignment Flywheel Demos

Interactive demonstrations of the **Alignment Flywheel** and its **Safety Oracle** across user-facing AI agents and embodied robot agents.

The central idea is simple:

> **An AI system proposes a response or action.  
> A Safety Oracle checks it before it reaches a user or affects the environment.**

The Safety Oracle is the maintained alignment artifact. Specifications are used to create and verify it before deployment, and the Alignment Flywheel provides a process for testing and correcting it when shortcomings are discovered.

---

## 🎥 5-minute demonstration video

[![Watch the Alignment Flywheel demo](https://img.youtube.com/vi/lsVreyMnrd0/maxresdefault.jpg)](https://youtu.be/lsVreyMnrd0)

**[Watch the video on YouTube](https://youtu.be/lsVreyMnrd0)**

---

## What is being demonstrated?

The demos show the same governance principle in two very different settings:

### 💬 User-facing AI agents

Users can choose an application domain such as:

- 🎓 Education
- 🏥 Healthcare
- 🎧 Customer service

The interface contains prepared examples, but it is **not limited to scripted cases**.

Visitors can type their own:

1. user message;
2. candidate agent response.

The Safety Oracle evaluates the interaction and returns an operational decision:

| Outcome | Meaning |
|---|---|
| ✅ **ALLOW** | The Safety Oracle considers the proposed response acceptable. |
| ⚠️ **ESCALATE** | The Safety Oracle is uncertain and routes the case for human review. |
| ⛔ **BLOCK** | The proposed response should not be sent. |

The interaction is evaluated **in context**. The same words can be appropriate or inappropriate depending on what the user asked, what the agent proposes to do, and the requirements of the selected domain.

### Free-form interaction

A major goal of the demo is to move beyond carefully prepared examples.

Visitors can enter their own wording, including informal or unexpected inputs, and inspect the resulting decision.

Supporting this required substantial implementation work around:

- user-message / proposed-response evaluation;
- domain-specific scope;
- uncertainty and escalation;
- structural checks;
- over-blocking;
- previously unseen phrasings;
- regression testing after corrections.

---

## 🔄 What happens when the Safety Oracle is uncertain?

Uncertainty is not treated as an automatic failure.

Instead:

```text
                 ┌──────── ALLOW
                 │
AI → Safety Oracle ─────── ESCALATE → Human review
                 │                         │
                 └──────── BLOCK           ▼
                                      Test similar cases
                                             │
                                             ▼
                                      Flywheel update
                                             │
                                             ▼
                                      Safety Oracle
