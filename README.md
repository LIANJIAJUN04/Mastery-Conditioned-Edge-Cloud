# Mastery-Conditioned Edge–Cloud Model Selection

A simulation-based study of edge–cloud AI model selection for wearable
skill training in network-engineering education.

**First Author:** Dr. Subarmaniam A/L Kannan
**Second Author:** Lian Jia Jun (243UT246W5)
**Faculty of Information Science and Technology (FIST), Multimedia University (MMU), Melaka Campus**

---

## What this project is

Imagine a student practising a hands-on networking skill — terminating an
RJ45 cable, preparing fibre, or punching down a patch panel — while
wearing a smartwatch or similar sensor. An AI system watches their
movements and gives feedback. That AI system has a choice to make for
every single action the student takes: should it use a tiny, fast model
running directly on the wearable device, a medium-sized model on a nearby
edge server, or send the data all the way to a large cloud model for the
most accurate (but slowest and most expensive) analysis?

Existing systems make this choice based on how *confident* the AI model
is and how *good the network connection* is at that moment. This
project tests a different idea, proposed by the supervisor: that the
choice should also depend on how **skilled the student already is**. A
complete beginner probably needs the most careful (cloud-level) feedback;
a student who has already mastered the basic motion can be handled
confidently by the tiny on-device model, freeing up network and server
resources for everyone else.

Testing this idea properly would normally require real wearable hardware,
university ethics approval, and a live class of students over an entire
semester. This internship instead builds an independent **code
simulation** that tests the same idea using real, publicly available
datasets wherever possible, so the core question can be answered now,
without waiting for the full field study.

## What was done

The simulation was built in stages, each one replacing a guessed
assumption with a real dataset, so that every result could be traced back
to a specific, real source of evidence rather than to made-up numbers:

1. A full simulation of four competing decision strategies (cloud-only,
   device-only, confidence-based, and skill-based — the strategy this
   project is actually testing).
2. Two independent real wearable-sensor datasets (UCI HAR and PAMAP2),
   used to train a real tiny AI model and measure how confident and how
   well-calibrated it actually is.
3. A real dataset of ~525,000 student learning records (ASSISTments),
   used to fit the skill-tracking model on real student behaviour instead
   of guessed numbers.
4. A real dataset of repeated hands-on practice attempts (ROSMA, robotic
   surgery training data), used as the closest available real-world test
   of whether a person's skill level can actually be measured from sensor
   data.

Every result was checked against the specific, numeric predictions the
original research proposal makes, so the test could not be "moved" after
the fact — including the one prediction that, once grounded in real data,
did not meet its target. That result is reported openly rather than
hidden, in keeping with this project's standing principle of reporting
negative and partial findings honestly, not only successes.

## Results, in brief

- The proposal's central claim — that skill-based routing produces a
  distinctive, rising-then-falling pattern in how often the system relies
  on the cloud — held up consistently, including once the simulation was
  fully grounded in real data.
- The system's latency and energy-efficiency targets were met.
- One specific numeric target (how much cloud usage drops, at what
  accuracy cost) was not met once real student data was used, for a
  specific, understood reason that is reported in full rather than
  smoothed over.
- Whether a person's skill level can be read directly from sensor data
  remains an open question after the first real test attempted here — not
  confirmed, not refuted.

Full technical detail, every number, and the complete methodology for
each stage are in [`simulation/README.md`](simulation/README.md). A
narrative summary of the whole project, written for inclusion in a
broader internship report, is in
[`Report/Project_Summary.md`](Report/Project_Summary.md).

## How to run

```powershell
cd simulation
.\venv\Scripts\python.exe run_experiment.py
```

This one command downloads everything it needs and runs the full
simulation. Results (CSVs, JSON reports, PNG charts) are written to
`simulation/results/`.
