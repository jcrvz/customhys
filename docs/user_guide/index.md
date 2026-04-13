# User Guide

Welcome to the CUSTOMHyS User Guide. This section explains the framework's architecture,
walks through hands-on tutorials, and describes the data structures produced by experiments.

```{toctree}
:maxdepth: 2

architecture
tutorials
data_structure
```

## Overview

CUSTOMHyS follows a layered heuristic design where:

1. **Search operators** (low-level heuristics) modify individual agent positions.
2. A **metaheuristic** (mid-level heuristic) sequences one or more operators into a
   complete optimisation algorithm.
3. A **hyper-heuristic** (high-level heuristic) searches the space of possible
   metaheuristics to find the best operator sequence for a given problem.
4. An **experiment** orchestrates many hyper-heuristic runs across problems and
   dimensions.

The diagram below illustrates the module dependency structure:

```{image} ../../docfiles/dependency_diagram.png
:alt: Module dependency diagram
:width: 80%
:align: center
```
