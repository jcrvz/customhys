# Example notebooks

| Notebook | Language | Contents |
| --- | --- | --- |
| [Tutorial_English.ipynb](Tutorial_English.ipynb) | English | Complete tutorial |
| [Tutorial_Spanish.ipynb](Tutorial_Spanish.ipynb) | Español | Tutorial completo |
| [Tutorial_Portuguese_BR.ipynb](Tutorial_Portuguese_BR.ipynb) | Português (Brasil) | Tutorial completo em PT-BR |
| [quickTest.ipynb](quickTest.ipynb) | English | Short linear-system operator example |

The three tutorials cover the same workflow: benchmark functions, population initialisation,
operator collections, manual search, metaheuristics, independent replicas, hyper-heuristic
search with initialisers, saved results, replay of the discovered algorithm, and CEC2005.
They also compare all ten population initialisation methods (including two beta
configurations), define and solve Rosenbrock through the custom-problem interface,
and show population snapshots, trajectories, occupancy maps and an inline animation.
These views record population states after selection, not rejected proposals.

## Run locally

Use Python 3.10–3.12. From the repository root, in your Python environment:

```shell
python -m pip install -e ".[examples]"
python -m jupyterlab examples
```

Select the kernel for that environment, then **Restart Kernel and Run All Cells**.
Repeat this after editing or updating the framework. Re-running imports does not
reload modules already in memory, even with an editable installation. Each notebook
prints its Python executable and operators module path to help identify the active checkout.
The notebooks also work when the kernel starts in the repository root or another
directory, provided this checkout is installed. Packaged collections are read via
`importlib.resources`; the historical copies in `examples/collections` are not required.

No LaTeX or TensorFlow installation is needed. An import warning about TensorFlow
can appear; these examples use the static solver, not the machine-learning module.
For a standalone notebook, `%pip install "customhys[examples]"` installs the published
release, which may differ from the API in this checkout.

The budgets are intentionally small and use explicit seeds. Reproducibility is
within the same dependency versions. Increase iterations, replicas and independent
seeds for research; these examples do not establish algorithm rankings.

Tutorial runs create a unique `data_files/raw/tutorial-…` directory and a merged JSON
file under the kernel's working directory. Collection-generation scratch files are
temporary. `quickTest` does not write result files. Saved notebook outputs belong to
the last saved execution; restart the kernel and run all cells to refresh them after changes.

## Executar em português

Abra [Tutorial_Portuguese_BR.ipynb](Tutorial_Portuguese_BR.ipynb). O conteúdo, os rótulos
dos gráficos e as explicações estão em português do Brasil; os identificadores da API
permanecem iguais aos do framework. A comparação das inicializações mostra onde cada
método coloca os indivíduos antes da busca. Os exemplos seguintes acompanham o movimento
da população e incluem Rosenbrock, definida diretamente por uma função Python e seus limites.
Uma animação permite pausar e rever essa execução. Instale o projeto com o comando acima,
selecione o kernel desse ambiente e reinicie o kernel antes de executar todas as células.

Reinicie também depois de alterar o framework. Executar os imports outra vez não
recarrega os módulos que já estão na memória, mesmo com a instalação `-e`. A primeira
célula de código mostra o Python e o caminho de `operators.py` que estão em uso.
Uma sessão que carregou os operadores antigos pode continuar retornando posições que
não correspondem ao valor da função informado. As verificações dos exemplos de Sphere
e Rosenbrock detectam essa inconsistência; mantenha essas verificações ao executar.

O tutorial usa limites pequenos para aprendizado e salva cada execução em uma pasta
exclusiva `data_files/raw/tutorial-…`, relativa ao diretório de trabalho do kernel.
Para experimentos científicos, aumente o número de iterações, repetições e sementes
independentes. Não é necessário instalar LaTeX nem TensorFlow.

## Validate examples

```shell
python -m pip install -e ".[dev,examples]"
python -m pytest tests/test_notebooks.py -v
```

These integration tests execute every notebook in a fresh kernel and empty temporary
directory, check numerical invariants and require rendered figures. Core-only test
environments skip this module if notebook dependencies are absent; the dedicated CI
job installs them explicitly. When editing a tutorial, keep all three languages aligned.
