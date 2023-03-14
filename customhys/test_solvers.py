from .experiment import Experiment

def test_brute_force():
  exp = Experiment(f"./exconf/tests/brute_force.json")
  exp.run()

def test_basic_mhs():
  exp = Experiment(f"./exconf/tests/basic_mhs.json")
  exp.run()

def test_static():
  exp = Experiment(f"./exconf/tests/static.json")
  exp.run()

def test_dynamic():
  exp = Experiment(f"./exconf/tests/dynamic.json")
  exp.run()

def test_neural_network():
  exp = Experiment(f"./exconf/tests/neural_network.json")
  exp.run()
    