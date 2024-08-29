from .experiment import Experiment

def test_brute_force():
  exp = Experiment("./exconf/tests/brute_force.json")
  exp.run()

def test_basic_mhs():
  exp = Experiment("./exconf/tests/basic_mhs.json")
  exp.run()

def test_static():
  exp = Experiment("./exconf/tests/static.json")
  exp.run()

def test_dynamic():
  exp = Experiment("./exconf/tests/dynamic.json")
  exp.run()

def test_neural_network():
  exp = Experiment("./exconf/tests/neural_network.json")
  exp.run()
    