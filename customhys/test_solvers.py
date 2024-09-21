from unittest.mock import patch

from customhys.experiment import Experiment


def test_brute_force():
  exp = Experiment("./tests/brute_force.json")
  exp.run()

def test_basic_mhs():
  exp = Experiment("./tests/basic_mhs.json")
  exp.run()

def test_static():
  exp = Experiment("./tests/static.json")
  exp.run()

def test_dynamic():
  exp = Experiment("./tests/dynamic.json")
  exp.run()

def test_neural_network():
  exp = Experiment("./tests/neural_network.json")
  exp.run()
    