import unittest
from experiment import Experiment

class CustomhysSolvers(unittest.TestCase):
  test_configs = './exconf/tests'

  def test_brute_force(self):
    exp = Experiment(f"{self.test_configs}/brute_force.json")
    exp.run()

  def test_basic_mhs(self):
    exp = Experiment(f"{self.test_configs}/basic_mhs.json")
    exp.run()

  def test_static(self):
    exp = Experiment(f"{self.test_configs}/static.json")
    exp.run()

  def test_dynamic(self):
    exp = Experiment(f"{self.test_configs}/dynamic.json")
    exp.run()

  def test_neural_network(self):
    exp = Experiment(f"{self.test_configs}/neural_network.json")
    exp.run()
            
if __name__ == "__main__":
  unittest.main()
  