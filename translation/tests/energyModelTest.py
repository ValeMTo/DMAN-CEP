import unittest
from translation.energyAgentModel import EnergyModelClass
from translation.tests.xmlGeneratorTest import to_pretty_xml
from unittest.mock import MagicMock

class TestEnergyModelClass(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        self.model = EnergyModelClass(self.logger
        )

    


    