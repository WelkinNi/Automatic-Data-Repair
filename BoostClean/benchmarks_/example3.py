#!/usr/bin/env python
import sys

sys.path.append('/Users/sanjayk/Documents/research/boostclean/activedetect/')

from USCensusBenchmark import USCensusBenchmark
from NFLBenchmark import NFLBenchmark
from BPBenchmark import BPBenchmark
from RetailBenchmark import RetailBenchmark
from SensorBenchmark import SensorBenchmark
from EmergBenchmark import EmergBenchmark
from HousingBenchmark import HousingBenchmark
from TitanicBenchmark import TitanicBenchmark

u = USCensusBenchmark()
print u.getResults()

"""
n = NFLBenchmark()
print n.getResults()

b = BPBenchmark()
print b.getResults()

r = RetailBenchmark()
print r.getResults()

s = SensorBenchmark()
print s.getResults()

e = EmergBenchmark()
print e.getResults()

t = TitanicBenchmark()
print t.getResults()
"""



