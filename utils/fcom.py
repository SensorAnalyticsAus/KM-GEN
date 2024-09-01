#!/usr/bin/python
import sys

if len(sys.argv) < 3:
  print('usage: {file_1} {file_2} ... {file_n}')
  print('prints common elements within files i.e sets itersection')
  sys.exit(1)

print ("".join(sorted(set.intersection(*[set(open(a).readlines()) 
                for a in sys.argv[1:]]))) )
