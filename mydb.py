#! /usr/bin/env python3

'''
2018.8.17 ver 0.1
'''

import sqlite3
from contextlib import contextmanager
import xml.etree.ElementTree as ET

# import os, pickle, re, subprocess, time
# from datetime import datetime
# from functools import wraps


####################################################################################################
# sqlite3
####################################################################################################

@contextmanager
def connect(file=':memory:'):
  conn = sqlite3.connect(file)
  try:
    c = conn.cursor()
    yield c
  finally:
    c.close()
    conn.commit()
    conn.close()


####################################################################################################
# xml
####################################################################################################

def xmlparse(path):
  root = ET.parse(path).getroot()
  return root
