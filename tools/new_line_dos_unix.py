# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 00:29:45 2019

@author: Román de las Heras monkshow92
"""

"""
convert dos linefeeds (crlf) to unix (lf)
it corrects the following erros:
    * UnpicklingError: the STRING opcode argument must be quoted;
 
"""

def unix_dos_pikle(address):
    original = address
    destination = address
    content = ''
    outsize = 0
    with open(original, 'rb') as infile:
        content = infile.read()
        with open(destination, 'wb') as output:
            for line in content.splitlines():
                outsize += len(line) + 1
                output.write(line + str.encode('\n'))
                
    print("Done. Saved %s bytes." % (len(content)-outsize))