import sys
from datetime import datetime
from main import main

sec2min_factor = 1.0/60.0

start_time = datetime.now()

main(sys.argv)

elapsed_time = datetime.now() - start_time

print ''
print ''
print '======================================================='
print ''
print 'program total wall time: ',  sec2min_factor*elapsed_time.seconds, ' min'
print ''
print '======================================================='

