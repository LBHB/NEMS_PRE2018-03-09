
# A Test of the above
with open('/tmp/ooo.csv', 'w') as f:
    n = 0
    for i in range(200):
        f.write(str(n))
        n += 1
        for j in range(1, 8):
            f.write(', ')
            f.write(str(n))
            n += 1
        f.write('\n')

q = loadfromcsv('/tmp/ooo.csv', '/tmp/ooo.json')
q.savetocsv('/tmp/')

assert(q.__matrix__[1,2,3] == 490)
assert(q.as_average_trial()[3,3] == 747.0)

p = q.jackknifed_by_reps(10, 0)
assert(np.isnan(p.__matrix__[1,3,0]))
assert(np.isnan(p.__matrix__[2,5,0]))
assert(not np.isnan(p.__matrix__[2,5,1]))

r = q.jackknifed_by_time(200, 199)
assert(np.isnan(r.__matrix__[-1,3,-1]))
assert(not np.isnan(r.__matrix__[1,3,1]))

#print(r.__matrix__)
#print(r.as_single_trial())

print("Tests passed")
