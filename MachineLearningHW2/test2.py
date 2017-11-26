a='IGADKYFHARGNYDAA'
b='KGADKYFHARGNYEAA'

u=zip(a,b)
d=dict(u)

c=0
for i,j in u: 
    if i!=j:
        c=c+1 
    else: 
        c=c+0

print c
