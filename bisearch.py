size = 7
start = 1#0#5#size-2
end = 0
a = [i for i in range(start, end, -1)]
s = [i-.5 for i in range(start+1, end, -1)]
def bisearch(a:list, e):
	l,r = 0,len(a)-1
	if r-l<1: return l
	i=(l+r)>>1
	while l!=i and r!=i:
		if e<a[i]:	l=i
		else     :	r=i
		i=(l+r)>>1
	if e<a[i]: i+=1
	return i

print('pos	', [bisearch(a, e) for e in s])
print('a	', a)
print('s	', s)

print('insert')
for e in s:
	if len(a)<size: a.append(None)
	i = bisearch(a, e)
	a[i+1:len(a)]=a[i:len(a)-1]
	a[i] = e
	print(i, e, a)
