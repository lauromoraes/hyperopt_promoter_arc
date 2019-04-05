def test():
	i=0
	while True:
		print(i)
		i+=1
		yield(i)
gen = test()

gen.next()
gen.next()
gen.next()
# cnt=0
# for t in test():
# 	if cnt>10:
# 		break
# 	cnt+=1