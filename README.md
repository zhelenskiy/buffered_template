# buffered_template

That is template that performs different test (mostly, performance ones) to compare speed of lazy sequences and eager ones.

Results are (on my computer):
```
Testing composition
6
8
10
12
14
Testing reduce
10
Performance tests:
	#0:	Just range - lazy
Time difference = 1128 ms
Time difference = 1128362 µs
Time difference = 1128362051 ns
-243309312
	#0:	Just range - eager (with storage)
Time difference = 2771 ms
Time difference = 2771340 µs
Time difference = 2771340145 ns
-243309312
	#0:	Just range - eager (without storage)
Time difference = 106 ms
Time difference = 106050 µs
Time difference = 106050688 ns
-243309312
	#1:	Big composition - lazy
Time difference = 2 ms
Time difference = 2004 µs
Time difference = 2004765 ns
	#1:	Big composition - eager (with storage, range loop)
Time difference = 2 ms
Time difference = 2114 µs
Time difference = 2114056 ns
-726379968
	#1:	Big composition - eager (with storage, index loop)
Time difference = 1 ms
Time difference = 1841 µs
Time difference = 1841015 ns
-726379968
	#2:	Big number of elements - lazy
Time difference = 320 ms
Time difference = 320181 µs
Time difference = 320181191 ns
-243309312
	#2:	Big number of elements - eager (max performance)
Time difference = 102 ms
Time difference = 102671 µs
Time difference = 102671398 ns
-243309312
	#3:	10 elements - lazy
Time difference = 0 ms
Time difference = 0 µs
Time difference = 140 ns
55
	#3:	10 elements - eager (max performance)
Time difference = 0 ms
Time difference = 0 µs
Time difference = 75 ns
55

	#3:	100 elements - lazy
Time difference = 0 ms
Time difference = 0 µs
Time difference = 164 ns
5050
	#3:	100 elements - eager (max performance)
Time difference = 0 ms
Time difference = 0 µs
Time difference = 82 ns
5050

	#3:	1000 elements - lazy
Time difference = 0 ms
Time difference = 1 µs
Time difference = 1438 ns
500500
	#3:	1000 elements - eager (max performance)
Time difference = 0 ms
Time difference = 0 µs
Time difference = 142 ns
500500

	#3:	10000 elements - lazy
Time difference = 0 ms
Time difference = 15 µs
Time difference = 15230 ns
50005000
	#3:	10000 elements - eager (max performance)
Time difference = 0 ms
Time difference = 1 µs
Time difference = 1702 ns
50005000

	#3:	100000 elements - lazy
Time difference = 0 ms
Time difference = 144 µs
Time difference = 144866 ns
705082704
	#3:	100000 elements - eager (max performance)
Time difference = 0 ms
Time difference = 10 µs
Time difference = 10969 ns
705082704

	#3:	1000000 elements - lazy
Time difference = 1 ms
Time difference = 1444 µs
Time difference = 1444325 ns
1784293664
	#3:	1000000 elements - eager (max performance)
Time difference = 0 ms
Time difference = 97 µs
Time difference = 97475 ns
1784293664

	#3:	10000000 elements - lazy
Time difference = 14 ms
Time difference = 14679 µs
Time difference = 14679608 ns
-2004260032
	#3:	10000000 elements - eager (max performance)
Time difference = 1 ms
Time difference = 1005 µs
Time difference = 1005247 ns
-2004260032

	#3:	100000000 elements - lazy
Time difference = 180 ms
Time difference = 180687 µs
Time difference = 180687806 ns
987459712
	#3:	100000000 elements - eager (max performance)
Time difference = 20 ms
Time difference = 20744 µs
Time difference = 20744136 ns
987459712

	#3:	1000000000 elements - lazy
Time difference = 1555 ms
Time difference = 1555279 µs
Time difference = 1555279487 ns
-243309312
	#3:	1000000000 elements - eager (max performance)
Time difference = 109 ms
Time difference = 109281 µs
Time difference = 109281638 ns
-243309312
```

As we can see, lazy sequences are up to 15 times slower (this implementation) than eager ones. 

The only exception is connected with those ones where additional storage was neccessary (allocation took the most of time there).

One of suggested (by me) solutions for enhancing lazy sequences is making them a bit less lazy: using bufferes, because

* That would reduce number of function calls
*(Every iteration makes `n` calls where `n` is the number of aplied functions* 
-> *As a result each call would be called not more often than once per `[buffer size]`).*
* It would become more cache-friendly, because each call would proceed more data in the one place.

---

###Update

Buffer tests added.

* As it was expected, performance increased due to using buffers.
* Details of benchmark are presented in `log.txt` and visualized in `Visualization.xlsx`.
* Output of running with valgrind memcheck is in `valgrind.txt` *(reduced version of test because of speed reducing of valgrind)*.

The test task was counting sum of numbers from `1` to `n` for different `n`s.
* For eager sequences code was:
```cpp
int res = 0;
for (int i = 0; i < n; ++i) {
    res += i;
}
```
* For lazy sequences different buffer sizes were tested *(details in the files)*.