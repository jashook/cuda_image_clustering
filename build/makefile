debug: $(FILES)
	gcc -pthread -g -I ../include -I ../lib/include ../src/*.c ../lib/src/*.c -o main -lpthread

debug-full: $(FILES)
	gcc -pthread -I ../include -I ../lib/include ../src/*.c ../lib/src/*.c -o main -lpthread

release: $(FILES)
	gcc -pthread -I ../include -I ../lib/include -O2 ../src/*.c ../lib/src/*.c -o main -lpthread

