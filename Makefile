
CC = nvcc
RUN = srun -n 1 --gres=gpu:1

particles_seq : particles_seq.cu
	$(CC) -o particles_seq particles_seq.cu

particles.exec : particles.cu
	$(CC) -o particles.exec particles.cu

run_particles : particles.exec
	$(RUN) particles.exec 32 1 100.0 0.01 1 32

run_test : particles.exec particles_seq
	$(RUN) particles.exec 32 1 10.0 0.01 2 32 > out1.txt
	$(RUN) particles_seq 32 1 10.0 0.01 2 32 > out2.txt
	diff -I '.*time.*' out1.txt out2.txt

run_performance : particles.exec
	$(RUN) particles.exec 1024 1 1.0 0.01 0 128
	$(RUN) particles.exec 2048 1 1.0 0.01 0 128
	$(RUN) particles.exec 4096 1 1.0 0.01 0 128
	$(RUN) particles.exec 8192 1 1.0 0.01 0 128
	$(RUN) particles.exec 16384 1 1.0 0.01 0 128
	$(RUN) particles.exec 32768 1 1.0 0.01 0 128
	$(RUN) particles.exec 65536 1 1.0 0.01 0 128

	$(RUN) particles.exec 16384 1 1.0 0.01 0 1
	$(RUN) particles.exec 16384 1 1.0 0.01 0 16
	$(RUN) particles.exec 16384 1 1.0 0.01 0 32
	$(RUN) particles.exec 16384 1 1.0 0.01 0 64
	$(RUN) particles.exec 16384 1 1.0 0.01 0 128
	$(RUN) particles.exec 16384 1 1.0 0.01 0 256
	$(RUN) particles.exec 16384 1 1.0 0.01 0 512
