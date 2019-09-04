#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#ifndef NULL
#define NULL 0
#endif

#define PI 3.14159265358979323846
float TWO_PI = PI + PI;
float HALF_PI = PI / 2.0;
float ONE_AND_HALF_PI = PI + HALF_PI;

#define Tuple float2

int DEBUG = 0; // only for debug on host side
int h_N; // host N
char* grid = NULL; // only used for printing on host side

int d_N; // for use with the algorithm on gpu side


// GPU SIDE CODE ---
void apply_velocity(Tuple & p, Tuple & v, float t) {
  p.x = p.x + v.x * t;
  p.y = p.y + v.y * t;
}
void apply_boundary_collision(Tuple & p, Tuple & v, Tuple & outV) {
  outV.x = (p.x < 0 || p.x > d_N) ? -v.x : v.x;
  outV.y = (p.y < 0 || p.y > d_N) ? -v.y : v.y;
}
float distance(Tuple & p1, Tuple & p2) {
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;
  return (float)sqrt(dx*dx + dy*dy);
}

void apply_nbody_collision(Tuple & p1, Tuple & v1, Tuple & p2, Tuple & v2, Tuple & outV) {
  float dx = p2.x - p1.x;
  float dy = p2.y - p1.y;
  float distance = (float)sqrt(dx * dx + dy * dy);
  // Unit vector in the direction of the collision
  float ax = dx / distance;
  float ay = dy / distance;
  // Projection of the velocities in these axes
  float vb1 = (-v1.x * ay + v1.y * ax);
  float va2 = (v2.x * ax + v2.y * ay);
  // New velocity for v1 in these axes (after collision)
  outV.x = va2 * ax - vb1 * ay;
  outV.y = va2 * ay + vb1 * ax;// new vx,vy for particle 1 after collision
}

void update_particle_velocities_boundary_collisions(Tuple positions[], Tuple velocities[]) {
  int i;
  for (i = 0; i < d_N; i++) {
    // writes to second half of velocities array
    apply_boundary_collision(positions[i], velocities[i], velocities[i+d_N]);
  }
}

void update_particle_velocities_nbody_collisions(Tuple positions[], Tuple velocities[]) {
  int i, j;
  int collisions;
  for (i = 0; i < d_N; i++) {
    collisions = 0;
    for (j = 0; j < d_N; j++) {
      if (i != j && distance(positions[i], positions[j]) < 1.0) {
        // reads from second half of array and writes to first half
        apply_nbody_collision(positions[i], velocities[i+d_N], positions[j], velocities[j+d_N],
          velocities[i]);
        collisions++;
      }
    }
    if (collisions == 0) {
      // need to copy over velocity if there was no collision
      velocities[i] = velocities[i+d_N];
    }
  }
}

void update_particle_positions(Tuple positions[], Tuple velocities[], float t) {
  int i;
  for (i = 0; i < d_N; i++) {
    apply_velocity(positions[i], velocities[i], t);
  }
}
// -- END GPU SIDE CODE


// HOST SIDE CODE -----
int grid_index(int i, int j) {
  return (i*(h_N+1))+j;
}

void print(Tuple positions[]) {
  int i, j;
  Tuple p;
  int size = (h_N+2)*(h_N+2); // 1 square buffer on all sides

  if (grid == NULL) {
    grid = (char*)malloc(sizeof(char)*size);
  }
  for (i = 0; i < size; i++) {
    grid[i] = ' ';
  }
  for (i = 0; i < h_N; i++) {
    p = positions[i];
    grid[grid_index((int)p.x+1, (int)p.y)+1] = 'o';
  }
  for (i = 0; i < h_N+2; i++) {
    for (j = 0; j < h_N+2; j++) {
      printf("%c", grid[grid_index(i,j)]);
    }
    printf("\n");
  }
}
void output(Tuple positions[], float t, float step) {
  printf("At time %.2f\n", t);
  print(positions);
}

float rand_float(float range) {
  return range * (float)rand() / (float)RAND_MAX;
}

void simulate(Tuple positions[], Tuple velocities[], float step, int steps) {
  int i;
  float t = 0;
  for (i = 0; i < steps; i++) {
    if (DEBUG > 1) {
      output(positions, t, step);
      t+=step;
    }
    update_particle_velocities_boundary_collisions(positions, velocities);
    update_particle_velocities_nbody_collisions(positions, velocities);
    update_particle_positions(positions, velocities, step);
  }
}

float diff(timespec *start, timespec *stop) {
  float result = (stop->tv_sec - start->tv_sec);
  result += ((float)(stop->tv_nsec - start->tv_nsec)) / (float)1000000000.0;
  return result;
}

void init_random(Tuple positions[], Tuple velocities[]) {
  int i;
  float angle;
  for (i = 0; i < h_N; i++) {
    positions[i].x = (int)rand_float(h_N);
    positions[i].y = i;
    angle = rand_float(TWO_PI);
    velocities[i].x = (float)cos(angle);
    velocities[i].y = (float)sin(angle);
  }
}

int main(int argc, char** args) {
  DEBUG = 2;
  int seed = 0;
  float t = 100.0;
  float step = 0.01;
  int n = 10;
  if (argc > 1) {
    n = atoi(args[1]);
  }
  if (argc > 2) {
    seed = atoi(args[2]);
  }
  if (argc > 3) {
    t = atof(args[3]);
  }
  if (argc > 4) {
    step = atof(args[4]);
  }
  if (argc > 5) {
    DEBUG = atoi(args[5]);
  }

  if (seed > 0) {
    srand(seed);
  }

  int steps = (int)(t / step);

  h_N = n;
  d_N = n;

  Tuple * positions = (Tuple*)malloc(sizeof(Tuple)*h_N);
  Tuple * velocities = (Tuple*)malloc(sizeof(Tuple)*h_N*2);
  init_random(positions, velocities);

  timespec time1, time2;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

  simulate(positions, velocities, step, steps);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
  if (DEBUG > 0) {
    int offset = (steps % 2) * h_N;
    output(positions+offset, t, step);
  }
  printf("Total time was %.2f seconds\n", diff(&time1,&time2));
  free(positions);
  free(velocities);
  if (grid != NULL) {
    free(grid);
  }
  return 0;
}
// -- END HOST SIDE CODE
