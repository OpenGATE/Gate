#ifndef GATEHYBRIDFORCEDDETECTIONPROCESSTYPE_HH
#define GATEHYBRIDFORCEDDETECTIONPROCESSTYPE_HH

// Define all the processes handled by the hybrid forced detection actor.
// Pruimary is not a process so it is defined at the end to serve as an
// upper limit when looping on the processes.
typedef enum {
  COMPTON=0,
  RAYLEIGH,
  PHOTOELECTRIC,
  PRIMARY
} ProcessType;


#endif
